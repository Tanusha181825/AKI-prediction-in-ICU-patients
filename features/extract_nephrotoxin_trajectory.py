import os
import numpy as np
import pandas as pd
from google.cloud import bigquery
from config import GCP_PROJECT, PHYSIONET_PROJECT, DATA_DIR, N_HOURS


NEPHROTOXIN_CLASSES = {
    'vancomycin':     ['vancomycin'],
    'aminoglycoside': ['gentamicin', 'tobramycin', 'amikacin'],
    'contrast':       ['iohexol', 'ioversol', 'iopamidol', 'gadolinium'],
    'nsaid':          ['ibuprofen', 'ketorolac', 'naproxen', 'indomethacin'],
    'ace_inhibitor':  ['lisinopril', 'enalapril', 'captopril', 'ramipril'],
    'arb':            ['losartan', 'valsartan', 'irbesartan', 'olmesartan'],
    'diuretic':       ['furosemide', 'torsemide', 'bumetanide'],
    'calcineurin':    ['tacrolimus', 'cyclosporine'],
    'antifungal':     ['amphotericin', 'fluconazole', 'voriconazole'],
    'antiviral':      ['acyclovir', 'ganciclovir', 'cidofovir'],
}


def get_client():
    return bigquery.Client(project=GCP_PROJECT)


def load_cohort():
    cohort = pd.read_parquet(f"{DATA_DIR}cohort.parquet")
    cohort["intime"] = pd.to_datetime(cohort["intime"])
    return cohort


def query_inputevents(client, stay_ids):
    query = f"""
    SELECT ie.stay_id, ie.starttime, ie.amount, ie.amountuom,
           LOWER(d.label) AS drug_name
    FROM `{PHYSIONET_PROJECT}.mimiciv_3_1_icu.inputevents` ie
    JOIN `{PHYSIONET_PROJECT}.mimiciv_3_1_icu.d_items` d
      ON ie.itemid = d.itemid
    WHERE ie.stay_id IN UNNEST(@stay_ids)
      AND ie.amount > 0
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("stay_ids", "INT64", stay_ids)]
    )
    return client.query(query, job_config=job_config).to_dataframe()


def query_prescriptions(client, stay_ids, cohort):
    hadm_ids = cohort.loc[cohort.stay_id.isin(stay_ids), "hadm_id"].tolist()

    query = f"""
    SELECT p.hadm_id, p.starttime, p.dose_val_rx AS amount,
           p.dose_unit_rx AS amountuom, LOWER(p.drug) AS drug_name
    FROM `{PHYSIONET_PROJECT}.mimiciv_3_1_hosp.prescriptions` p
    WHERE p.hadm_id IN UNNEST(@hadm_ids)
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("hadm_ids", "INT64", hadm_ids)]
    )
    df = client.query(query, job_config=job_config).to_dataframe()

    hadm_to_stay = cohort.set_index("hadm_id")["stay_id"].to_dict()
    df["stay_id"] = df["hadm_id"].map(hadm_to_stay)
    return df.drop(columns=["hadm_id"])


def query_urine_output(client, stay_ids):
    query = f"""
    SELECT stay_id, charttime, value AS amount
    FROM `{PHYSIONET_PROJECT}.mimiciv_3_1_icu.outputevents`
    WHERE stay_id IN UNNEST(@stay_ids)
      AND value IS NOT NULL
      AND value >= 0
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("stay_ids", "INT64", stay_ids)]
    )
    return client.query(query, job_config=job_config).to_dataframe()


def query_weight(client, stay_ids):
    """
    226512 = Admission Weight (Kg), 224639 = Daily Weight (Kg).
    Bounded to plausible adult range (20-300 kg) to exclude charting errors.
    """
    query = f"""
    SELECT stay_id, itemid, charttime, valuenum
    FROM `{PHYSIONET_PROJECT}.mimiciv_3_1_icu.chartevents`
    WHERE stay_id IN UNNEST(@stay_ids)
      AND itemid IN (226512, 224639)
      AND valuenum IS NOT NULL
      AND valuenum BETWEEN 20 AND 300
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("stay_ids", "INT64", stay_ids)]
    )
    return client.query(query, job_config=job_config).to_dataframe()


def build_weight_map(weight_df):
    """
    Per stay_id: prefer admission weight (226512); if absent, use the
    earliest daily weight (224639). Falls back to population median
    (70kg) only if no weight was ever charted for that stay.
    """
    weight_df = weight_df.copy()
    weight_df["charttime"] = pd.to_datetime(weight_df["charttime"])

    admission = weight_df[weight_df.itemid == 226512].sort_values("charttime")
    daily = weight_df[weight_df.itemid == 224639].sort_values("charttime")

    admission_map = admission.groupby("stay_id")["valuenum"].first().to_dict()
    daily_map = daily.groupby("stay_id")["valuenum"].first().to_dict()

    def resolve(stay_id):
        if stay_id in admission_map:
            return admission_map[stay_id]
        if stay_id in daily_map:
            return daily_map[stay_id]
        return 70.0  # population fallback, only when truly no weight recorded

    return resolve


def get_drug_class(drug_name):
    if pd.isna(drug_name):
        return None
    for cls, drugs in NEPHROTOXIN_CLASSES.items():
        if any(d in drug_name for d in drugs):
            return cls
    return None


def align_hours(df, time_col, cohort, n_hours=N_HOURS):
    df = df.copy()
    intime_map = cohort.set_index("stay_id")["intime"].to_dict()
    df["intime"] = df["stay_id"].map(intime_map)
    df[time_col] = pd.to_datetime(df[time_col])

    # Drop rows where we couldn't resolve a valid time (bad stay_id mapping, null starttime, etc.)
    df = df.dropna(subset=[time_col, "intime"])

    df["hour"] = ((df[time_col] - df["intime"]).dt.total_seconds() / 3600).astype(int)
    return df[(df["hour"] >= 0) & (df["hour"] < n_hours)]


def build_nephrotoxin_features(meds_batch, stay_ids_batch, n_hours=N_HOURS):
    records = []
    for stay_id in stay_ids_batch:
        patient_meds = meds_batch[meds_batch.stay_id == stay_id]
        record = {"stay_id": stay_id}

        for cls in NEPHROTOXIN_CLASSES:
            cls_meds = patient_meds[patient_meds.drug_class == cls].sort_values("hour")

            for h in range(n_hours):
                given_up_to_h = cls_meds[cls_meds.hour <= h]
                given_this_hour = cls_meds[cls_meds.hour == h]

                binary = int(len(given_this_hour) > 0)
                cum_dose = given_up_to_h.amount.fillna(1.0).sum()

                if len(given_up_to_h) > 0:
                    hours_since_first = h - given_up_to_h.hour.min()
                    hours_since_last = h - given_up_to_h.hour.max()
                else:
                    hours_since_first = 0
                    hours_since_last = 999

                record[f"{cls}_binary_h{h:02d}"] = binary
                record[f"{cls}_cum_dose_h{h:02d}"] = cum_dose
                record[f"{cls}_h_since_first_h{h:02d}"] = hours_since_first
                record[f"{cls}_h_since_last_h{h:02d}"] = hours_since_last

        records.append(record)

    return pd.DataFrame(records).set_index("stay_id")


def build_trajectory_features(creatinine_df, uo_df, stay_ids_batch, weight_resolver, n_hours=N_HOURS):
    records = []
    for stay_id in stay_ids_batch:
        cr = creatinine_df[creatinine_df.stay_id == stay_id].sort_values("hour")
        uo = uo_df[uo_df.stay_id == stay_id].sort_values("hour")
        weight = weight_resolver(stay_id)

        record = {"stay_id": stay_id, "weight_used": weight}

        cr_hourly = cr.set_index("hour")["valuenum"].reindex(range(n_hours))
        cr_hourly = cr_hourly.infer_objects(copy=False).interpolate()

        prev_slope = 0.0
        for h in range(n_hours):
            window_start = max(0, h - 5)
            window = cr_hourly.iloc[window_start:h + 1].dropna()
            if len(window) >= 2:
                slope = np.polyfit(window.index.values, window.values, 1)[0]
            else:
                slope = 0.0
            accel = slope - prev_slope
            prev_slope = slope

            record[f"creat_slope_h{h:02d}"] = slope
            record[f"creat_accel_h{h:02d}"] = accel

        uo_hourly = uo.groupby("hour")["amount"].sum().reindex(range(n_hours)).fillna(0)
        for h in range(n_hours):
            window_start = max(0, h - 3)
            uo_4h = uo_hourly.iloc[window_start:h + 1].sum()
            oliguric = int((uo_4h / (weight * 4)) < 0.5)

            record[f"uo_rolling4h_h{h:02d}"] = uo_4h
            record[f"uo_oliguric_h{h:02d}"] = oliguric

        records.append(record)

    return pd.DataFrame(records).set_index("stay_id")


def load_creatinine_long(cohort):
    """
    Rebuild a long-format (stay_id, hour, valuenum) creatinine series
    from labs_values.parquet, since that file is wide-format (one column per hour).
    """
    labs_values = pd.read_parquet(f"{DATA_DIR}labs_values.parquet")
    cr_cols = [c for c in labs_values.columns if c.startswith("creatinine_raw_h")]

    long_rows = []
    for stay_id, row in labs_values[cr_cols].iterrows():
        for col in cr_cols:
            hour = int(col.split("_h")[-1])
            long_rows.append({"stay_id": stay_id, "hour": hour, "valuenum": row[col]})

    return pd.DataFrame(long_rows)


def process_in_batches(client, cohort, creatinine_full, batch_size=2000):
    all_stay_ids = cohort.stay_id.tolist()

    nephrotoxin_batches = []
    trajectory_batches = []

    for i in range(0, len(all_stay_ids), batch_size):
        batch_ids = all_stay_ids[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1} "
              f"({i}-{min(i + batch_size, len(all_stay_ids))} of {len(all_stay_ids)})")

        # --- Nephrotoxins ---
        iv_meds = query_inputevents(client, batch_ids)
        rx_meds = query_prescriptions(client, batch_ids, cohort)
        meds = pd.concat([iv_meds, rx_meds], ignore_index=True)
        meds["amount"] = pd.to_numeric(meds["amount"], errors="coerce")  # <-- ADD THIS LINE
        meds["drug_class"] = meds["drug_name"].apply(get_drug_class)
        meds = meds[meds["drug_class"].notna()]
        meds = align_hours(meds, "starttime", cohort)

        neph_df = build_nephrotoxin_features(meds, batch_ids)
        nephrotoxin_batches.append(neph_df)

        # --- Trajectory: weight + creatinine + urine output ---
        weight_df = query_weight(client, batch_ids)
        weight_resolver = build_weight_map(weight_df)

        uo_raw = query_urine_output(client, batch_ids)
        uo_raw = align_hours(uo_raw, "charttime", cohort)

        cr_batch = creatinine_full[creatinine_full.stay_id.isin(batch_ids)]

        traj_df = build_trajectory_features(cr_batch, uo_raw, batch_ids, weight_resolver)
        trajectory_batches.append(traj_df)

    nephrotoxin = pd.concat(nephrotoxin_batches)
    trajectory = pd.concat(trajectory_batches)
    return nephrotoxin, trajectory


def main():
    client = get_client()
    cohort = load_cohort()

    # TEMPORARY — test on a small slice first, remove once verified
    #  cohort = cohort.head(50)

    print(f"Cohort loaded: {len(cohort)} patients")
    print("Rebuilding creatinine long-format series from labs_values.parquet...")
    creatinine_full = load_creatinine_long(cohort)

    print("\n--- Extracting nephrotoxin + trajectory features ---")
    nephrotoxin, trajectory = process_in_batches(client, cohort, creatinine_full)

    os.makedirs(DATA_DIR, exist_ok=True)
    nephrotoxin.to_parquet(f"{DATA_DIR}nephrotoxin_features.parquet")
    trajectory.to_parquet(f"{DATA_DIR}trajectory_features.parquet")

    print(f"\nNephrotoxin matrix shape: {nephrotoxin.shape}")
    print(f"Trajectory matrix shape: {trajectory.shape}")
    print("Saved to data/")


if __name__ == "__main__":
    main()