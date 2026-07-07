import os
import numpy as np
import pandas as pd
from google.cloud import bigquery
from config import GCP_PROJECT, PHYSIONET_PROJECT, DATA_DIR, N_HOURS


VITAL_ITEMS = {
    220045: 'heart_rate',
    220179: 'sbp',        # NBP systolic (was 220050, ABP-only)
    220180: 'dbp',        # NBP diastolic (was 220051, ABP-only)
    220181: 'map',        # NBP mean (was 220052, ABP-only)
    220210: 'resp_rate',
    220277: 'spo2',
    223761: 'temp_f',
    220739: 'gcs_total',
    223835: 'fio2',       # was 220235, wrong itemid entirely
}

LAB_ITEMS = {
    51006: 'bun',
    50971: 'potassium',
    50983: 'sodium',
    50803: 'bicarbonate',
    51222: 'hemoglobin',
    51301: 'wbc',
    50885: 'bilirubin',
    50813: 'lactate',
    50862: 'albumin',
    50861: 'alt',
    50863: 'ast',
    51265: 'platelets',
    50912: 'creatinine_raw',  # kept only for slope computation later, not a direct feature
}

RANGE_FILTERS = {
    'heart_rate': (0, 300), 'sbp': (0, 300), 'dbp': (0, 200),
    'map': (0, 250), 'resp_rate': (0, 80), 'spo2': (50, 100),
    'temp_f': (80, 115), 'gcs_total': (3, 15), 'fio2': (21, 100),
}


def get_client():
    return bigquery.Client(project=GCP_PROJECT)


def load_cohort():
    cohort = pd.read_parquet(f"{DATA_DIR}cohort.parquet")
    cohort["intime"] = pd.to_datetime(cohort["intime"])
    return cohort


def query_chartevents(client, stay_ids, item_ids):
    query = f"""
    SELECT stay_id, itemid, charttime, valuenum
    FROM `{PHYSIONET_PROJECT}.mimiciv_3_1_icu.chartevents`
    WHERE stay_id IN UNNEST(@stay_ids)
      AND itemid IN UNNEST(@item_ids)
      AND valuenum IS NOT NULL
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("stay_ids", "INT64", stay_ids),
            bigquery.ArrayQueryParameter("item_ids", "INT64", item_ids),
        ]
    )
    return client.query(query, job_config=job_config).to_dataframe()


def query_labevents(client, stay_ids, item_ids, cohort):
    # labevents is keyed by hadm_id, not stay_id — join via cohort mapping
    hadm_ids = cohort.loc[cohort.stay_id.isin(stay_ids), "hadm_id"].tolist()

    query = f"""
    SELECT hadm_id, itemid, charttime, valuenum
    FROM `{PHYSIONET_PROJECT}.mimiciv_3_1_hosp.labevents`
    WHERE hadm_id IN UNNEST(@hadm_ids)
      AND itemid IN UNNEST(@item_ids)
      AND valuenum IS NOT NULL
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("hadm_ids", "INT64", hadm_ids),
            bigquery.ArrayQueryParameter("item_ids", "INT64", item_ids),
        ]
    )
    df = client.query(query, job_config=job_config).to_dataframe()

    # Map hadm_id -> stay_id (first stay per hadm_id, matches cohort's 1-stay-per-patient design)
    hadm_to_stay = cohort.set_index("hadm_id")["stay_id"].to_dict()
    df["stay_id"] = df["hadm_id"].map(hadm_to_stay)
    return df.drop(columns=["hadm_id"])


def clean_and_align_hours(df, item_map, cohort, n_hours=N_HOURS):
    df = df.copy()
    df["feature"] = df["itemid"].map(item_map)
    df = df.dropna(subset=["feature"])

    intime_map = cohort.set_index("stay_id")["intime"].to_dict()
    df["intime"] = df["stay_id"].map(intime_map)
    df["charttime"] = pd.to_datetime(df["charttime"])
    df["hour"] = ((df["charttime"] - df["intime"]).dt.total_seconds() / 3600).astype(int)

    df = df[(df["hour"] >= 0) & (df["hour"] < n_hours)]

    for feat, (lo, hi) in RANGE_FILTERS.items():
        mask = df["feature"] == feat
        df = df[~(mask & ((df["valuenum"] < lo) | (df["valuenum"] > hi)))]

    return df


def build_feature_matrix_batch(long_df, stay_ids_batch, features, n_hours=N_HOURS):
    """
    Build hour x feature matrix per patient, plus the missingness mask.
    Returns two DataFrames: values (wide) and mask (wide), both indexed by stay_id.
    """
    value_rows = {}
    mask_rows = {}

    for stay_id in stay_ids_batch:
        patient_df = long_df[long_df.stay_id == stay_id]

        pivot = (
            patient_df.groupby(["hour", "feature"])["valuenum"]
            .mean()
            .unstack("feature")
            .reindex(index=range(n_hours), columns=features)
        )

        # Missingness mask computed BEFORE imputation
        mask = (~pivot.isna()).astype(int)

        filled = pivot.ffill().bfill()

        value_flat = {}
        mask_flat = {}
        for feat in features:
            for h in range(n_hours):
                col = f"{feat}_h{h:02d}"
                value_flat[col] = filled.loc[h, feat] if not pd.isna(filled.loc[h, feat]) else 0.0
                mask_flat[f"{feat}_mask_h{h:02d}"] = mask.loc[h, feat]

        value_rows[stay_id] = value_flat
        mask_rows[stay_id] = mask_flat

    values_df = pd.DataFrame.from_dict(value_rows, orient="index")
    values_df.index.name = "stay_id"

    masks_df = pd.DataFrame.from_dict(mask_rows, orient="index")
    masks_df.index.name = "stay_id"

    return values_df, masks_df


def process_in_batches(client, cohort, item_map, features, query_fn, batch_size=2000):
    all_stay_ids = cohort.stay_id.tolist()
    item_ids = list(item_map.keys())

    value_batches = []
    mask_batches = []

    for i in range(0, len(all_stay_ids), batch_size):
        batch_ids = all_stay_ids[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1} "
              f"({i}-{min(i + batch_size, len(all_stay_ids))} of {len(all_stay_ids)})")

        raw = query_fn(client, batch_ids)
        raw = clean_and_align_hours(raw, item_map, cohort)

        values_df, masks_df = build_feature_matrix_batch(
            raw, batch_ids, list(item_map.values())
        )
        value_batches.append(values_df)
        mask_batches.append(masks_df)

    values = pd.concat(value_batches)
    masks = pd.concat(mask_batches)
    return values, masks


def main():
    client = get_client()
    cohort = load_cohort()

    # TEMPORARY — test on a small slice first, remove this line once verified
    #  cohort = cohort.head(50)

    print(f"Cohort loaded: {len(cohort)} patients")

    print("\n--- Extracting vitals ---")
    vitals_values, vitals_mask = process_in_batches(
        client, cohort, VITAL_ITEMS, list(VITAL_ITEMS.values()),
        lambda c, ids: query_chartevents(c, ids, list(VITAL_ITEMS.keys()))
    )

    print("\n--- Extracting labs ---")
    labs_values, labs_mask = process_in_batches(
        client, cohort, LAB_ITEMS, list(LAB_ITEMS.values()),
        lambda c, ids: query_labevents(c, ids, list(LAB_ITEMS.keys()), cohort)
    )

    os.makedirs(DATA_DIR, exist_ok=True)
    vitals_values.to_parquet(f"{DATA_DIR}vitals_values.parquet")
    vitals_mask.to_parquet(f"{DATA_DIR}vitals_mask.parquet")
    labs_values.to_parquet(f"{DATA_DIR}labs_values.parquet")
    labs_mask.to_parquet(f"{DATA_DIR}labs_mask.parquet")

    print(f"\nVitals matrix shape: {vitals_values.shape}")
    print(f"Labs matrix shape: {labs_values.shape}")
    print("Saved to data/")


if __name__ == "__main__":
    main()