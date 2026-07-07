import json
import os
import pandas as pd
from google.cloud import bigquery
from config import GCP_PROJECT, PHYSIONET_PROJECT, DATA_DIR


def get_client():
    return bigquery.Client(project=GCP_PROJECT)


def build_cohort(client):
    with open("sql/cohort.sql", "r") as f:
        query = f.read()

    cohort = client.query(query).to_dataframe()

    os.makedirs(DATA_DIR, exist_ok=True)
    cohort.to_parquet(f"{DATA_DIR}cohort_base.parquet", index=False)

    print(f"Cohort size: {len(cohort)}")
    return cohort


def label_aki(client, cohort, obs_hours=24, outcome_hours=48):
    query = f"""
    SELECT
      k.stay_id,
      k.charttime,
      k.aki_stage_creat,
      k.aki_stage_uo,
      GREATEST(
        COALESCE(k.aki_stage_creat, 0),
        COALESCE(k.aki_stage_uo, 0)
      ) AS aki_stage
    FROM `{PHYSIONET_PROJECT}.mimiciv_3_1_derived.kdigo_stages` k
    """

    kdigo = client.query(query).to_dataframe()

    # Ensure timestamps are proper datetimes for comparison
    cohort = cohort.copy()
    cohort["intime"] = pd.to_datetime(cohort["intime"])
    cohort["outtime"] = pd.to_datetime(cohort["outtime"])
    kdigo["charttime"] = pd.to_datetime(kdigo["charttime"])

    cohort["obs_end"] = cohort["intime"] + pd.Timedelta(hours=obs_hours)
    cohort["outcome_end"] = cohort["obs_end"] + pd.Timedelta(hours=outcome_hours)

    # Merge kdigo onto cohort by stay_id (vectorized, not a per-row loop)
    merged = cohort.merge(kdigo, on="stay_id", how="left")

    # Flag rows that fall inside the observation window (pre-existing AKI)
    in_obs_window = (
        (merged.charttime >= merged.intime)
        & (merged.charttime < merged.obs_end)
        & (merged.aki_stage > 0)
    )

    # Flag rows that fall inside the outcome window (the label we want)
    in_outcome_window = (
        (merged.charttime >= merged.obs_end)
        & (merged.charttime < merged.outcome_end)
        & (merged.aki_stage > 0)
    )

    # Per-stay_id: did they already have AKI in the observation window? -> exclude
    excluded_stays = merged.loc[in_obs_window, "stay_id"].unique()

    # Per-stay_id: aggregate outcome window results
    outcome_rows = merged[in_outcome_window]
    agg = outcome_rows.groupby("stay_id").agg(
        aki_stage=("aki_stage", "max"),
        aki_onset=("charttime", "min"),
    )

    cohort = cohort[~cohort.stay_id.isin(excluded_stays)].copy()
    cohort = cohort.merge(agg, on="stay_id", how="left")

    cohort["aki_label"] = cohort["aki_stage"].notna().astype(int)
    cohort["aki_stage"] = cohort["aki_stage"].fillna(0).astype(int)
    # aki_onset stays NaT where aki_label == 0

    cohort = cohort.drop(columns=["obs_end", "outcome_end"])

    return cohort


def flag_competing_risk(cohort_labeled, client):
    query = f"""
        SELECT hadm_id, deathtime
        FROM `{PHYSIONET_PROJECT}.mimiciv_3_1_hosp.admissions`
        WHERE deathtime IS NOT NULL
    """
    deaths = client.query(query).to_dataframe()
    deaths["deathtime"] = pd.to_datetime(deaths["deathtime"])

    cohort_labeled = cohort_labeled.merge(deaths, on="hadm_id", how="left")

    obs_end = cohort_labeled.intime + pd.Timedelta(hours=24)
    outcome_end = obs_end + pd.Timedelta(hours=48)

    cohort_labeled["competing_risk"] = (
        cohort_labeled.deathtime.notna()
        & (cohort_labeled.deathtime >= obs_end)
        & (cohort_labeled.deathtime < outcome_end)
        & (cohort_labeled.aki_label == 0)
    ).astype(int)

    return cohort_labeled


def temporal_split(cohort, train_frac=0.60, calib_frac=0.20):
    df = cohort[cohort.competing_risk == 0].sort_values("intime").reset_index(drop=True)

    n = len(df)
    n_train = int(n * train_frac)
    n_calib = int(n * calib_frac)

    train_ids = df.iloc[:n_train].stay_id.tolist()
    calib_ids = df.iloc[n_train:n_train + n_calib].stay_id.tolist()
    test_ids = df.iloc[n_train + n_calib:].stay_id.tolist()

    print(f"Train: {len(train_ids)} | Calibration: {len(calib_ids)} | Test: {len(test_ids)}")
    print(f"Train AKI rate: {df.iloc[:n_train].aki_label.mean():.3f}")
    print(f"Test AKI rate:  {df.iloc[n_train + n_calib:].aki_label.mean():.3f}")

    return train_ids, calib_ids, test_ids


def main():
    client = get_client()

    cohort = build_cohort(client)

    cohort_labeled = label_aki(client, cohort)
    cohort_labeled = flag_competing_risk(cohort_labeled, client)

    os.makedirs(DATA_DIR, exist_ok=True)
    cohort_labeled.to_parquet(f"{DATA_DIR}cohort.parquet", index=False)

    print(cohort_labeled.aki_label.value_counts())
    print(f"Competing risk patients: {cohort_labeled.competing_risk.sum()}")

    train_ids, calib_ids, test_ids = temporal_split(cohort_labeled)

    json.dump(train_ids, open(f"{DATA_DIR}train_ids.json", "w"))
    json.dump(calib_ids, open(f"{DATA_DIR}calib_ids.json", "w"))
    json.dump(test_ids, open(f"{DATA_DIR}test_ids.json", "w"))


if __name__ == "__main__":
    main()