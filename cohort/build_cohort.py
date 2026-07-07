import json
import pandas as pd
from sqlalchemy import create_engine
from config import MIMIC_DB, DATA_DIR


def build_cohort(engine):
    with open("sql/cohort.sql", "r") as f:
        query = f.read()

    cohort = pd.read_sql(query, engine)
    cohort.to_parquet(f"{DATA_DIR}cohort_base.parquet", index=False)

    print(f"Cohort size: {len(cohort)}")
    return cohort


def label_aki(engine, cohort, obs_hours=24, outcome_hours=48):
    query = """
    SELECT
      k.stay_id,
      k.charttime,
      k.aki_stage_creat,
      k.aki_stage_uo,
      GREATEST(
        COALESCE(k.aki_stage_creat, 0),
        COALESCE(k.aki_stage_uo, 0)
      ) AS aki_stage
    FROM mimiciv_derived.kdigo_stages k
    """

    kdigo = pd.read_sql(query, engine)

    results = []

    for _, row in cohort.iterrows():
        obs_end = row.intime + pd.Timedelta(hours=obs_hours)
        outcome_end = obs_end + pd.Timedelta(hours=outcome_hours)

        obs_window = kdigo[
            (kdigo.stay_id == row.stay_id)
            & (kdigo.charttime >= row.intime)
            & (kdigo.charttime < obs_end)
        ]

        if (obs_window.aki_stage > 0).any():
            continue

        outcome_window = kdigo[
            (kdigo.stay_id == row.stay_id)
            & (kdigo.charttime >= obs_end)
            & (kdigo.charttime < outcome_end)
        ]

        aki_label = int((outcome_window.aki_stage > 0).any())
        aki_stage = int(outcome_window.aki_stage.max()) if aki_label else 0
        aki_onset = (
            outcome_window[outcome_window.aki_stage > 0].charttime.min()
            if aki_label
            else pd.NaT
        )

        results.append({
            **row.to_dict(),
            "aki_label": aki_label,
            "aki_stage": aki_stage,
            "aki_onset": aki_onset,
        })

    return pd.DataFrame(results)


def flag_competing_risk(cohort_labeled, engine):
    deaths = pd.read_sql("""
        SELECT hadm_id, deathtime
        FROM mimiciv_hosp.admissions
        WHERE deathtime IS NOT NULL
    """, engine)

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
    engine = create_engine(MIMIC_DB)

    cohort = build_cohort(engine)

    cohort_labeled = label_aki(engine, cohort)
    cohort_labeled = flag_competing_risk(cohort_labeled, engine)

    cohort_labeled.to_parquet(f"{DATA_DIR}cohort.parquet", index=False)

    print(cohort_labeled.aki_label.value_counts())
    print(f"Competing risk patients: {cohort_labeled.competing_risk.sum()}")

    train_ids, calib_ids, test_ids = temporal_split(cohort_labeled)

    json.dump(train_ids, open(f"{DATA_DIR}train_ids.json", "w"))
    json.dump(calib_ids, open(f"{DATA_DIR}calib_ids.json", "w"))
    json.dump(test_ids, open(f"{DATA_DIR}test_ids.json", "w"))


if __name__ == "__main__":
    main()