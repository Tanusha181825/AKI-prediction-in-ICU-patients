import pandas as pd
from config import DATA_DIR

pd.set_option("display.max_columns", 10)
pd.set_option("display.width", 120)

vitals_values = pd.read_parquet(f"{DATA_DIR}vitals_values.parquet")
vitals_mask = pd.read_parquet(f"{DATA_DIR}vitals_mask.parquet")
labs_values = pd.read_parquet(f"{DATA_DIR}labs_values.parquet")
labs_mask = pd.read_parquet(f"{DATA_DIR}labs_mask.parquet")

print("=== VITALS VALUES (first patient, heart_rate across 24h) ===")
hr_cols = [c for c in vitals_values.columns if c.startswith("heart_rate_h")]
print(vitals_values.iloc[0][hr_cols])

print("\n=== VITALS MASK (same patient, same feature) — 1=observed, 0=imputed ===")
hr_mask_cols = [c for c in vitals_mask.columns if c.startswith("heart_rate_mask_h")]
print(vitals_mask.iloc[0][hr_mask_cols])

print("\n=== Overall vitals value ranges (sanity check) ===")
print(vitals_values.describe().T[["min", "max", "mean"]].head(20))

print("\n=== Overall mask fill rate (% of hours with real data, per feature) ===")
print(vitals_mask.mean().groupby(lambda c: c.split("_mask_h")[0]).mean())

print("\n=== LABS spot check — creatinine_raw for first patient ===")
cr_cols = [c for c in labs_values.columns if c.startswith("creatinine_raw_h")]
print(labs_values.iloc[0][cr_cols])
