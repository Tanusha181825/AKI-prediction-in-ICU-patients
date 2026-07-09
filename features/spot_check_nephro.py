import pandas as pd
from config import DATA_DIR

pd.set_option("display.max_columns", 10)
pd.set_option("display.width", 120)

neph = pd.read_parquet(f"{DATA_DIR}nephrotoxin_features.parquet")
traj = pd.read_parquet(f"{DATA_DIR}trajectory_features.parquet")

print("=== Weight used (real vs fallback) ===")
print(traj["weight_used"].describe())
print(f"\nPatients at exactly 70.0 (fallback): {(traj['weight_used'] == 70.0).sum()} / {len(traj)}")

print("\n=== Nephrotoxin binary flags — any drug activity at all? ===")
binary_cols = [c for c in neph.columns if "_binary_h" in c]
print(f"Total binary-flag columns: {len(binary_cols)}")
print(f"Sum across all patients/hours (0 = no drugs matched at all): {neph[binary_cols].sum().sum()}")

print("\n=== Vancomycin binary flags, first patient ===")
vanc_cols = [c for c in neph.columns if c.startswith("vancomycin_binary_h")]
print(neph.iloc[0][vanc_cols])

print("\n=== Creatinine slope, first patient ===")
slope_cols = [c for c in traj.columns if c.startswith("creat_slope_h")]
print(traj.iloc[0][slope_cols])

print("\n=== UO oliguric flag rate (overall) ===")
oliguric_cols = [c for c in traj.columns if c.startswith("uo_oliguric_h")]
print(traj[oliguric_cols].mean().mean())