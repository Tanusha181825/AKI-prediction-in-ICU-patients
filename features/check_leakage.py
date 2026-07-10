import pandas as pd
from config import DATA_DIR

features = pd.read_parquet(f"{DATA_DIR}features_full.parquet")

# Check correlation for the last few hours of UO against the label
for h in [20, 21, 22, 23]:
    col = f"uo_rolling4h_h{h:02d}"
    corr = features[col].corr(features["aki_label"])
    print(f"Correlation between {col} and aki_label: {corr:.4f}")

# Also check the oliguric binary flag, since that's a more direct KDIGO-style signal
for h in [20, 21, 22, 23]:
    col = f"uo_oliguric_h{h:02d}"
    corr = features[col].corr(features["aki_label"])
    print(f"Correlation between {col} and aki_label: {corr:.4f}")