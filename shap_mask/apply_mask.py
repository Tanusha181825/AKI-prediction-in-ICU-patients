import os
import json
import numpy as np
import pandas as pd
from config import DATA_DIR


def load_full_splits():
    features = pd.read_parquet(f"{DATA_DIR}features_full.parquet")
    train_ids = json.load(open(f"{DATA_DIR}train_ids.json"))
    calib_ids = json.load(open(f"{DATA_DIR}calib_ids.json"))
    test_ids  = json.load(open(f"{DATA_DIR}test_ids.json"))

    X_train = features.loc[features.index.isin(train_ids)].drop(columns=["aki_label", "aki_stage"])
    X_calib = features.loc[features.index.isin(calib_ids)].drop(columns=["aki_label", "aki_stage"])
    X_test  = features.loc[features.index.isin(test_ids)].drop(columns=["aki_label", "aki_stage"])

    return X_train, X_calib, X_test


def apply_mask(X, mask):
    assert len(mask) == X.shape[1], f"Mask length {len(mask)} != feature count {X.shape[1]}"
    selected_cols = np.where(mask == 1)[0]
    return X.iloc[:, selected_cols]


def main():
    X_train, X_calib, X_test = load_full_splits()

    mask = np.load(f"{DATA_DIR}shap_mask/shap_mask.npy")
    print(f"Loaded mask: {mask.sum()} active out of {len(mask)} total features")

    X_train_c = apply_mask(X_train, mask)
    X_calib_c = apply_mask(X_calib, mask)
    X_test_c  = apply_mask(X_test, mask)

    print(f"Compressed shapes — Train: {X_train_c.shape} | Calib: {X_calib_c.shape} | Test: {X_test_c.shape}")

    X_train_c.to_parquet(f"{DATA_DIR}X_train_compressed.parquet")
    X_calib_c.to_parquet(f"{DATA_DIR}X_calib_compressed.parquet")
    X_test_c.to_parquet(f"{DATA_DIR}X_test_compressed.parquet")

    print(f"Saved compressed matrices to {DATA_DIR}")


if __name__ == "__main__":
    main()