import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from collections import defaultdict
from config import DATA_DIR, RANDOM_SEED, N_ESTIMATORS, LEARNING_RATE, NUM_LEAVES, N_HOURS, MASK_PERCENTILE


def load_splits():
    features = pd.read_parquet(f"{DATA_DIR}features_full.parquet")
    train_ids = json.load(open(f"{DATA_DIR}train_ids.json"))
    calib_ids = json.load(open(f"{DATA_DIR}calib_ids.json"))
    test_ids  = json.load(open(f"{DATA_DIR}test_ids.json"))

    X_train = features.loc[features.index.isin(train_ids)].drop(columns=["aki_label", "aki_stage"])
    y_train = features.loc[features.index.isin(train_ids)]["aki_label"]

    X_calib = features.loc[features.index.isin(calib_ids)].drop(columns=["aki_label", "aki_stage"])
    y_calib = features.loc[features.index.isin(calib_ids)]["aki_label"]

    X_test = features.loc[features.index.isin(test_ids)].drop(columns=["aki_label", "aki_stage"])
    y_test = features.loc[features.index.isin(test_ids)]["aki_label"]

    print(f"Train: {X_train.shape} | Calib: {X_calib.shape} | Test: {X_test.shape}")
    return X_train, y_train, X_calib, y_calib, X_test, y_test


def train_pilot_model(X, y):
    model = lgb.LGBMClassifier(
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        num_leaves=NUM_LEAVES,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X, y)
    return model


def evaluate_pilot(model, X, y, split_name="Calibration"):
    from sklearn.metrics import average_precision_score, roc_auc_score
    probs = model.predict_proba(X)[:, 1]
    auprc = average_precision_score(y, probs)
    auroc = roc_auc_score(y, probs)
    print(f"{split_name} AUPRC: {auprc:.4f} | AUROC: {auroc:.4f}")
    return auprc, auroc


def compute_shap_mask(model, X_calib, n_hours=N_HOURS, percentile=MASK_PERCENTILE):
    """
    Compute SHAP values on the calibration set ONLY — never on train or test.
    This is the calibration-set-only design from the plan: the mask is
    derived independently of the data used to train or evaluate the final model,
    preventing the mask itself from leaking information about test performance.
    """
    print("\nComputing SHAP values on calibration set...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_calib)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # class 1 (positive/AKI) for binary classification

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_names = X_calib.columns.tolist()

    # Group columns by base feature name (strip the _hXX suffix) to build the 2D heatmap matrix
    feature_hour_shap = defaultdict(dict)
    for i, col in enumerate(feature_names):
        if "_h" in col:
            parts = col.rsplit("_h", 1)
            base_feat, hour_str = parts[0], parts[1]
            try:
                hour = int(hour_str)
                feature_hour_shap[base_feat][hour] = mean_abs_shap[i]
            except ValueError:
                # Column doesn't end in a valid hour number (e.g., weight_used) — skip from heatmap
                continue

    base_features = sorted(feature_hour_shap.keys())
    importance_matrix = np.zeros((len(base_features), n_hours))
    for fi, feat in enumerate(base_features):
        for h in range(n_hours):
            importance_matrix[fi, h] = feature_hour_shap[feat].get(h, 0.0)

    threshold = np.percentile(mean_abs_shap, percentile)
    mask = (mean_abs_shap >= threshold).astype(int)

    n_active = mask.sum()
    print(f"SHAP threshold ({percentile}th percentile): {threshold:.6f}")
    print(f"Active feature-hour cells: {n_active} / {len(mask)} ({100*n_active/len(mask):.1f}%)")

    return mask, importance_matrix, base_features, feature_names


def main():
    X_train, y_train, X_calib, y_calib, X_test, y_test = load_splits()

    print("\nTraining pilot model (for SHAP computation only, not the final model)...")
    pilot_model = train_pilot_model(X_train, y_train)

    evaluate_pilot(pilot_model, X_calib, y_calib, "Calibration")

    mask, importance_matrix, base_features, feature_names = compute_shap_mask(pilot_model, X_calib)

    os.makedirs(f"{DATA_DIR}shap_mask", exist_ok=True)
    np.save(f"{DATA_DIR}shap_mask/shap_mask.npy", mask)
    np.save(f"{DATA_DIR}shap_mask/importance_matrix.npy", importance_matrix)
    json.dump(base_features, open(f"{DATA_DIR}shap_mask/base_features.json", "w"))
    json.dump(feature_names, open(f"{DATA_DIR}shap_mask/feature_names.json", "w"))

    print(f"\nSaved mask and importance matrix to {DATA_DIR}shap_mask/")
    print(f"Base features tracked in heatmap: {len(base_features)}")


if __name__ == "__main__":
    main()