import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from config import DATA_DIR, RANDOM_SEED


def load_splits():
    features = pd.read_parquet(f"{DATA_DIR}features_full.parquet")
    train_ids = json.load(open(f"{DATA_DIR}train_ids.json"))
    test_ids  = json.load(open(f"{DATA_DIR}test_ids.json"))

    X_train = features.loc[features.index.isin(train_ids)].drop(columns=["aki_label", "aki_stage"])
    y_train = features.loc[features.index.isin(train_ids)]["aki_label"]

    X_test = features.loc[features.index.isin(test_ids)].drop(columns=["aki_label", "aki_stage"])
    y_test = features.loc[features.index.isin(test_ids)]["aki_label"]

    return X_train, y_train, X_test, y_test


def apply_mask(X, mask):
    return X.iloc[:, np.where(mask == 1)[0]]


def main():
    X_train, y_train, X_test, y_test = load_splits()

    mask = np.load(f"{DATA_DIR}shap_mask/frozen_mask.npy")
    best_params = json.load(open(f"{DATA_DIR}models/best_hyperparams.json"))

    print(f"Using frozen mask: {mask.sum()} / {len(mask)} features")
    print(f"Using tuned hyperparameters: {best_params}")

    X_train_c = apply_mask(X_train, mask)
    X_test_c = apply_mask(X_test, mask)

    print(f"\nTraining FINAL MODEL A on all {len(X_train_c)} training patients...")
    model = lgb.LGBMClassifier(
        **best_params,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train_c, y_train)

    # Single, final evaluation on the held-out test set — never touched before now
    test_probs = model.predict_proba(X_test_c)[:, 1]
    auprc = average_precision_score(y_test, test_probs)
    auroc = roc_auc_score(y_test, test_probs)
    brier = brier_score_loss(y_test, test_probs)

    print(f"\n=== FINAL MODEL A — Test Set Results (single evaluation) ===")
    print(f"AUPRC: {auprc:.4f}")
    print(f"AUROC: {auroc:.4f}")
    print(f"Brier Score: {brier:.4f}")

    model.booster_.save_model(f"{DATA_DIR}models/final_model_A.txt")
    np.save(f"{DATA_DIR}models/final_model_A_test_probs.npy", test_probs)

    results = {
        "auprc": float(auprc), "auroc": float(auroc), "brier": float(brier),
        "n_train": len(X_train_c), "n_test": len(X_test_c),
        "n_features": int(mask.sum()), "hyperparams": best_params,
    }
    json.dump(results, open(f"{DATA_DIR}models/final_model_A_results.json", "w"))
    print(f"\nModel and results saved to {DATA_DIR}models/")


if __name__ == "__main__":
    main()