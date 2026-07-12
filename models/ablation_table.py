import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score
from config import DATA_DIR, RANDOM_SEED, N_ESTIMATORS, LEARNING_RATE, NUM_LEAVES


NEPHROTOXIN_CLASSES = [
    "vancomycin", "aminoglycoside", "contrast", "nsaid", "ace_inhibitor",
    "arb", "diuretic", "calcineurin", "antifungal", "antiviral",
]
TRAJECTORY_PREFIXES = ["creat_slope_h", "creat_accel_h", "uo_rolling4h_h", "uo_oliguric_h", "weight_used"]


def classify_column(col):
    """
    Classifies each feature column as 'novel' (nephrotoxin or trajectory —
    Contribution 1) or 'baseline' (vitals/labs, replicating the original
    paper's feature set) based on naming convention.
    """
    for cls in NEPHROTOXIN_CLASSES:
        if col.startswith(f"{cls}_binary_h") or col.startswith(f"{cls}_cum_dose_h") \
           or col.startswith(f"{cls}_h_since_first_h") or col.startswith(f"{cls}_h_since_last_h"):
            return "novel"
    for prefix in TRAJECTORY_PREFIXES:
        if col.startswith(prefix):
            return "novel"
    return "baseline"


def load_data():
    features = pd.read_parquet(f"{DATA_DIR}features_full.parquet")
    train_ids = json.load(open(f"{DATA_DIR}train_ids.json"))
    test_ids = json.load(open(f"{DATA_DIR}test_ids.json"))

    X = features.drop(columns=["aki_label", "aki_stage"])
    y = features["aki_label"]

    X_train = X.loc[X.index.isin(train_ids)]
    y_train = y.loc[y.index.isin(train_ids)]
    X_test = X.loc[X.index.isin(test_ids)]
    y_test = y.loc[y.index.isin(test_ids)]

    return X_train, y_train, X_test, y_test


def get_column_groups(all_columns):
    baseline_cols = [c for c in all_columns if classify_column(c) == "baseline"]
    novel_cols = [c for c in all_columns if classify_column(c) == "novel"]
    print(f"Baseline (vitals+labs) columns: {len(baseline_cols)}")
    print(f"Novel (nephrotoxin+trajectory) columns: {len(novel_cols)}")
    return baseline_cols, novel_cols


def train_default(X, y, seed=RANDOM_SEED):
    model = lgb.LGBMClassifier(
        n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE, num_leaves=NUM_LEAVES,
        class_weight="balanced", random_state=seed, n_jobs=-1, verbose=-1,
    )
    model.fit(X, y)
    return model


def evaluate(model, X_test, y_test):
    probs = model.predict_proba(X_test)[:, 1]
    return average_precision_score(y_test, probs), roc_auc_score(y_test, probs)


def main():
    X_train, y_train, X_test, y_test = load_data()
    baseline_cols, novel_cols = get_column_groups(X_train.columns.tolist())

    results = []

    # --- Row 1: Baseline (vitals + labs only), untuned ---
    print("\n--- Row 1: Baseline (vitals + labs only) ---")
    X_train_r1 = X_train[baseline_cols]
    X_test_r1 = X_test[baseline_cols]
    model_r1 = train_default(X_train_r1, y_train)
    auprc_r1, auroc_r1 = evaluate(model_r1, X_test_r1, y_test)
    print(f"AUPRC: {auprc_r1:.4f} | AUROC: {auroc_r1:.4f} | Features: {X_train_r1.shape[1]}")
    results.append({"row": "1. Baseline (vitals+labs only)", "n_features": X_train_r1.shape[1],
                     "auprc": auprc_r1, "auroc": auroc_r1})

    # --- Row 2: + Novel features (nephrotoxin + trajectory), uncompressed ---
    print("\n--- Row 2: + Novel features (nephrotoxin + trajectory) ---")
    all_cols_r2 = baseline_cols + novel_cols
    X_train_r2 = X_train[all_cols_r2]
    X_test_r2 = X_test[all_cols_r2]
    model_r2 = train_default(X_train_r2, y_train)
    auprc_r2, auroc_r2 = evaluate(model_r2, X_test_r2, y_test)
    print(f"AUPRC: {auprc_r2:.4f} | AUROC: {auroc_r2:.4f} | Features: {X_train_r2.shape[1]}")
    results.append({"row": "2. + Novel features (Contribution 1)", "n_features": X_train_r2.shape[1],
                     "auprc": auprc_r2, "auroc": auroc_r2,
                     "delta_auprc_vs_prev": auprc_r2 - auprc_r1})

    # --- Row 3: + SHAP compression (frozen 317-feature mask), untuned ---
    print("\n--- Row 3: + SHAP compression ---")
    mask = np.load(f"{DATA_DIR}shap_mask/frozen_mask.npy")
    mask_cols = [X_train.columns[i] for i in np.where(mask == 1)[0]]
    X_train_r3 = X_train[mask_cols]
    X_test_r3 = X_test[mask_cols]
    model_r3 = train_default(X_train_r3, y_train)
    auprc_r3, auroc_r3 = evaluate(model_r3, X_test_r3, y_test)
    print(f"AUPRC: {auprc_r3:.4f} | AUROC: {auroc_r3:.4f} | Features: {X_train_r3.shape[1]}")
    results.append({"row": "3. + SHAP compression (Contribution 2)", "n_features": X_train_r3.shape[1],
                     "auprc": auprc_r3, "auroc": auroc_r3,
                     "delta_auprc_vs_prev": auprc_r3 - auprc_r2})

    # --- Row 4: + Hyperparameter tuning (Final Model A, already trained) ---
    print("\n--- Row 4: + Hyperparameter tuning (Final Model A) ---")
    final_results = json.load(open(f"{DATA_DIR}models/final_model_A_results.json"))
    auprc_r4 = final_results["auprc"]
    auroc_r4 = final_results["auroc"]
    print(f"AUPRC: {auprc_r4:.4f} | AUROC: {auroc_r4:.4f} | Features: {final_results['n_features']}")
    results.append({"row": "4. + Hyperparameter tuning (Final Model A)", "n_features": final_results["n_features"],
                     "auprc": auprc_r4, "auroc": auroc_r4,
                     "delta_auprc_vs_prev": auprc_r4 - auprc_r3})

    # --- Separate note: Active learning data efficiency (Contribution 3) ---
    al_curve = pd.read_csv(f"{DATA_DIR}active_learning/learning_curve.csv")
    al_at_2000 = al_curve[al_curve["n_labeled"] == 2000].iloc[0]
    print(f"\n--- Reference: Active Learning efficiency (Contribution 3, separate dimension) ---")
    print(f"AL model at 2,000 labels (15.7% of data): AUPRC={al_at_2000['auprc']:.4f} "
          f"({100*al_at_2000['auprc']/auprc_r3:.1f}% of Row 3's full-data AUPRC using same mask)")

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{DATA_DIR}models/ablation_table.csv", index=False)

    print("\n=== FULL ABLATION TABLE ===\n")
    print(results_df.to_string(index=False))
    print(f"\nSaved to {DATA_DIR}models/ablation_table.csv")


if __name__ == "__main__":
    main()