import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score
from config import DATA_DIR, RANDOM_SEED, N_ESTIMATORS, LEARNING_RATE, NUM_LEAVES


def load_full_splits():
    features = pd.read_parquet(f"{DATA_DIR}features_full.parquet")
    train_ids = json.load(open(f"{DATA_DIR}train_ids.json"))
    test_ids  = json.load(open(f"{DATA_DIR}test_ids.json"))

    X_train = features.loc[features.index.isin(train_ids)].drop(columns=["aki_label", "aki_stage"])
    y_train = features.loc[features.index.isin(train_ids)]["aki_label"]

    X_test = features.loc[features.index.isin(test_ids)].drop(columns=["aki_label", "aki_stage"])
    y_test = features.loc[features.index.isin(test_ids)]["aki_label"]

    return X_train, y_train, X_test, y_test


def apply_mask(X, mask):
    selected_cols = np.where(mask == 1)[0]
    return X.iloc[:, selected_cols]


def train_lgbm(X, y, seed=RANDOM_SEED):
    model = lgb.LGBMClassifier(
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        num_leaves=NUM_LEAVES,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X, y)
    return model


def main():
    X_train, y_train, X_test, y_test = load_full_splits()
    print(f"Full train pool: {X_train.shape} | Test: {X_test.shape}")

    # Use the SAME mask the AL loop used (317 features) — the original Stage-3
    # mask, since none of the AL loop's recompute attempts were accepted.
    mask = np.load(f"{DATA_DIR}shap_mask/frozen_mask.npy")
    print(f"Using mask: {mask.sum()} / {len(mask)} features active")

    X_train_c = apply_mask(X_train, mask)
    X_test_c = apply_mask(X_test, mask)

    print(f"\nTraining full-data model on ALL {len(X_train_c)} labeled patients, "
          f"same {mask.sum()}-feature compressed space as the AL run...")
    full_model = train_lgbm(X_train_c, y_train)

    full_probs = full_model.predict_proba(X_test_c)[:, 1]
    full_auprc = average_precision_score(y_test, full_probs)
    full_auroc = roc_auc_score(y_test, full_probs)

    print(f"\n=== Full-data baseline (same 317 features, all {len(X_train_c)} labels) ===")
    print(f"AUPRC: {full_auprc:.4f} | AUROC: {full_auroc:.4f}")

    # Load the AL run's learning curve to compare against
    al_curve = pd.read_csv(f"{DATA_DIR}active_learning/learning_curve.csv")
    al_final = al_curve.iloc[-1]

    print(f"\n=== AL-selected model (2000 labels, same 317 features) ===")
    print(f"AUPRC: {al_final['auprc']:.4f} | AUROC: {al_final['auroc']:.4f}")

    pct_of_full = 100 * al_final['auprc'] / full_auprc
    print(f"\n=== Comparison ===")
    print(f"AL model reaches {pct_of_full:.1f}% of full-data AUPRC "
          f"using {al_final['n_labeled']:.0f} / {len(X_train_c)} labels "
          f"({100 * al_final['n_labeled'] / len(X_train_c):.1f}% of training data)")

    # Find the exact iteration where 95% of full-data performance was first reached
    target_auprc = 0.95 * full_auprc
    crossing = al_curve[al_curve['auprc'] >= target_auprc]
    if len(crossing) > 0:
        first_cross = crossing.iloc[0]
        pct_labels = 100 * first_cross['n_labeled'] / len(X_train_c)
        print(f"\n95% of full-data AUPRC ({target_auprc:.4f}) first reached at "
              f"iteration {first_cross['iteration']:.0f} "
              f"({first_cross['n_labeled']:.0f} labels, {pct_labels:.1f}% of training data)")
    else:
        print(f"\n95% of full-data AUPRC ({target_auprc:.4f}) was NOT reached within the AL budget.")


if __name__ == "__main__":
    main()