import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score
from config import DATA_DIR, RANDOM_SEED


def load_full_splits():
    features = pd.read_parquet(f"{DATA_DIR}features_full.parquet")
    train_ids = json.load(open(f"{DATA_DIR}train_ids.json"))
    calib_ids = json.load(open(f"{DATA_DIR}calib_ids.json"))

    X_train = features.loc[features.index.isin(train_ids)].drop(columns=["aki_label", "aki_stage"])
    y_train = features.loc[features.index.isin(train_ids)]["aki_label"]

    X_calib = features.loc[features.index.isin(calib_ids)].drop(columns=["aki_label", "aki_stage"])
    y_calib = features.loc[features.index.isin(calib_ids)]["aki_label"]

    return X_train, y_train, X_calib, y_calib


def apply_mask(X, mask):
    selected_cols = np.where(mask == 1)[0]
    return X.iloc[:, selected_cols]


def sample_hyperparams(rng):
    """
    Search space chosen to bracket the original defaults (n_estimators=500,
    learning_rate=0.05, num_leaves=63) so the search can move meaningfully
    in either direction rather than just confirming the defaults were fine.
    """
    return {
        "n_estimators": int(rng.choice([200, 300, 500, 800, 1200])),
        "learning_rate": float(rng.choice([0.01, 0.02, 0.05, 0.08, 0.1])),
        "num_leaves": int(rng.choice([15, 31, 63, 95, 127])),
        "min_child_samples": int(rng.choice([5, 10, 20, 40, 60])),
        "max_depth": int(rng.choice([-1, 4, 6, 8, 10])),
        "reg_alpha": float(rng.choice([0.0, 0.1, 0.5, 1.0])),
        "reg_lambda": float(rng.choice([0.0, 0.1, 0.5, 1.0])),
    }


def train_and_eval(params, X_train, y_train, X_calib, y_calib, seed=RANDOM_SEED):
    model = lgb.LGBMClassifier(
        **params,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_calib)[:, 1]
    auprc = average_precision_score(y_calib, probs)
    auroc = roc_auc_score(y_calib, probs)
    return auprc, auroc, model


def run_search(n_trials=30, seed=RANDOM_SEED):
    X_train, y_train, X_calib, y_calib = load_full_splits()

    mask = np.load(f"{DATA_DIR}shap_mask/frozen_mask.npy")
    print(f"Using frozen (AL-converged) mask: {mask.sum()} / {len(mask)} features active")

    X_train_c = apply_mask(X_train, mask)
    X_calib_c = apply_mask(X_calib, mask)

    print(f"Train: {X_train_c.shape} | Calib: {X_calib_c.shape}")
    print(f"Running {n_trials} random search trials, selecting by calibration AUPRC...\n")

    rng = np.random.RandomState(seed)
    results = []
    best_auprc = -1
    best_params = None

    for trial in range(n_trials):
        params = sample_hyperparams(rng)
        auprc, auroc, _ = train_and_eval(params, X_train_c, y_train, X_calib_c, y_calib, seed=seed)

        print(f"Trial {trial + 1}/{n_trials} | AUPRC={auprc:.4f} | AUROC={auroc:.4f} | {params}")

        results.append({**params, "auprc": auprc, "auroc": auroc, "trial": trial + 1})

        if auprc > best_auprc:
            best_auprc = auprc
            best_params = params
            print(f"  --> New best (AUPRC={auprc:.4f})")

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{DATA_DIR}models/hyperparam_search_results.csv", index=False)

    print(f"\n=== Best hyperparameters (calib AUPRC={best_auprc:.4f}) ===")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    json.dump(best_params, open(f"{DATA_DIR}models/best_hyperparams.json", "w"))
    print(f"\nSaved best hyperparams to {DATA_DIR}models/best_hyperparams.json")
    print(f"Full search results saved to {DATA_DIR}models/hyperparam_search_results.csv")

    return best_params, best_auprc


if __name__ == "__main__":
    run_search(n_trials=30)