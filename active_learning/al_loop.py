import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import average_precision_score, roc_auc_score
from config import (
    DATA_DIR, RANDOM_SEED, N_ESTIMATORS, LEARNING_RATE, NUM_LEAVES,
    SEED_SIZE, QUERY_SIZE, MAX_ITER, AL_RECOMPUTE_EVERY,
    JACCARD_THRESHOLD, JACCARD_WINDOW, MASK_PERCENTILE,
)
from active_learning.badge_query import stratified_badge_query, jaccard_similarity


def load_full_splits():
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

    return X_train, y_train, X_calib, y_calib, X_test, y_test


def stratified_seed_split(X_train, y_train, seed_size=SEED_SIZE, seed=RANDOM_SEED):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=len(X_train) - seed_size, random_state=seed)
    seed_idx, pool_idx = next(sss.split(X_train, y_train))
    labeled_ids = list(X_train.index[seed_idx])
    unlabeled_ids = list(X_train.index[pool_idx])
    return labeled_ids, unlabeled_ids


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


def apply_mask(X, mask):
    selected_cols = np.where(mask == 1)[0]
    return X.iloc[:, selected_cols]


def compute_mask_from_full_features(X_lab_full, y_lab, X_calib_full, percentile=MASK_PERCENTILE):
    """
    Trains a model on the FULL (uncompressed) feature space and computes SHAP
    on the full space too. This lets the mask genuinely gain or drop features
    between recomputations, rather than only ever shrinking within whatever
    was already active (which is what happened when SHAP was computed on an
    already-compressed subset — the new mask could never be anything but a
    strict subset of the old one).
    """
    full_model = train_lgbm(X_lab_full, y_lab)

    import shap
    explainer = shap.TreeExplainer(full_model)
    shap_values = explainer.shap_values(X_calib_full)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    threshold = np.percentile(mean_abs_shap, percentile)
    mask = (mean_abs_shap >= threshold).astype(int)
    return mask, full_model


def run_al_loop():
    X_train, y_train, X_calib, y_calib, X_test, y_test = load_full_splits()
    print(f"Train pool: {X_train.shape} | Calib: {X_calib.shape} | Test: {X_test.shape}")
    print(f"Training set AKI prevalence: {y_train.mean():.3f}")

    labeled_ids, unlabeled_ids = stratified_seed_split(X_train, y_train)
    print(f"Seed set: {len(labeled_ids)} labeled | {len(unlabeled_ids)} unlabeled")
    print(f"Seed set AKI prevalence: {y_train.loc[labeled_ids].mean():.3f}")

    # Starting mask computed from the FULL feature space using only the seed set
    print("Computing seed mask from full feature space...")
    current_mask, _ = compute_mask_from_full_features(
        X_train.loc[labeled_ids], y_train.loc[labeled_ids], X_calib
    )
    print(f"Seed-set mask: {current_mask.sum()} / {len(current_mask)} features active")

    learning_curve = []
    mask_history = [current_mask.copy()]
    mask_frozen = False
    budget = int(0.15 * len(X_train))

    for iteration in range(MAX_ITER):
        n_labeled = len(labeled_ids)
        print(f"\n--- Iteration {iteration + 1} | Labeled: {n_labeled} | Budget: {budget} ---")

        active_mask = current_mask.copy()

        X_lab_c = apply_mask(X_train.loc[labeled_ids], active_mask)
        X_test_c = apply_mask(X_test, active_mask)

        model = train_lgbm(X_lab_c, y_train.loc[labeled_ids])

        test_probs = model.predict_proba(X_test_c)[:, 1]
        auprc = average_precision_score(y_test, test_probs)
        auroc = roc_auc_score(y_test, test_probs)
        current_prevalence = y_train.loc[labeled_ids].mean()

        print(f"AUPRC: {auprc:.4f} | AUROC: {auroc:.4f} | Labeled-pool prevalence: {current_prevalence:.3f}")

        learning_curve.append({
            "iteration": iteration + 1,
            "n_labeled": n_labeled,
            "auprc": auprc,
            "auroc": auroc,
            "labeled_prevalence": current_prevalence,
            "mask_frozen": mask_frozen,
            "n_active_features": int(active_mask.sum()),
        })

        if n_labeled >= budget:
            print(f"Budget reached ({n_labeled} >= {budget}). Stopping.")
            break

        if len(unlabeled_ids) == 0:
            print("Unlabeled pool exhausted. Stopping.")
            break

        X_pool_c = apply_mask(X_train.loc[unlabeled_ids], active_mask)
        queried_ids = stratified_badge_query(
            model, X_pool_c, current_prevalence, query_size=min(QUERY_SIZE, len(unlabeled_ids))
        )
        labeled_ids += queried_ids
        unlabeled_ids = [i for i in unlabeled_ids if i not in queried_ids]

        if not mask_frozen and (iteration + 1) % AL_RECOMPUTE_EVERY == 0:
            print("Recomputing mask from full feature space...")
            candidate_mask, candidate_full_model = compute_mask_from_full_features(
                X_train.loc[labeled_ids], y_train.loc[labeled_ids], X_calib
            )

            jaccard = jaccard_similarity(active_mask, candidate_mask)

            # candidate_full_model was trained on FULL features — predict on full X_calib
            candidate_auprc = average_precision_score(
                y_calib, candidate_full_model.predict_proba(X_calib)[:, 1]
            )

            X_calib_current = apply_mask(X_calib, active_mask)
            current_auprc = average_precision_score(
                y_calib, model.predict_proba(X_calib_current)[:, 1]
            )

            print(f"Mask recompute candidate: Jaccard={jaccard:.4f} | "
                  f"Current mask calib AUPRC={current_auprc:.4f} | "
                  f"New mask calib AUPRC={candidate_auprc:.4f} | "
                  f"New mask size={candidate_mask.sum()}")

            if candidate_auprc >= current_auprc:
                print("New mask ACCEPTED (non-regressing).")
                mask_history.append(candidate_mask.copy())
                current_mask = candidate_mask

                if len(mask_history) >= JACCARD_WINDOW:
                    recent = mask_history[-JACCARD_WINDOW:]
                    all_stable = all(
                        jaccard_similarity(recent[i], recent[i + 1]) >= JACCARD_THRESHOLD
                        for i in range(len(recent) - 1)
                    )
                    if all_stable:
                        mask_frozen = True
                        print(f"Mask frozen at iteration {iteration + 1}")
            else:
                print("New mask REJECTED (would regress performance). Keeping current mask.")

    return learning_curve, model, current_mask, labeled_ids


def main():
    np.random.seed(RANDOM_SEED)
    learning_curve, final_model, frozen_mask, labeled_ids = run_al_loop()

    os.makedirs(f"{DATA_DIR}active_learning", exist_ok=True)
    os.makedirs(f"{DATA_DIR}models", exist_ok=True)

    pd.DataFrame(learning_curve).to_csv(f"{DATA_DIR}active_learning/learning_curve.csv", index=False)
    np.save(f"{DATA_DIR}shap_mask/frozen_mask.npy", frozen_mask)
    json.dump([int(x) for x in labeled_ids], open(f"{DATA_DIR}active_learning/final_labeled_ids.json", "w"))
    final_model.booster_.save_model(f"{DATA_DIR}models/final_model.txt")

    print(f"\nLearning curve saved. Final labeled set size: {len(labeled_ids)}")
    print(f"Frozen mask: {frozen_mask.sum()} active features")


if __name__ == "__main__":
    main()