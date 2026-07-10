import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    confusion_matrix, precision_recall_curve, roc_curve,
    precision_score, recall_score, f1_score
)
from config import DATA_DIR


def load_test_labels():
    features = pd.read_parquet(f"{DATA_DIR}features_full.parquet")
    test_ids = json.load(open(f"{DATA_DIR}test_ids.json"))
    y_test = features.loc[features.index.isin(test_ids)]["aki_label"]
    return y_test


def plot_calibration_curve(y_test, test_probs):
    """
    Reliability diagram: bins predictions into deciles, plots mean predicted
    probability vs. observed AKI rate in each bin. A perfectly calibrated
    model sits exactly on the diagonal.
    """
    prob_true, prob_pred = calibration_curve(y_test, test_probs, n_bins=10, strategy="quantile")

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#888780", label="Perfect calibration")
    ax.plot(prob_pred, prob_true, "o-", color="#1D9E75", linewidth=2, markersize=7,
             label="Model A (test set)")

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Observed AKI Rate")
    ax.set_title("Calibration Curve — Final Model A (Test Set)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    plt.tight_layout()

    out_path = f"{DATA_DIR}models/calibration_curve.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved calibration curve to {out_path}")


def plot_pr_and_roc_curves(y_test, test_probs):
    """Combined PR + ROC curves side by side, since AUPRC/AUROC are your two primary metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    precision, recall, _ = precision_recall_curve(y_test, test_probs)
    baseline_prevalence = y_test.mean()
    ax1.plot(recall, precision, color="#1D9E75", linewidth=2)
    ax1.axhline(baseline_prevalence, linestyle="--", color="#888780",
                label=f"Random baseline (prevalence={baseline_prevalence:.3f})")
    ax1.set_xlabel("Recall")
    ax1.set_ylabel("Precision")
    ax1.set_title("Precision-Recall Curve")
    ax1.legend()
    ax1.grid(alpha=0.25)

    fpr, tpr, _ = roc_curve(y_test, test_probs)
    ax2.plot(fpr, tpr, color="#1D9E75", linewidth=2)
    ax2.plot([0, 1], [0, 1], linestyle="--", color="#888780", label="Random classifier")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve")
    ax2.legend()
    ax2.grid(alpha=0.25)

    plt.tight_layout()
    out_path = f"{DATA_DIR}models/pr_roc_curves.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved PR/ROC curves to {out_path}")


def find_optimal_threshold(y_test, test_probs):
    """
    Rather than defaulting to 0.5 (inappropriate given 34% prevalence), find
    the threshold that maximizes F1 — a reasonable clinical default balancing
    precision and recall. Reported alongside, not as the only valid choice.
    """
    precision, recall, thresholds = precision_recall_curve(y_test, test_probs)
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]


def plot_confusion_matrix(y_test, test_probs, threshold):
    preds = (test_probs >= threshold).astype(int)
    cm = confusion_matrix(y_test, preds)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No AKI", "AKI"])
    ax.set_yticklabels(["No AKI", "AKI"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix (threshold={threshold:.3f})")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    out_path = f"{DATA_DIR}models/confusion_matrix.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix to {out_path}")

    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    return {
        "threshold": float(threshold), "precision": float(precision),
        "recall": float(recall), "f1": float(f1),
        "sensitivity": float(sensitivity), "specificity": float(specificity),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def main():
    y_test = load_test_labels()
    test_probs = np.load(f"{DATA_DIR}models/final_model_A_test_probs.npy")

    print(f"Test set: {len(y_test)} patients, {y_test.sum()} AKI-positive ({y_test.mean():.3f} prevalence)")

    plot_calibration_curve(y_test, test_probs)
    plot_pr_and_roc_curves(y_test, test_probs)

    best_threshold, best_f1 = find_optimal_threshold(y_test, test_probs)
    print(f"\nF1-optimal threshold: {best_threshold:.4f} (F1={best_f1:.4f})")

    cm_results = plot_confusion_matrix(y_test, test_probs, best_threshold)

    print(f"\n=== Confusion Matrix Summary (threshold={best_threshold:.3f}) ===")
    for k, v in cm_results.items():
        print(f"  {k}: {v}")

    # Also report at 0.5 for reference, since some readers expect it
    cm_results_default = plot_confusion_matrix(y_test, test_probs, 0.5)
    print(f"\n=== Confusion Matrix Summary (default threshold=0.5, for reference) ===")
    for k, v in cm_results_default.items():
        print(f"  {k}: {v}")

    all_results = {
        "f1_optimal": cm_results,
        "default_0.5": cm_results_default,
    }
    json.dump(all_results, open(f"{DATA_DIR}models/evaluation_summary.json", "w"))
    print(f"\nFull evaluation summary saved to {DATA_DIR}models/evaluation_summary.json")


if __name__ == "__main__":
    main()