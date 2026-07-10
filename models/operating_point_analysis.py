import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from config import DATA_DIR


def load_test_labels():
    features = pd.read_parquet(f"{DATA_DIR}features_full.parquet")
    test_ids = json.load(open(f"{DATA_DIR}test_ids.json"))
    return features.loc[features.index.isin(test_ids)]["aki_label"]


def compute_metrics_at_threshold(y_test, test_probs, threshold, n_patients_ref=100):
    preds = (test_probs >= threshold).astype(int)
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = precision_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)

    # False alarm rate: how many alerts fire per n_patients_ref non-AKI patients
    # This reframes specificity into something a clinician / deployer intuitively
    # understands as "alarm burden" rather than an abstract statistical rate.
    n_negative = tn + fp
    false_alarms_per_n = (fp / n_negative) * n_patients_ref if n_negative > 0 else 0

    # Of all alerts fired, what fraction are real (this is precision, restated
    # as "1 in every X alerts is a true positive" — more intuitive for clinicians)
    n_alerts = tp + fp
    alerts_per_true_case = n_alerts / tp if tp > 0 else float("inf")

    return {
        "threshold": round(float(threshold), 3),
        "sensitivity": round(float(sensitivity), 3),
        "specificity": round(float(specificity), 3),
        "precision": round(float(precision), 3),
        "f1": round(float(f1), 3),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "false_alarms_per_100_no_aki": round(float(false_alarms_per_n), 1),
        "alerts_needed_per_true_case": round(float(alerts_per_true_case), 2),
    }


def build_operating_point_table(y_test, test_probs):
    """
    Named operating points spanning the sensitivity-specificity tradeoff,
    directly engaging with the alarm-fatigue deployment gap (Gap 5) rather
    than reporting a single 'optimal' threshold as if it were the only
    valid choice. A hospital could pick based on their actual alert tolerance.
    """
    thresholds_to_test = np.arange(0.30, 0.71, 0.05)

    rows = []
    for t in thresholds_to_test:
        m = compute_metrics_at_threshold(y_test, test_probs, t)
        rows.append(m)

    df = pd.DataFrame(rows)

    # Label a few key named points for the paper
    named_points = {
        0.35: "High-sensitivity (screening)",
        0.42: "F1-optimal (balanced)",
        0.50: "Default (reference)",
        0.60: "High-specificity (low alarm burden)",
    }
    df["label"] = df["threshold"].apply(
        lambda t: next((v for k, v in named_points.items() if abs(t - k) < 0.026), "")
    )

    return df

def plot_confusion_matrix_for_threshold(y_test, test_probs, threshold, label, out_dir):
    from sklearn.metrics import confusion_matrix
    preds = (test_probs >= threshold).astype(int)
    cm = confusion_matrix(y_test, preds)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["No AKI", "AKI"]); ax.set_yticklabels(["No AKI", "AKI"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    safe_label = label.replace(" ", "_").replace("(", "").replace(")", "").lower()
    ax.set_title(f"{label}\n(threshold={threshold:.2f})")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    out_path = f"{out_dir}confusion_matrix_{safe_label}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")

def plot_operating_curve(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(df["threshold"], df["sensitivity"], "o-", color="#1D9E75",
              linewidth=2, label="Sensitivity (recall)")
    ax1.plot(df["threshold"], df["specificity"], "o-", color="#EF9F27",
              linewidth=2, label="Specificity")
    ax1.set_xlabel("Decision Threshold")
    ax1.set_ylabel("Rate")
    ax1.set_title("Sensitivity vs. Specificity Across Thresholds")
    ax1.legend()
    ax1.grid(alpha=0.25)

    ax2.plot(df["threshold"], df["false_alarms_per_100_no_aki"], "o-",
              color="#C0392B", linewidth=2)
    ax2.set_xlabel("Decision Threshold")
    ax2.set_ylabel("False Alarms per 100 Non-AKI Patients")
    ax2.set_title("Alarm Burden Across Thresholds")
    ax2.grid(alpha=0.25)

    plt.tight_layout()
    out_path = f"{DATA_DIR}models/operating_point_tradeoff.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved operating point tradeoff plot to {out_path}")


def main():
    y_test = load_test_labels()
    test_probs = np.load(f"{DATA_DIR}models/final_model_A_test_probs.npy")

    df = build_operating_point_table(y_test, test_probs)

    print("\n=== Operating Point Table (deployment threshold tradeoff) ===\n")
    display_cols = ["threshold", "label", "sensitivity", "specificity", "precision",
                     "false_alarms_per_100_no_aki", "alerts_needed_per_true_case"]
    print(df[display_cols].to_string(index=False))

    df.to_csv(f"{DATA_DIR}models/operating_point_table.csv", index=False)
    print(f"\nSaved full table to {DATA_DIR}models/operating_point_table.csv")

    plot_operating_curve(df)

    print("\n=== Key operating points for paper narrative ===")
    for _, row in df[df["label"] != ""].iterrows():
        print(f"\n{row['label']} (threshold={row['threshold']}):")
        print(f"  Sensitivity: {row['sensitivity']:.1%} | Specificity: {row['specificity']:.1%}")
        print(f"  Precision: {row['precision']:.1%} | "
              f"False alarms per 100 non-AKI patients: {row['false_alarms_per_100_no_aki']:.1f}")
        print(f"  Alerts needed per true AKI case caught: {row['alerts_needed_per_true_case']:.2f}")

    named = df[df["label"] != ""]
    for _, row in named.iterrows():
        plot_confusion_matrix_for_threshold(
            y_test, test_probs, row["threshold"], row["label"], f"{DATA_DIR}models/"
        )


if __name__ == "__main__":
    main()