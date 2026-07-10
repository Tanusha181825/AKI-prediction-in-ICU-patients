import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from config import DATA_DIR, N_HOURS


def load_shap_artifacts():
    importance_matrix = np.load(f"{DATA_DIR}shap_mask/importance_matrix.npy")
    base_features = json.load(open(f"{DATA_DIR}shap_mask/base_features.json"))
    return importance_matrix, base_features


def plot_temporal_shap_heatmap(importance_matrix, base_features, top_n=20, n_hours=N_HOURS):
    # Select top N features by their single highest hourly SHAP value
    row_max = importance_matrix.max(axis=1)
    top_idx = np.argsort(row_max)[-top_n:][::-1]

    matrix_top = importance_matrix[top_idx]
    labels_top = [base_features[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(
        matrix_top,
        ax=ax,
        cmap="YlOrRd",
        xticklabels=[f"H{h}" for h in range(n_hours)],
        yticklabels=labels_top,
        cbar_kws={"label": "Mean |SHAP|"},
    )
    ax.set_xlabel("Hour of ICU stay (0–23)")
    ax.set_ylabel("Feature")
    ax.set_title("Temporal SHAP Importance — Calibration Set (Top 20 Features)")
    plt.tight_layout()

    out_path = f"{DATA_DIR}shap_mask/temporal_shap_heatmap.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved heatmap to {out_path}")

    return labels_top


def print_top_features_summary(importance_matrix, base_features, labels_top, n_hours=N_HOURS):
    """
    Quick text summary alongside the plot — which hour each top feature peaks at.
    Useful for identifying the clinical narrative (e.g., 'nephrotoxin features
    dominate hours 1-8, creatinine slope takes over hours 14-24').
    """
    print("\n=== Top 20 features — peak importance hour ===")
    for label in labels_top:
        idx = base_features.index(label)
        row = importance_matrix[idx]
        peak_hour = int(np.argmax(row))
        peak_val = row[peak_hour]
        print(f"  {label:35s}  peak at H{peak_hour:02d}  (SHAP={peak_val:.5f})")


def main():
    importance_matrix, base_features = load_shap_artifacts()
    print(f"Loaded importance matrix: {importance_matrix.shape} "
          f"({len(base_features)} base features x {N_HOURS} hours)")

    labels_top = plot_temporal_shap_heatmap(importance_matrix, base_features)
    print_top_features_summary(importance_matrix, base_features, labels_top)


if __name__ == "__main__":
    main()