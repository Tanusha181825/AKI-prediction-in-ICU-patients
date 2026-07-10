import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from config import DATA_DIR, N_HOURS


def load_shap_artifacts():
    importance_matrix = np.load(f"{DATA_DIR}shap_mask/importance_matrix.npy")
    base_features = json.load(open(f"{DATA_DIR}shap_mask/base_features.json"))
    return importance_matrix, base_features


def plot_absolute_heatmap(importance_matrix, base_features, top_n=20, n_hours=N_HOURS):
    """Original absolute-scale heatmap — shows true magnitude, but dominant
    features (like uo_rolling4h) wash out the color scale for the rest."""
    row_max = importance_matrix.max(axis=1)
    top_idx = np.argsort(row_max)[-top_n:][::-1]
    matrix_top = importance_matrix[top_idx]
    labels_top = [base_features[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(
        matrix_top, ax=ax, cmap="YlOrRd",
        xticklabels=[f"H{h}" for h in range(n_hours)],
        yticklabels=labels_top,
        cbar_kws={"label": "Mean |SHAP|"},
    )
    ax.set_xlabel("Hour of ICU stay (0–23)")
    ax.set_ylabel("Feature")
    ax.set_title("Temporal SHAP Importance — Absolute Scale (Top 20 Features)")
    plt.tight_layout()
    out_path = f"{DATA_DIR}shap_mask/temporal_shap_heatmap_absolute.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved absolute-scale heatmap to {out_path}")
    return labels_top, top_idx


def plot_row_normalized_heatmap(importance_matrix, base_features, top_idx, labels_top, n_hours=N_HOURS):
    """
    Per-row normalization: each feature's 24-hour importance is scaled to its
    OWN 0-1 range. This reveals each feature's temporal SHAPE (when does it
    peak relative to itself) even when its absolute magnitude is small
    compared to a dominant feature like uo_rolling4h.
    """
    matrix_top = importance_matrix[top_idx]

    row_min = matrix_top.min(axis=1, keepdims=True)
    row_max = matrix_top.max(axis=1, keepdims=True)
    row_range = np.where(row_max - row_min == 0, 1, row_max - row_min)
    normalized = (matrix_top - row_min) / row_range

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(
        normalized, ax=ax, cmap="YlOrRd",
        xticklabels=[f"H{h}" for h in range(n_hours)],
        yticklabels=labels_top,
        cbar_kws={"label": "Relative importance (per-feature, 0=lowest hour, 1=peak hour)"},
    )
    ax.set_xlabel("Hour of ICU stay (0–23)")
    ax.set_ylabel("Feature")
    ax.set_title("Temporal SHAP Importance — Per-Feature Normalized (Top 20 Features)")
    plt.tight_layout()
    out_path = f"{DATA_DIR}shap_mask/temporal_shap_heatmap_normalized.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved row-normalized heatmap to {out_path}")


def print_top_features_summary(importance_matrix, base_features, labels_top, n_hours=N_HOURS):
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

    labels_top, top_idx = plot_absolute_heatmap(importance_matrix, base_features)
    plot_row_normalized_heatmap(importance_matrix, base_features, top_idx, labels_top)
    print_top_features_summary(importance_matrix, base_features, labels_top)


if __name__ == "__main__":
    main()