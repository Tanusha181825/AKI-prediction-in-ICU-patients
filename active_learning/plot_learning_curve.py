import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import DATA_DIR


def plot_learning_curve():
    curve = pd.read_csv(f"{DATA_DIR}active_learning/learning_curve.csv")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(curve["n_labeled"], curve["auprc"], "o-", color="#1D9E75",
            linewidth=2, markersize=5, label="SHAP-BADGE Active Learning")

    # Mark the 15% budget point (2000 labels) — the headline result
    headline_row = curve[curve["n_labeled"] == 2000]
    if len(headline_row) > 0:
        ax.axvline(2000, linestyle="--", color="#888780", alpha=0.6)
        ax.annotate(
            f"15% budget\n(AUPRC={headline_row['auprc'].values[0]:.4f})",
            xy=(2000, headline_row["auprc"].values[0]),
            xytext=(2000 + 150, headline_row["auprc"].values[0] - 0.025),
            fontsize=9, color="#555555",
            arrowprops=dict(arrowstyle="->", color="#888780"),
        )

    # Shade the plateau region (post-iteration-16) to visually show saturation
    plateau_start = 2000
    ax.axvspan(plateau_start, curve["n_labeled"].max(), alpha=0.08, color="#1D9E75",
               label="Plateau region")

    # Mark mask recompute / acceptance points
    recompute_iters = [5, 10, 15, 20]
    n_labeled_at_recompute = curve[curve["iteration"].isin(recompute_iters)]["n_labeled"]
    for n in n_labeled_at_recompute:
        ax.axvline(n, linestyle=":", color="#EF9F27", alpha=0.4, linewidth=1)

    ax.set_xlabel("Number of Labeled Patients")
    ax.set_ylabel("AUPRC (Test Set)")
    ax.set_title("SHAP-BADGE Active Learning: Performance vs. Labeling Budget")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)

    plt.tight_layout()
    out_path = f"{DATA_DIR}active_learning/al_learning_curve.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved learning curve plot to {out_path}")

    # Print a quick summary of the plateau
    pre_plateau = curve[curve["n_labeled"] <= 2000]
    post_plateau = curve[curve["n_labeled"] > 2000]

    print(f"\nPre-2000-label AUPRC range: {pre_plateau['auprc'].min():.4f} – {pre_plateau['auprc'].max():.4f}")
    print(f"Post-2000-label AUPRC range: {post_plateau['auprc'].min():.4f} – {post_plateau['auprc'].max():.4f}")
    print(f"Post-2000-label AUPRC std dev: {post_plateau['auprc'].std():.4f} "
          f"(low std = flat/plateaued, high std = still trending)")


if __name__ == "__main__":
    plot_learning_curve()