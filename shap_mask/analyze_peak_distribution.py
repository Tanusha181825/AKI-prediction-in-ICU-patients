import json
import numpy as np
from config import DATA_DIR, N_HOURS


def analyze_peak_hour_distribution():
    importance_matrix = np.load(f"{DATA_DIR}shap_mask/importance_matrix.npy")
    base_features = json.load(open(f"{DATA_DIR}shap_mask/base_features.json"))

    row_max = importance_matrix.max(axis=1)
    top_idx = np.argsort(row_max)[-20:][::-1]

    peak_hours = []
    for idx in top_idx:
        row = importance_matrix[idx]
        peak_hour = int(np.argmax(row))
        peak_hours.append(peak_hour)

    peak_hours = np.array(peak_hours)

    # Split into thirds of the 24-hour window
    early = np.sum(peak_hours < 8)       # H0-H7
    middle = np.sum((peak_hours >= 8) & (peak_hours < 16))  # H8-H15
    late = np.sum(peak_hours >= 16)      # H16-H23

    n = len(peak_hours)
    print(f"=== Peak hour distribution across top {n} features ===")
    print(f"Early third  (H0–H7):   {early:2d} features ({100*early/n:.1f}%)")
    print(f"Middle third (H8–H15):  {middle:2d} features ({100*middle/n:.1f}%)")
    print(f"Late third   (H16–H23): {late:2d} features ({100*late/n:.1f}%)")

    print(f"\nMean peak hour: {peak_hours.mean():.2f}")
    print(f"Median peak hour: {np.median(peak_hours):.1f}")
    print(f"Std dev of peak hour: {peak_hours.std():.2f}")

    # Simple bimodality check: gap in the middle third
    print(f"\nMiddle-third feature count ({middle}) vs expected under uniform "
          f"distribution ({n/3:.1f}) — {'notably sparse' if middle < n/3 * 0.6 else 'roughly as expected'}")


if __name__ == "__main__":
    analyze_peak_hour_distribution()