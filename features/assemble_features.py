import os
import pandas as pd
from config import DATA_DIR


def load_all_blocks():
    print("Loading feature blocks...")
    vitals = pd.read_parquet(f"{DATA_DIR}vitals_values.parquet")
    vitals_mask = pd.read_parquet(f"{DATA_DIR}vitals_mask.parquet")
    labs = pd.read_parquet(f"{DATA_DIR}labs_values.parquet")
    labs_mask = pd.read_parquet(f"{DATA_DIR}labs_mask.parquet")
    nephrotoxin = pd.read_parquet(f"{DATA_DIR}nephrotoxin_features.parquet")
    trajectory = pd.read_parquet(f"{DATA_DIR}trajectory_features.parquet")

    print(f"  vitals:        {vitals.shape}")
    print(f"  vitals_mask:   {vitals_mask.shape}")
    print(f"  labs:          {labs.shape}")
    print(f"  labs_mask:     {labs_mask.shape}")
    print(f"  nephrotoxin:   {nephrotoxin.shape}")
    print(f"  trajectory:    {trajectory.shape}")

    return vitals, vitals_mask, labs, labs_mask, nephrotoxin, trajectory


def load_labels():
    cohort = pd.read_parquet(f"{DATA_DIR}cohort.parquet")
    labels = cohort.set_index("stay_id")[["aki_label", "aki_stage"]]
    print(f"  labels:        {labels.shape}")
    return labels


def check_index_alignment(blocks_dict):
    """
    Before joining, verify all blocks share the same stay_id index.
    Mismatches here would silently produce NaN rows after the join —
    better to catch it explicitly now.
    """
    index_sets = {name: set(df.index) for name, df in blocks_dict.items()}
    reference_name = list(index_sets.keys())[0]
    reference_set = index_sets[reference_name]

    print(f"\nChecking index alignment against '{reference_name}' ({len(reference_set)} stay_ids)...")
    all_match = True
    for name, idx_set in index_sets.items():
        if idx_set != reference_set:
            missing_from_this = reference_set - idx_set
            extra_in_this = idx_set - reference_set
            print(f"  MISMATCH in '{name}': "
                  f"{len(missing_from_this)} missing, {len(extra_in_this)} extra")
            all_match = False
        else:
            print(f"  '{name}': OK, matches exactly")

    if all_match:
        print("All blocks aligned correctly.")
    return all_match


def assemble_features():
    vitals, vitals_mask, labs, labs_mask, nephrotoxin, trajectory = load_all_blocks()
    labels = load_labels()

    blocks = {
        "vitals": vitals,
        "vitals_mask": vitals_mask,
        "labs": labs,
        "labs_mask": labs_mask,
        "nephrotoxin": nephrotoxin,
        "trajectory": trajectory,
        "labels": labels,
    }

    aligned = check_index_alignment(blocks)
    if not aligned:
        print("\nWARNING: index mismatch detected — proceeding with an inner join, "
              "which will only keep stay_ids present in ALL blocks. "
              "This may silently drop patients. Review the mismatch counts above.")

    # Inner join on stay_id index across all blocks
    full = vitals.join(vitals_mask, how="inner")
    full = full.join(labs, how="inner")
    full = full.join(labs_mask, how="inner")
    full = full.join(nephrotoxin, how="inner")
    full = full.join(trajectory, how="inner")
    full = full.join(labels, how="inner")

    print(f"\nFinal assembled matrix shape: {full.shape}")
    print(f"AKI label distribution:\n{full['aki_label'].value_counts()}")

    return full


def main():
    full = assemble_features()

    os.makedirs(DATA_DIR, exist_ok=True)
    full.to_parquet(f"{DATA_DIR}features_full.parquet")
    print(f"\nSaved to {DATA_DIR}features_full.parquet")


if __name__ == "__main__":
    main()