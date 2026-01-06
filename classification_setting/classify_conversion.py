import pandas as pd
import os

def add_oxidation_zone_to_csv(filepath):
    """
    Reads a CSV, categorizes 'target_var' into 'oxidation_zone',
    and overwrites the file.
    """
    # Check if file exists to avoid errors
    if not os.path.exists(filepath):
        print(f"Skipping {filepath} — file not found.")
        return

    df = pd.read_csv(filepath)

    # Skip if target column doesn't exist
    if "target_var" not in df.columns:
        print(f"Skipping {filepath} — target column missing.")
        return

    # Add oxidation zone
    # Arbitrary, decided on the basis of domain knowledge
    bins = [0.0, 0.08, 0.13, float("inf")]
    labels = ["low", "intermediate", "high"]

    df["oxidation_zone"] = pd.cut(
        df["target_var"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    # Save back to same file
    df.to_csv(filepath, index=False)
    print(f"Updated: {filepath}")


if __name__ == "__main__":
    # === Filepaths ===
    csv_paths = [
        "mito_data/nets/group1_net_sheet.csv",
        "mito_data/non-nets/group1_standalone_sheet.csv",
        "mito_data/nets/group2_net_sheet.csv",
        "mito_data/non-nets/group2_standalone_sheet.csv",
        "mito_data/nets/group3_sheet.csv",
        "mito_data/non-nets/group3_standalone_sheet.csv",
        "mito_data/nets/group4_net_sheet.csv",
        "mito_data/non-nets/group4_standalone_sheet.csv",
        "mito_data/nets/group5_net_sheet.csv",
        "mito_data/non-nets/group5_standalone_sheet.csv",
        "mito_data/nets/group6_net_sheet.csv",
        "mito_data/non-nets/group6_standalone_sheet.csv",
        "mito_data/nets/group7_net_sheet.csv",
        "mito_data/non-nets/group7_standalones_sheet.csv",
        "mito_data/nets/group8_net_sheet.csv",
        "mito_data/non-nets/group8_standalones_sheet.csv",
        "mito_data/nets/group9_net_sheet.csv",
        "mito_data/non-nets/group9_standalones_sheet.csv",
        "mito_data/nets/group10_net_sheet.csv",
        "mito_data/non-nets/group10_standalones_sheet.csv"
    ]

    # === Run on all files ===
    for path in csv_paths:
        add_oxidation_zone_to_csv(path)