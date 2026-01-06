import pandas as pd
from classify_conversion import add_oxidation_zone_to_csv
from classify_loader import load_network_data
from classify_generalisation import run_standard_split_evaluation, run_group_generalization_test
from classify_analysis import plot_generalization_results

# Define the file paths to process (matches classify_conversion.py)
CSV_PATHS = [
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

def main():
    # Update Datasets
    print("Updating CSVs with Oxidation Zones ")
    for path in CSV_PATHS:
        try:
            add_oxidation_zone_to_csv(path)
        except Exception as e:
            print(f"Could not update {path}: {e}")

    # Load Data
    print("\nLoading and Preprocessing Data ")
    try:
        # Assumes data directories exist as per classify_loader default
        df = load_network_data() 
        if df.empty:
            print("Error: No data loaded. Please check your 'mito_data' folder structure.")
            return
        print(f"Successfully loaded dataset with {len(df)} samples.")
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    #  Standard Evaluation
    print("\n Running Standard Split Evaluation ")
    try:
        run_standard_split_evaluation(df)
    except Exception as e:
        print(f"Error during standard evaluation: {e}")

    #  Generalization Tests
    print("\nRunning Generalization Tests ===")
    try:
        gen_results = run_group_generalization_test(df)
        
        # Save results specifically for the analysis script
        results_file = "generalization_results.csv"
        gen_results.to_csv(results_file, index=False)
        print(f"Generalization results saved to '{results_file}'")
        
        # Analysis/Plotting
        print("\n Generating Analysis Plots ")
        plot_generalization_results(results_file)
        
    except Exception as e:
        print(f"Error during generalization tests or plotting: {e}")

    print("\n=== Execution Complete ===")

if __name__ == "__main__":
    main()