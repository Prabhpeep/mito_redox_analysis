import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_generalization_results(results_csv="generalization_results.csv"):
    """
    Reads generalization results and plots Avg F1 Score per Experimental Group.
    """
    if not os.path.exists(results_csv):
        print(f"Results file {results_csv} not found. Run classify_generalisation.py first.")
        return

    df_results = pd.read_csv(results_csv)

    # Setup Plot
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")

    # Bar plot of F1 scores per group
    ax = sns.barplot(
        data=df_results,
        x='group_name',
        y='f1',
        hue='group_name',
        palette='viridis',
        legend=False
    )

    plt.title("Generalization Test: F1 Score per Experimental Group", fontsize=15)
    plt.ylabel("F1 Score (Weighted)", fontsize=12)
    plt.xlabel("Test Group (Unseen during training)", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.0)

    # Add text labels on bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 9), 
                    textcoords='offset points')

    plt.tight_layout()
    plt.show()
