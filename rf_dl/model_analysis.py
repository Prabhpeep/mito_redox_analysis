import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_feature_importance(model, feature_names, top_n=10, title="Top 10 Feature Importances", output_path=None):
    """
    Extracts and plots feature importances from a trained Random Forest model.
    """
    if not hasattr(model, 'feature_importances_'):
        print("Error: The provided model does not have 'feature_importances_'.")
        return

    # Extract importances
    importances = model.feature_importances_
    
    # Create DataFrame of the results
    feature_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Feature', y='Importance', data=feature_imp_df.head(top_n), color='red')
    plt.title(title)
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.xticks(rotation=45, ha='right')
    plt.grid(False)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Feature importance plot saved to {output_path}")
    plt.show()

def plot_predicted_vs_actual(y_test, y_pred, title="Predicted vs Actual", output_path=None):
    """
    Plots a regression scatter plot of Actual vs. Predicted values with a 45 Degree perfect fit line.
    """
    plt.figure(figsize=(8, 8))
    
    # Scatter plot
    plt.scatter(y_test, y_pred, alpha=0.8, color='firebrick', s=15, label='Data Points')
    
    # Perfect prediction line (y=x)
    min_val = min(np.min(y_test), np.min(y_pred))
    max_val = max(np.max(y_test), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], color='black', alpha=0.7, linestyle='--', linewidth=1.2, label='Ideal Prediction (y = x)')
    
    # Regression line (Least Squares)
    m, b = np.polyfit(y_test, y_pred, 1)
    x_line = np.array([min_val, max_val])
    plt.plot(x_line, m*x_line + b, color='red', linewidth=2.5, label='Regression Line (Least Squares)')
    
    plt.title(title)
    plt.xlabel("Actual Values (element_pixel_intensity_ratio)")
    plt.ylabel("Predicted Values")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Regression plot saved to {output_path}")
    plt.show()

def plot_mae_vs_binned_target(y_test, y_pred, step=0.025, title="Mean Absolute Error and Target Value Distribution", output_path=None):
    """
    Plots Mean Absolute Error against binned target values along with percentage of test set.
    """
    df = pd.DataFrame({
        'Actual': y_test,
        'Error': np.abs(y_test - y_pred)
    })
    
    # Define bins
    bins = np.arange(0.0, 0.5 + step, step)
    df['Bin'] = pd.cut(df['Actual'], bins=bins, right=False)
    
    # Calculate MAE and count per bin
    try:
        bin_stats = df.groupby('Bin', observed=False).agg(
            MAE=('Error', 'mean'),
            Count=('Actual', 'count')
        ).reset_index()
    except TypeError:
        bin_stats = df.groupby('Bin').agg(
            MAE=('Error', 'mean'),
            Count=('Actual', 'count')
        ).reset_index()
    
    # In case of empty bins, fill with 0
    bin_stats['MAE'] = bin_stats['MAE'].fillna(0)
    
    total_count = bin_stats['Count'].sum()
    bin_stats['Percentage'] = (bin_stats['Count'] / total_count) * 100
    
    # Get bin edges for plotting
    bin_centers = [b.left for b in bin_stats['Bin']]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Bar plot for MAE on Left Y-Axis
    width = step * 0.8
    # Transparent bars with red edges
    bars = ax1.bar(bin_centers, bin_stats['MAE'], width=width, align='edge',
                   facecolor='none', edgecolor='red', linewidth=2, label='Mean Absolute Error')
    
    ax1.set_xlabel("Binned Target Value")
    ax1.set_ylabel("Mean Absolute Prediction Error")
    ax1.grid(False)
    
    # Line plot for Percentage on Right Y-Axis
    ax2 = ax1.twinx()
    # Centering the points above the bars
    line_x = [b + width/2 for b in bin_centers]
    line, = ax2.plot(line_x, bin_stats['Percentage'], color='red', linewidth=2.5, marker='o',
                     label='Target Value Distribution (scaled)')
    
    ax2.set_ylabel("Percentage of Test Set")
    ax2.set_ylim(bottom=0)
    ax2.grid(False)
    
    plt.title(title)
    
    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper center')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"MAE vs Binned Target plot saved to {output_path}")
    plt.show()

def plot_group_cv_scores(pooled_scores, group_names, group_scores, title="Networks - Model CV Results", y_lim=None, output_path=None):
    """
    Plots CV scores for pooled model vs individual group test scores.
    """
    plt.figure(figsize=(12, 6))
    
    # Combine the data, inserting a dummy value for the space (gap)
    n_pooled = len(pooled_scores)
    labels = [f"CV Fold {i+1}" for i in range(n_pooled)] + [""] + list(group_names)
    values = list(pooled_scores) + [0.0] + list(group_scores)
    
    # x positions
    x_positions = np.arange(len(values))
    
    # Create the bars (solid red)
    plt.bar(x_positions, values, color='red', edgecolor='red', linewidth=2)
    
    plt.ylabel("R² (test set)", color='red')
    if y_lim:
        plt.ylim(y_lim)
    else:
        min_val = min(0.0, min(values) - 0.1)
        plt.ylim(min_val, 1.0)
    
    # Customize the x ticks
    plt.xticks(x_positions, labels, rotation=45, ha='right')
    plt.grid(False)
    
    # Add the "CV for Pooled" subtitle text below the pooled bars
    pooled_center_x = (n_pooled - 1) / 2.0
    plt.text(pooled_center_x, -0.15, "CV for Pooled", color='darkred', weight='bold', 
             ha='center', va='top')
    
    plt.title(title, loc='left', color='red', weight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25) # Ensure space for text
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Group CV score plot saved to {output_path}")
    plt.show()