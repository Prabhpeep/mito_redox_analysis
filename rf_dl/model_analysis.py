import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score  # <--

def plot_feature_importance(model, feature_names, top_n=10, title="Top 10 Feature Importances", is_networked=True, output_path=None):
    if not hasattr(model, 'feature_importances_'):
        print("Error: The provided model does not have 'feature_importances_'.")
        return

    importances = model.feature_importances_
    
    feature_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    color = '#CC0000' if is_networked else 'black'
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Feature', y='Importance', data=feature_imp_df.head(top_n), 
                facecolor='none', edgecolor=color, linewidth=4)
    
    plt.title(title)
    plt.xlabel('Features')
    plt.ylabel('Importance Score', fontsize=28)
    
    # FIX: Make the y-axis numbers bigger
    plt.tick_params(axis='y', labelsize=28)
    
    plt.xticks(rotation=45, ha='right')
    plt.grid(False)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
    plt.show()

def plot_predicted_vs_actual(y_test, y_pred, title="Predicted vs Actual", is_networked=True, output_path=None):
    color = '#CC0000' if is_networked else 'black'
    plt.figure(figsize=(8, 8))
    
    # 1. Calculate the R² score
    r2 = r2_score(y_test, y_pred)
    
    # 2. Add an "invisible" element strictly for the legend text
    plt.plot([], [], ' ', label=f'R² Score: {r2:.3f}')
    
    # 3. Scatter plot (Notice we REMOVED the 'label' argument here)
    plt.scatter(y_test, y_pred, alpha=0.8, color=color, s=15)
    
    # Perfect prediction line (y=x)
    min_val = min(np.min(y_test), np.min(y_pred))
    max_val = max(np.max(y_test), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], color='gray', alpha=0.7, linestyle='--', linewidth=2, label='Ideal Prediction (y = x)')
    
    # Regression line (Least Squares)
    m, b = np.polyfit(y_test, y_pred, 1)
    x_line = np.array([min_val, max_val + 0.05])
    if is_networked:
        plt.plot(x_line, m*x_line + b, color='#800000', linewidth=4.0, label='Regression Line (Least Squares)')
    else:
        plt.plot(x_line, m*x_line + b, color='#333333', linewidth=4.0, label='Regression Line (Least Squares)')
    
    plt.title(title)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values", fontsize=28)

    current_ymin, current_ymax = plt.ylim()
    plt.ylim(current_ymin, max(current_ymax, 0.52))
    
    # Dialed down to 24px as previously discussed
    plt.tick_params(axis='y', labelsize=26)
     # Dialed down to 24px as previously discussed
    plt.tick_params(axis='x', labelsize=26)
    
    # This will now automatically pick up our invisible R² entry instead of the scatter points
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
    plt.show()

def plot_mae_vs_binned_target(y_test, y_pred, step=0.025, title="Mean Absolute Error and Target Value Distribution", is_networked=True, output_path=None):
    df = pd.DataFrame({
        'Actual': y_test,
        'Error': np.abs(y_test - y_pred)
    })
    
    bins = np.arange(0.0, 0.5 + step, step)
    df['Bin'] = pd.cut(df['Actual'], bins=bins, right=False)
    
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
    
    bin_stats['MAE'] = bin_stats['MAE'].fillna(0)
    total_count = bin_stats['Count'].sum()
    bin_stats['Percentage'] = (bin_stats['Count'] / total_count) * 100
    bin_centers = [b.left for b in bin_stats['Bin']]
    
    color = '#CC0000' if is_networked else 'black'
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    width = step * 0.8
    bars = ax1.bar(bin_centers, bin_stats['MAE'], width=width, align='edge',
                   facecolor='none', edgecolor=color, linewidth=2, label='Mean Absolute Error')
    
    ax1.set_xlabel("Binned Target Value")
    ax1.set_ylabel("Mean Absolute Prediction Error", fontsize=24)
    ax1.grid(False)
    
    # FIX: Make the y-axis numbers bigger on the left axis
    ax1.tick_params(axis='y', labelsize=28)
     # Dialed down to 24px as previously discussed
    plt.tick_params(axis='x', labelsize=28)
    
    ax2 = ax1.twinx()
    line_x = [b + width/2 for b in bin_centers]
    line, = ax2.plot(line_x, bin_stats['Percentage'], color=color, linewidth=2.5, marker='o',
                     label='Target Value Distribution (scaled)')
    
    ax2.set_ylabel("Percentage of Test Set", fontsize=24)
    ax2.set_ylim(bottom=0)
    ax2.grid(False)
    
    # FIX: Make the y-axis numbers bigger on the right axis
    ax2.tick_params(axis='y', labelsize=28)
    
    plt.title(title)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper center')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
    plt.show()

def plot_group_cv_scores(pooled_scores, group_names, group_scores, title="Networks - Model CV Results", y_lim=None, is_networked=True, output_path=None):
    color = '#CC0000' if is_networked else 'black'
    plt.figure(figsize=(12, 6))
    
    n_pooled = len(pooled_scores)
    labels = [f"CV Fold {i+1}" for i in range(n_pooled)] + [""] + list(group_names)
    values = list(pooled_scores) + [0.0] + list(group_scores)
    
    x_positions = np.arange(len(values))
    non_zero = np.array(values) != 0.0
    plt.bar(x_positions[non_zero], np.array(values)[non_zero], facecolor='none', edgecolor=color, linewidth=3)
    
    plt.ylabel("R² (test set)", color=color, fontsize=28)
    
    # FIX: Make the y-axis numbers bigger
    plt.tick_params(axis='y', labelsize=28)
    
    # FIX: Better padding logic for extreme negative values
    if y_lim:
        current_ymin, current_ymax = y_lim
    else:
        current_ymin = min(0.0, min(values))
        current_ymax = 1.0
        
    # We enforce a larger minimum padding to prevent the bar from cutting off
    y_padding = max(0.2, abs(current_ymin) * 0.2) 
    plt.ylim(current_ymin - y_padding, current_ymax + 0.1)
    
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.title(title, loc='left', color=color, weight='bold')
    
    plt.grid(False)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
    plt.show()