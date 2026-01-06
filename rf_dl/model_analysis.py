import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_feature_importance(model, feature_names, top_n=10, output_path=None):
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
    sns.barplot(x='Importance', y='Feature', data=feature_imp_df.head(top_n), palette='viridis')
    plt.title(f'Top {top_n} Feature Importances')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Feature importance plot saved to {output_path}")
    else:
        plt.show()

def plot_predicted_vs_actual(y_test, y_pred, title="Predicted vs Actual", output_path=None):
    """
    Plots a regression scatter plot of Actual vs. Predicted values with a 45 Degree perfect fit line.
    """
    plt.figure(figsize=(8, 8))
    
    # Scatter plot
    plt.scatter(y_test, y_pred, alpha=0.5, color='royalblue', edgecolor='k', s=50, label='Data Points')
    
    # Perfect prediction line (y=x)
    min_val = min(np.min(y_test), np.min(y_pred))
    max_val = max(np.max(y_test), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Perfect Fit')
    
    plt.title(title)
    plt.xlabel("Actual Values (cc_pixel_intensity_ratio)")
    plt.ylabel("Predicted Values")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Regression plot saved to {output_path}")
    else:
        plt.show()