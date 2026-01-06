import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.base import clone

def run_reduction_experiment(X, y, base_model, fractions=None, random_state=0):
    """
    Iteratively trains the model on smaller subsets of the training data 
    and evaluates performance on a fixed test set.
    
    Args:
        X, y: Full feature and target sets.
        base_model: The orignal model instance to clone and train.
        fractions: List of float fractions (0.0 to 1.0) to use for training size.
    """
    if fractions is None:
        fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
    # 1. Split into Train (for subsampling) and Test (fixed for evaluation)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    
    results = {
        'fraction': [],
        'train_size': [],
        'train_score': [],
        'test_score': []
    }
    
    print(f"--- Starting Dataset Size Reduction Experiment (Max Train: {len(X_train_full)}) ---")
    
    for frac in fractions:
        # Determine subset size
        if frac == 1.0:
            X_subset = X_train_full
            y_subset = y_train_full
        else:
            # We shuffle and take the first N samples
            subset_size = int(len(X_train_full) * frac)
            
            # Ensure we have enough samples to train
            if subset_size < 10:
                print(f"Skipping fraction {frac}: too few samples ({subset_size})")
                continue
                
            # Random sampling
            indices = np.random.choice(len(X_train_full), subset_size, replace=False)
            X_subset = X_train_full.iloc[indices] if hasattr(X_train_full, 'iloc') else X_train_full[indices]
            y_subset = y_train_full.iloc[indices] if hasattr(y_train_full, 'iloc') else y_train_full[indices]
            
        # Clone and Train Model
        model = clone(base_model)
        model.fit(X_subset, y_subset)
        
        # Evaluate
        tr_score = model.score(X_subset, y_subset)
        te_score = model.score(X_test, y_test)
        
        # Store results
        results['fraction'].append(frac)
        results['train_size'].append(len(X_subset))
        results['train_score'].append(tr_score)
        results['test_score'].append(te_score)
        
        print(f"Fraction: {frac:.1f} | Size: {len(X_subset):5d} | Train R2: {tr_score:.4f} | Test R2: {te_score:.4f}")
        
    return results

def plot_experiment_results(results, output_path=None):
    """
    Plots the Train vs Test scores across different dataset sizes.
    """
    sizes = results['train_size']
    
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, results['train_score'], 'o-', label='Train Score (R²)', color='blue', linewidth=2)
    plt.plot(sizes, results['test_score'], 'o-', label='Test Score (R²)', color='green', linewidth=2)
    
    plt.title("Model Performance vs. Dataset Size")
    plt.xlabel("Number of Training Samples")
    plt.ylabel("R² Score")
    plt.ylim(0, 1.05) # R2 is typically <= 1
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    if output_path:
        plt.savefig(output_path)
        print(f"Experiment plot saved to {output_path}")
    else:
        plt.show()