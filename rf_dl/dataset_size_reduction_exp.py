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

def plot_experiment_results(results, output_path=None, title="", is_networked=True):
    """
    Plots the Train vs Test scores across different dataset sizes.
    """
    sizes = results['train_size']
    
    # Using the cleaner color logic we set up earlier
    color = '#CC0000' if is_networked else '#333333'
        
    # 1. Make the figure much more compact and square
    plt.figure(figsize=(6, 5))
    
    # Thicker line (4), bigger dots on the line (markersize=12)
    plt.plot(sizes, results['test_score'], 'o-', label='Test Score (R²)', color=color, linewidth=4, markersize=12)
    
    ax = plt.gca()
    ax.invert_xaxis() # Reverse the x axis
    
    # 2. Remove top and right borders to match reference
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if title:
        plt.title(title)
        
    # Updated Y-label to match reference
    plt.ylabel("R² (Test set)", fontsize=28)
    
    # 3. Explicit Y-axis intervals to match reference exactly
    plt.yticks([0, 0.25, 0.5, 0.75, 1.0])
    
    # 4. Format X-axis with 2 decimal places and capital 'K'
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    # Reduced number of bins so the vertical text doesn't overlap
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7)) 
    
    def format_k(x, pos):
        if x >= 1000:
            # Formats to 2 decimal places, e.g., 33.58K
            return f'{x/1000:.2f}K'.replace('.00K', 'K')
        return f'{int(x)}'
        
    ax.xaxis.set_major_formatter(FuncFormatter(format_k))
    
    # 5. Rotate X labels 90 degrees and make tick marks thicker
    plt.xticks(rotation=90)

    ax.tick_params(axis='y', labelsize=24, width=2, length=8) 
    ax.tick_params(axis='x', labelsize=24, width=2, length=8) 
    
    plt.ylim(0, 1.05) 
    
    # Removed the box around the legend to keep it clean
    plt.legend(frameon=False, fontsize=18) 
    plt.grid(False)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Experiment plot saved to {output_path}")
    
    plt.show()