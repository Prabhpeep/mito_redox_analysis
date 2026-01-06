import pandas as pd
import random
import itertools
import gen_eval as utils

def run_strategy(groups, group_files, max_combos=8):
    """
    Strategy: Varying Train/Test Split Sizes (1:9 to 9:1).
    """
    print("\n Running Variable Split Strategy")
    results = []
    target_col = "target_var"
    total = len(groups)

    # j = number of test groups
    for j in range(1, total): 
        i = total - j # i = number of train groups
        
        # Random combinations
        all_combos = list(itertools.combinations(range(total), i))
        random.seed(0)
        random.shuffle(all_combos)
        
        for train_idx in all_combos[:max_combos]:
            test_pool = list(set(range(total)) - set(train_idx))
            test_idx = random.sample(test_pool, j)

            train_df = pd.concat([groups[k] for k in train_idx])
            test_df  = pd.concat([groups[k] for k in test_idx])
            
            X_gen = test_df.drop(columns=["group_id", target_col, "line_id"], errors='ignore')
            y_gen = test_df[target_col]

            # Evaluate
            metrics = utils.evaluate_model_performance(train_df, X_gen, y_gen)
            
            results.append({
                "n_train_groups": i,
                "n_test_groups": j,
                "train_files": [group_files[k] for k in train_idx],
                "test_files": [group_files[k] for k in test_idx],
                "R2_Gen_Test": metrics['gen_test_score'],
                "R2_Internal_Test": metrics['internal_test_score'],
                "CV_Mean": metrics['cv_mean']
            })
            
    return pd.DataFrame(results).sort_values("R2_Gen_Test", ascending=False)