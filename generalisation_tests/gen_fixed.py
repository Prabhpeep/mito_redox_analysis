import pandas as pd
import itertools
import gen_eval as utils

def run_strategy(groups, group_files, train_group_size=3):
    """
    Strategy: Fixed Test Group, Exhaustive Training on `train_group_size` groups.
    This strategy produces all possible combinations of 3 training groups and 1 testing group.
    For each of the 10 groups chosen for testing, we have 9 C 3 remaining groups for training,
    giving total 10 * 126 = 1260 combinations to test generalisation performance.
    
    """
    print(f"\n--- Running Fixed Strategy (Test 1 Group, Train on {train_group_size}) ---")
    results = []
    target_col = "cc_pixel_intensity_ratio"
    total = len(groups)

    for test_idx in range(total):
        # Pool of groups available for training
        train_pool = [g for g in range(total) if g != test_idx]
        
        # Exhaustive combinations
        combos = list(itertools.combinations(train_pool, train_group_size))
        
        print(f"Testing on Group {test_idx} ({len(combos)} combinations)...")

        for train_idx in combos:
            train_df = pd.concat([groups[k] for k in train_idx])
            test_df  = groups[test_idx]
            
            X_gen = test_df.drop(columns=["group_id", target_col, "line_id"], errors='ignore')
            y_gen = test_df[target_col]

            # Evaluate
            metrics = utils.evaluate_model_performance(train_df, X_gen, y_gen)
            
            results.append({
                "test_group": group_files[test_idx],
                "train_groups": [group_files[k] for k in train_idx],
                "R2_Gen_Test": metrics['gen_test_score'],
                "R2_Internal_Test": metrics['internal_test_score'],
                "CV_Mean": metrics['cv_mean']
            })

    return pd.DataFrame(results).sort_values("R2_Gen_Test", ascending=False)