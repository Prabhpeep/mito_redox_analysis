import os
import pandas as pd
import numpy as np
import data_loader as dp
import tune_hyperparams as th
import eval as ev
import model_analysis as pa
import dataset_size_reduction_exp as dsr

def main():

    # Pooling & Processing

    net_dir = "../mito_data/nets"
    nnet_dir = "../mito_data/non-nets"
    
    # 1. Load the pooled datasets using data_loader.py logic
    if not os.path.exists(net_dir) or not os.path.exists(nnet_dir):
        print(f"Error: {net_dir} or {nnet_dir} directory not found.")
        return

    # Load raw concatenated data
    df_net_raw, df_nnet_raw = dp.load_raw_data(net_dir, nnet_dir)
    
    # Process the pooled datasets
    net_pooled, nnet_pooled = dp.pool_and_process_data(df_net_raw, df_nnet_raw)
    
    print(f"Final Pooled Net Shape: {net_pooled.shape}")
    print(f"Final Pooled Non Networked Shape: {nnet_pooled.shape}")

    # 2. Extract specific individual network groups for CV Analysis
    net_dfs = []
    net_group_names = []
    
    for f in sorted(os.listdir(net_dir)):
        if f.endswith(".csv"):
            df = pd.read_csv(os.path.join(net_dir, f))
            # Process single network DataFrame (pass empty df for non-network to reuse logic)
            processed_net, _ = dp.pool_and_process_data(df, pd.DataFrame())
            if not processed_net.empty:
                net_dfs.append(processed_net)
                net_group_names.append(f.replace("_net_sheet.csv", "").replace(".csv", ""))

    print(f"Processed {len(net_dfs)} individual Network groups for CV plotting.")

    # 3. Extract specific individual non networked groups for CV Analysis
    nnet_dfs = []
    nnet_group_names = []
    
    for f in sorted(os.listdir(nnet_dir)):
        if f.endswith(".csv"):
            df = pd.read_csv(os.path.join(nnet_dir, f))
            # Process single non networked DataFrame (pass empty df for network to reuse logic)
            _, processed_nnet = dp.pool_and_process_data(pd.DataFrame(), df)
            if not processed_nnet.empty:
                nnet_dfs.append(processed_nnet)
                nnet_group_names.append(f.replace("_non_networked_sheet.csv", "").replace(".csv", ""))

    print(f"Processed {len(nnet_dfs)} individual Non Networked groups for CV plotting.")


    # Modeling Pipeline

    
    # 1. Evaluate Non-Networks (Standard Params) ---
    print("\n=== Evaluating Non-Networked Mitochondria ===")
    nnet_results = ev.evaluate_model(nnet_pooled, model_name="Pooled Non Networked")
    
    # 2. Evaluate Network (Standard Params) ---
    print("\n=== Evaluating Network Mitochondria ===")
    # saving results for analysis
    net_results = ev.evaluate_model(net_pooled, model_name="Pooled Networks")
    

    # Analysis of Network Model through feature Importance and Prediction plot

    if net_results and net_results['model']:
        trained_model = net_results['model']
        cv_scores = net_results['cv_scores']
        
        # We need the specific Test set used for evaluation to plot Pred vs Actual
        # Since random_state=0 is fixed in eval.py, we can reproduce the split here.
        X_train, X_test, y_train, y_test = ev.get_train_test_split(net_pooled)
        
        # Generate Predictions
        y_pred = trained_model.predict(X_test)
        
        print("\n=== Running Prediction Analysis (Networks) ===")
        
        # Group-wise Scores for Networks calculation
        group_scores = []
        for df_g in net_dfs:
            target_col = 'element_pixel_intensity_ratio'
            drop_cols = ['line_id']
            cols_to_drop = [c for c in [target_col] + drop_cols if c in df_g.columns]
            X_g = df_g.drop(columns=cols_to_drop, axis=1)
            y_g = df_g[target_col]
            score = trained_model.score(X_g, y_g)
            group_scores.append(score)
            
        # Group-wise Scores for Non Networked calculation (Pre-computed for scaling)
        group_scores_nnet = []
        if nnet_results and nnet_results['model']:
            trained_model_nnet = nnet_results['model']
            for df_g in nnet_dfs:
                cols_to_drop = [c for c in [target_col] + drop_cols if c in df_g.columns]
                X_g = df_g.drop(columns=cols_to_drop, axis=1)
                y_g = df_g[target_col]
                score = trained_model_nnet.score(X_g, y_g)
                group_scores_nnet.append(score)

        # Global CV Scaling
        all_cv_scores = group_scores + group_scores_nnet
        if net_results: all_cv_scores.append(cv_scores.mean())
        if nnet_results: all_cv_scores.append(nnet_results['cv_scores'].mean())
        global_cv_min = min(0.0, min(all_cv_scores) - 0.1) if all_cv_scores else 0.0
        global_cv_max = min(1.0, max(all_cv_scores) + 0.1) if all_cv_scores else 1.0
        cv_ylim = (global_cv_min, global_cv_max)
        
        # Feature Importance
        # Make sure directory exists for plots
        os.makedirs("analysis_outputs", exist_ok=True)
        
        pa.plot_feature_importance(
            trained_model, 
            feature_names=X_train.columns, 
            top_n=10, 
            title="Top 10 Feature Importances (Networks)",
            is_networked=True,
            output_path="analysis_outputs/feature_importance_net.png"
        )
        
        #  Plot Predicted vs Actual
        pa.plot_predicted_vs_actual(
            y_test, 
            y_pred, 
            title="Random Forest: Predicted vs Actual (Networks)", 
            is_networked=True,
            output_path="analysis_outputs/pred_vs_actual_net.png"
        )
        
        # Plot MAE vs Binned Target
        pa.plot_mae_vs_binned_target(
            y_test, 
            y_pred, 
            step=0.025,
            title="Mean Absolute Error and Target Value Distribution (Networks)",
            is_networked=True,
            output_path="analysis_outputs/mae_vs_binned_net.png"
        )
        
        pa.plot_group_cv_scores(
            cv_scores, 
            net_group_names, 
            group_scores, 
            title=f"Networks (N={len(X_test)}) - Model CV Results", 
            y_lim=cv_ylim,
            is_networked=True,
            output_path="analysis_outputs/group_cv_scores_net.png"
        )

    if nnet_results and nnet_results['model']:
        trained_model_nnet = nnet_results['model']
        cv_scores_nnet = nnet_results['cv_scores']
        
        X_train_nnet, X_test_nnet, y_train_nnet, y_test_nnet = ev.get_train_test_split(nnet_pooled)
        y_pred_nnet = trained_model_nnet.predict(X_test_nnet)
        
        print("\n=== Running Prediction Analysis (Non Networked) ===")
        
        pa.plot_feature_importance(
            trained_model_nnet, 
            feature_names=X_train_nnet.columns, 
            top_n=10, 
            title="Top 10 Feature Importances (Non Networked)",
            is_networked=False,
            output_path="analysis_outputs/feature_importance_nnet.png"
        )
        
        pa.plot_predicted_vs_actual(
            y_test_nnet, 
            y_pred_nnet, 
            title="Random Forest: Predicted vs Actual (Non Networked)", 
            is_networked=False,
            output_path="analysis_outputs/pred_vs_actual_nnet.png"
        )
        
        pa.plot_mae_vs_binned_target(
            y_test_nnet, 
            y_pred_nnet, 
            step=0.025, 
            title="Mean Absolute Error and Target Value Distribution (Non Networked)",
            is_networked=False,
            output_path="analysis_outputs/mae_vs_binned_nnet.png"
        )
        
        pa.plot_group_cv_scores(
            cv_scores_nnet, 
            nnet_group_names, 
            group_scores_nnet, 
            title=f"Non Networked (N={len(X_test_nnet)}) - Model CV Results", 
            y_lim=cv_ylim,
            is_networked=False,
            output_path="analysis_outputs/group_cv_scores_nnet.png"
        )

        # Dataset Size Reduction Experiment

        print("\n=== Running Dataset Size Reduction Experiment ===")
        # We use the full X, y from the network dataset
        X_full = pd.concat([X_train, X_test])
        y_full = pd.concat([y_train, y_test])
        
        # Run experiment using an unfitted base model
        base_rf = ev.get_default_model()
        
        # Calculate reduction fraction limit based on nnet dataset size
        min_frac = len(nnet_pooled) / len(net_pooled)
        dynamic_fractions = np.linspace(min_frac, 1.0, 10).tolist()
        
        experiment_results = dsr.run_reduction_experiment(
            X_full, 
            y_full, 
            base_model=base_rf,
            fractions=dynamic_fractions
        )
        
        dsr.plot_experiment_results(
            experiment_results, 
            output_path="analysis_outputs/dataset_reduction_experiment.png"
        )
        
    print("\nAll tasks completed successfully.")

if __name__ == "__main__":
    main()