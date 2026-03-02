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

    net_dir = "mito_data/nets"
    nnet_dir = "mito_data/non-nets"
    
    net_dfs = []
    net_group_names = []
    nnet_dfs = []
    
    # Load processed individual files
    if os.path.exists(net_dir):
        for f in sorted(os.listdir(net_dir)):
            if f.endswith(".csv"):
                df = pd.read_csv(os.path.join(net_dir, f))
                processed = dp.process_single_df(df, is_network=True)
                if not processed.empty:
                    net_dfs.append(processed)
                    net_group_names.append(f.replace("_net_sheet.csv", "").replace(".csv", ""))

    if os.path.exists(nnet_dir):
        for f in sorted(os.listdir(nnet_dir)):
            if f.endswith(".csv"):
                df = pd.read_csv(os.path.join(nnet_dir, f))
                processed = dp.process_single_df(df, is_network=False)
                if not processed.empty:
                    nnet_dfs.append(processed)
                
    print(f"Pooling {len(net_dfs)} Network groups and {len(nnet_dfs)} Non-Netwroked groups...")
    
    # Create final pooled datasets
    net_pooled = dp.pool_and_transform(net_dfs, is_network=True)
    nnet_pooled = dp.pool_and_transform(nnet_dfs, is_network=False)
    
    print(f"Final Pooled Net Shape: {net_pooled.shape}")
    print(f"Final Pooled Standalone Shape: {nnet_pooled.shape}")

    # Modeling Pipeline

    
    # 1. Evaluate Non-Networks (Standard Params) ---
    print("\n=== Evaluating Non-Netwokred Mitochondria ===")
    ev.evaluate_model(nnet_pooled, model_name="Pooled Standalones")
    
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
        
        print("\n=== Running Prediction Analysis ===")
        
        # Feature Importance
        # Make sure directory exists for plots
        os.makedirs("analysis_outputs", exist_ok=True)
        
        pa.plot_feature_importance(
            trained_model, 
            feature_names=X_train.columns, 
            top_n=10, 
            output_path="analysis_outputs/feature_importance.png"
        )
        
        #  Plot Predicted vs Actual
        pa.plot_predicted_vs_actual(
            y_test, 
            y_pred, 
            title="Random Forest: Predicted vs Actual (Networks)", 
            output_path="analysis_outputs/pred_vs_actual.png"
        )
        
        # Plot MAE vs Binned Target
        pa.plot_mae_vs_binned_target(
            y_test, 
            y_pred, 
            step=0.025, 
            output_path="analysis_outputs/mae_vs_binned.png"
        )
        
        # Group-wise Scores for Networks
        group_scores = []
        for df_g in net_dfs:
            target_col = 'element_pixel_intensity_ratio'
            drop_cols = ['line_id']
            cols_to_drop = [c for c in [target_col] + drop_cols if c in df_g.columns]
            X_g = df_g.drop(columns=cols_to_drop, axis=1)
            y_g = df_g[target_col]
            score = trained_model.score(X_g, y_g)
            group_scores.append(score)
            
        pa.plot_group_cv_scores(
            cv_scores, 
            net_group_names, 
            group_scores, 
            title=f"Networks (N={len(net_pooled)}) - Model CV Results", 
            output_path="analysis_outputs/group_cv_scores.png"
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