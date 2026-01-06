import os
import pandas as pd
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
    nnet_dfs = []
    
    # Load processed individual files
    if os.path.exists(net_dir):
        for f in os.listdir(net_dir):
            if f.endswith(".csv"):
                df = pd.read_csv(os.path.join(net_dir, f))
                net_dfs.append(dp.process_single_df(df, is_network=True))

    if os.path.exists(nnet_dir):
        for f in os.listdir(nnet_dir):
            if f.endswith(".csv"):
                df = pd.read_csv(os.path.join(nnet_dir, f))
                nnet_dfs.append(dp.process_single_df(df, is_network=False))
                
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

        # Dataset Size Reduction Experiment

        print("\n=== Running Dataset Size Reduction Experiment ===")
        # We use the full X, y from the network dataset
        X_full = pd.concat([X_train, X_test])
        y_full = pd.concat([y_train, y_test])
        
        # Run experiment using an unfitted base model
        base_rf = ev.get_default_model()
        
        experiment_results = dsr.run_reduction_experiment(
            X_full, 
            y_full, 
            base_model=base_rf
        )
        
        dsr.plot_experiment_results(
            experiment_results, 
            output_path="analysis_outputs/dataset_reduction_experiment.png"
        )
        
    print("\nAll tasks completed successfully.")

if __name__ == "__main__":
    main()