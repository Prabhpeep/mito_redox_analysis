import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


# Columns to keep during preprocessing
KEEP_COLUMNS_NET = [
    'feature_1', 'feature_2', 'feature_3', 'feature_4', 
    'feature_5', 'feature_6', 'feature_7', 'feature_8', 
    'feature_9_net', 'feature_10', 'target_var'
]

KEEP_COLUMNS_NNET = [
    'feature_1', 'feature_2', 'feature_3', 'feature_4', 
    'feature_5', 'feature_6', 'feature_7', 'feature_8', 
    'feature_9_nonnet', 'feature_10', 'target_var'
]


# Data Loading & Splitting Functions


def load_raw_data(net_path="net.csv", nnet_path="nnet.csv"):
    
    # Loads the raw CSV files.
    
    print(f"Loading raw data from {net_path} and {nnet_path}...")
    # Using low_memory=False to handle mixed types warning
    df_net = pd.read_csv(net_path, low_memory=False)
    df_standalone = pd.read_csv(nnet_path, low_memory=False)
    return df_net, df_standalone



    

# Preprocessing




def clean_dataframe(df, keep_columns, length_col_for_dedup='feature_9_net'):
    
    # Drops unnecessary columns and removes duplicates.
    
    # Keep only columns that exist in the df AND are in our keep list
    cols_to_keep = [c for c in keep_columns if c in df.columns]
    df = df[cols_to_keep].copy()
    
    # Drop duplicates
    if length_col_for_dedup in df.columns:
        df.drop_duplicates(subset=[length_col_for_dedup], inplace=True, ignore_index=True)
        
    return df


# Pooling & Transformation


def pool_and_process_data(net_files, standalone_files):
    
    # Loads specific subset files, cleans them, pools them, and applies log transforms.
    
    net_dfs = []
    nnet_dfs = []
    
    # Process Network Files
    for f in net_files:
        if os.path.exists(f):
            df = pd.read_csv(f)
            df = clean_dataframe(df, KEEP_COLUMNS_NET)
            net_dfs.append(df)
    
    # Process Standalone Files
    for f in standalone_files:
        if os.path.exists(f):
            df = pd.read_csv(f)
            df = clean_dataframe(df, KEEP_COLUMNS_NNET, 'element_length_(um)')
            nnet_dfs.append(df)
            
    # Pool datasets
    net_pooled = pd.concat(net_dfs, axis=0, ignore_index=True)
    nnet_pooled = pd.concat(nnet_dfs, axis=0, ignore_index=True)
    
    # Log Transformations
    cols_to_log_net = ['feature_1', 'feature_2', 'feature_9_net']
    cols_to_log_nnet = ['feature_1', 'feature_2', 'feature_9_nnet']
    
    for col in cols_to_log_net:
        if col in net_pooled.columns:
            net_pooled[col] = np.log(net_pooled[col])
            
    for col in cols_to_log_nnet:
        if col in nnet_pooled.columns:
            nnet_pooled[col] = np.log(nnet_pooled[col])
            
    # Handling NaNs and Infs
    net_pooled.replace([np.inf, -np.inf], np.nan, inplace=True)
    net_pooled.dropna(how="all", inplace=True) # Matches notebook logic cell 6
    
    nnet_pooled.replace([np.inf, -np.inf], np.nan, inplace=True)
    nnet_pooled.dropna(how="any", inplace=True) # Matches notebook logic cell 6
    
    # Filter Outliers 
    net_pooled = net_pooled[net_pooled['target_var'] <= 0.5]
    nnet_pooled = nnet_pooled[nnet_pooled['target_var'] <= 0.5]
    
    return net_pooled, nnet_pooled

# ==========================================
# 5. Main Execution Block
# ==========================================

def main():
    # 1. Load Raw Data
    # Ensure net.csv and nnet.csv are in the working directory
    if not os.path.exists("net.csv") or not os.path.exists("nnet.csv"):
        print("Error: net.csv or nnet.csv not found.")
        return

    df_net, df_nnet = load_raw_data()
    
    # 2. Split and Save Subgroups
    # Note: The notebook saves these to 'mito_data/nets' and 'mito_data/non-nets' implicitly in later cells
    # but initially to 'mito_data'. We will use specific dirs for clarity.
    output_dir_net = "mito_data/nets"
    output_dir_standalone = "mito_data/non-nets"
    
    
    # 3. Pool and Process
    # We pass the lists of files we just created
    net_pooled_df, nnet_pooled_df = pool_and_process_data(df_net, df_nnet)
    
    print(f"Final Pooled Net Shape: {net_pooled_df.shape}")
    print(f"Final Pooled Standalone Shape: {nnet_pooled_df.shape}")
    
    # 4. Optional: Save final pooled data
    net_pooled_df.to_csv("mito_data/net_pooled_final.csv", index=False)
    nnet_pooled_df.to_csv("mito_data/standalones_pooled_final.csv", index=False)
    print("Processing complete. Final files saved.")

if __name__ == "__main__":
    main()