import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt


# Columns to keep during preprocessing
KEEP_COLUMNS_NET = [
    'cc_length_(um)', 'nodes', 'edges', 'element_pixel_intensity_ratio', 'cc_max_PK', 'line_id',
    'diameter', 'element_length_(um)', 'normalized_length_by_networks', 'edge_density', 
    'node_density'
]

KEEP_COLUMNS_NNET = [
    'cc_length_(um)', 'nodes', 'edges', 'element_pixel_intensity_ratio', 'cc_max_PK', 'line_id', 
    'diameter', 'element_length_(um)', 'normalized_length_by_non_networked', 'edge_density', 
    'node_density'
    ]


# Data Loading & Splitting Functions


def load_raw_data(nets_dir="mito_data/nets", nnets_dir="mito_data/non-nets"):
    
    # Loads all raw CSV files from the specified directories and concatenates them.
    
    print(f"Loading raw data from '{nets_dir}' and '{nnets_dir}'...")
    
    # Load and pool net files
    net_files = glob.glob(os.path.join(nets_dir, "*.csv"))
    df_net = pd.concat([pd.read_csv(f, low_memory=False) for f in net_files], ignore_index=True) if net_files else pd.DataFrame()
    
    # Load and pool non networked files
    nnet_files = glob.glob(os.path.join(nnets_dir, "*.csv"))
    df_non_networked = pd.concat([pd.read_csv(f, low_memory=False) for f in nnet_files], ignore_index=True) if nnet_files else pd.DataFrame()
    
    return df_net, df_non_networked




def clean_dataframe(df, keep_columns, length_col_for_dedup='element_length_(um)'):
    
    # Drops unnecessary columns and removes duplicates.
    
    # Keep only columns that exist in the df AND are in our keep list
    cols_to_keep = [c for c in keep_columns if c in df.columns]
    df = df[cols_to_keep].copy()
    
    # Drop duplicates
    if length_col_for_dedup in df.columns:
        df.drop_duplicates(subset=[length_col_for_dedup], inplace=True, ignore_index=True)
        
    return df


# Pooling & Transformation


def pool_and_process_data(net_pooled, nnet_pooled):
    
    # Cleans the previously pooled dataframes and applies log transforms.
    
    # Clean datasets
    net_pooled = clean_dataframe(net_pooled, KEEP_COLUMNS_NET, 'element_length_(um)')
    nnet_pooled = clean_dataframe(nnet_pooled, KEEP_COLUMNS_NNET, 'element_length_(um)')
    
    # Log Transformations
    cols_to_log_net = ['normalized_length_by_networks', 'element_length_(um)', 'cc_length_(um)']
    cols_to_log_nnet = ['normalized_length_by_non_networked', 'element_length_(um)', 'cc_length_(um)']
    
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
    if not net_pooled.empty and 'element_pixel_intensity_ratio' in net_pooled.columns:
        net_pooled = net_pooled[net_pooled['element_pixel_intensity_ratio'] <= 0.5]
        
    if not nnet_pooled.empty and 'element_pixel_intensity_ratio' in nnet_pooled.columns:
        nnet_pooled = nnet_pooled[nnet_pooled['element_pixel_intensity_ratio'] <= 0.5]
    
    return net_pooled, nnet_pooled

# ==========================================
# 5. Main Execution Block
# ==========================================

def main():
    # 1. Load Raw Data
    nets_dir = "mito_data/nets"
    nnets_dir = "mito_data/non-nets"
    
    if not os.path.exists(nets_dir) or not os.path.exists(nnets_dir):
        print(f"Error: {nets_dir} or {nnets_dir} directory not found.")
        return

    df_net, df_nnet = load_raw_data(nets_dir, nnets_dir)
    
    # 2. Pool and Process
    # Pass the concatenated dataframes to be cleaned and processed
    net_pooled_df, nnet_pooled_df = pool_and_process_data(df_net, df_nnet)
    
    print(f"Final Pooled Net Shape: {net_pooled_df.shape}")
    print(f"Final Pooled Non Networked Shape: {nnet_pooled_df.shape}")
    
    # 4. Optional: Save final pooled data
    net_pooled_df.to_csv("mito_data/net_pooled_final.csv", index=False)
    nnet_pooled_df.to_csv("mito_data/non_networked_pooled_final.csv", index=False)
    print("Processing complete. Final files saved.")

if __name__ == "__main__":
    main()