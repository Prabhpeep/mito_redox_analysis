import os
import pandas as pd
import numpy as np

# ==========================================


# Columns required for the generalization model
GEN_KEEP_COLUMNS = [
    'feature_1', 'feature_2', 'feature_3', 'feature_4', 
    'feature_5', 'feature_6', 'feature_7', 'feature_8', 
    'feature_9_net', 'feature_10', 'target_var'
]

def preprocess_for_gen(df):
    """
    Cleans and transforms a single group dataframe.
    """
    # Filter Columns
    cols_to_keep = list(set(GEN_KEEP_COLUMNS) & set(df.columns))
    df = df[cols_to_keep].copy()
    
    # Drop Duplicates
    if 'feature_9_net' in df.columns:
        df.drop_duplicates(subset=['feature_9_net'], inplace=True, ignore_index=True)

   
    # Log Transformations
    cols_to_log = ['feature_1', 'feature_9_net', 'feature_2']
    for col in cols_to_log:
        if col in df.columns:
            # Add small epsilon or handle existing logic if needed, 
            # but notebook uses direct np.log which assumes > 0
            df[col] = np.log(df[col])

    # Clean NaNs and Infs
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(how="any", inplace=True)
    
    # Filter Outliers
    if 'target_var' in df.columns:
        df = df[df['target_var'] <= 0.5]
    
    return df

def load_groups(data_dir):
    """
    Reads all '_net_sheet.csv' files from data_dir.
    Returns:
        groups (list of pd.DataFrame): The processed dataframes.
        files (list of str): The corresponding file paths.
    """
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} not found.")
        return [], []

    # Sort files to ensure deterministic ordering of groups
    files = sorted([
        os.path.join(data_dir, f) for f in os.listdir(data_dir)
        if f.endswith('_net_sheet.csv')
    ])
    
    groups = []
    print(f"Loading {len(files)} groups from {data_dir}...")
    
    for idx, file in enumerate(files):
        try:
            df = pd.read_csv(file)
            df = preprocess_for_gen(df)
            
            # Assign metadata required for tracking
            df['group_id'] = idx
            df['file_name'] = os.path.basename(file)
            
            groups.append(df)
        except Exception as e:
            print(f"Skipping {file} due to error: {e}")
            
    return groups, files