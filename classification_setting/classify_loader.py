import pandas as pd
import numpy as np
import os

# Mapping for the target variable
ZONE_MAPPING = {'low': 0, 'intermediate': 1, 'high': 2}

def preprocess(df):
    """
    Applies filtering, feature selection, log transformations, 
    and target mapping to the dataframe.
    """
    # Keep only relevant columns
    cols_to_keep = [
    'feature_1', 'feature_2', 'feature_3', 'feature_4', 
    'feature_5', 'feature_6', 'feature_7', 'feature_8', 
    'feature_9_net', 'feature_10', 'target_var', 
    'oxidation_zone'
    ]
    df.drop(labels=df.columns.difference(cols_to_keep), axis=1, inplace=True)
    
    # Remove duplicates based on specific feature
    df.drop_duplicates(subset=['feature_9_net'], inplace=True, ignore_index=True)

    # Apply the mapping
    if 'oxidation_zone' in df.columns:
        df['oxidation_zone'] = df['oxidation_zone'].map(ZONE_MAPPING)

    # Log transformations
    df['feature_1'] = np.log(df['feature_1'])
    df['feature_9_net'] = np.log(df['feature_9_net'])
    df['feature_2'] = np.log(df['feature_2'])
    
    # Clean up Infinite/NaN values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(how="any", inplace=True)
    
    # Filter by pixel intensity ratio
    df = df[df['target_var'] <= 0.5]
    
    return df

def load_network_data(data_dir="mito_data/nets"):
    """
    Loads all _net_sheet.csv files from the directory, processes them,
    and returns a single pooled dataframe with group identifiers.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    group_files = sorted([
        os.path.join(data_dir, f) for f in os.listdir(data_dir)
        if f.endswith('_net_sheet.csv')
    ])

    groups = []
    for idx, file in enumerate(group_files):
        try:
            df = pd.read_csv(file)
            df = preprocess(df)
            df['group_id'] = idx
            df['file_name'] = os.path.basename(file)
            groups.append(df)
        except Exception as e:
            print(f"Error loading {file}: {e}")

    if not groups:
        return pd.DataFrame()

    net_pooled_df = pd.concat(groups, ignore_index=True)
    return net_pooled_df