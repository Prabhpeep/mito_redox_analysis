import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor

def get_train_test_split(df, target_col='target_var'):
    # Internal split for training validation.
    cols_to_drop = [target_col]
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns], axis=1)
    y = df[target_col]
    return train_test_split(X, y, test_size=0.3, random_state=0)

def evaluate_model_performance(train_df, X_external, y_external):
   
    """
    Trains a Random Forest on train_df and evaluates on:
    1. Internal CV
    2. Internal Test Split
    3. External Generalization Set
    """
    
    X_train, X_int_test, y_train, y_int_test = get_train_test_split(train_df)
    
    # Define Model (Standard Parameters from Bayes Search)
    model = RandomForestRegressor(
        ccp_alpha=0, max_depth=45, n_estimators=50, 
        n_jobs=-1, random_state=0
    )

    #  Cross Validation
    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=KFold(n_splits=5, shuffle=True, random_state=0), 
        scoring='r2', n_jobs=-1
    )

    #  Train and Internal Test
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    int_test_score = model.score(X_int_test, y_int_test)
    
    #  External Generalization Test
    gen_score = model.score(X_external, y_external)

    return {
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'train_score': train_score,
        'internal_test_score': int_test_score,
        'gen_test_score': gen_score
    }