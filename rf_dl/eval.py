import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor

def get_train_test_split(df, target_col='target_var', drop_cols=['feature_10'], test_size=0.3):
    """
    Splits dataframe into X and y, then train and test sets.
    
    """
    # Filter out columns that actually exist
    cols_to_drop = [c for c in [target_col] + drop_cols if c in df.columns]
    
    X = df.drop(columns=cols_to_drop, axis=1)
    y = df[target_col]
    
    return train_test_split(X, y, test_size=test_size, random_state=0)

def get_default_model():
    """Returns the RandomForestRegressor with optimized defaults."""
    return RandomForestRegressor(
        max_depth=35, 
        n_estimators=75, 
        criterion='squared_error',
        random_state=0
    )

def evaluate_model(df, model=None, model_name="Model"):
    """
    Performs 5-fold CV and Train/Test evaluation.
    """
    if df.empty:
        print(f"[{model_name}] DataFrame is empty. Skipping.")
        return None

    if model is None:
        model = get_default_model()

    X_train, X_test, y_train, y_test = get_train_test_split(df)
    
    # 5-Fold Cross Validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    
    # Fit on full training set for final scores
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"\n--- Results for {model_name} ---")
    print(f"CV Scores: {cv_scores}")
    print(f"CV Mean (R2): {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
    print(f"Train Score:  {train_score:.3f}")
    print(f"Test Score:   {test_score:.3f}")
    
    return {
        'model': model,
        'cv_mean': cv_scores.mean(),
        'test_score': test_score
    }