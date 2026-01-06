import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

def get_rf_model(random_state=0):
    """
    Returns the RandomForestClassifier with the tuned hyperparameters
    """
    
    return RandomForestClassifier(
        n_estimators=415,
        max_depth=50,
        min_samples_split=6,
        min_samples_leaf=10,
        class_weight='balanced',
        max_features='log2',
        ccp_alpha=0,
        random_state=random_state,
        n_jobs=-1
    )

def calculate_metrics(y_true, y_pred):
    """
    Calculates dictionary of classification metrics.
    """
    return {
        'accuracy':  accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall':    recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1':        f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }

def get_confusion_matrix(y_true, y_pred, labels=[0, 1, 2]):
    """
    Returns the confusion matrix for the predictions.
    """
    return confusion_matrix(y_true, y_pred, labels=labels)