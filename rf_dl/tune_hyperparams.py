from sklearn.ensemble import RandomForestRegressor
from skopt import BayesSearchCV
from skopt.space import Real

def run_bayes_search(X_train, y_train, n_iter=50, random_state=0):
    """
    Runs Bayesian Optimization to find best Random Forest parameters.
    Returns the fitted search object.
    """
    print(f"Starting BayesSearchCV with {n_iter} iterations...")
    
    # Search space 
    search_spaces = {
        'n_estimators': (70, 100),
        'max_depth': (30, 50),
        'ccp_alpha': Real(0, 0.5),
    }
    
    opt = BayesSearchCV(
        estimator=RandomForestRegressor(random_state=random_state),
        search_spaces=search_spaces,
        n_iter=n_iter,
        cv=3, # Standard internal CV
        random_state=random_state,
        n_jobs=-1
    )
    
    opt.fit(X_train, y_train)
    
    print(f"Best Score: {opt.best_score_:.4f}")
    print(f"Best Params: {opt.best_params_}")
    
    return opt.best_estimator_