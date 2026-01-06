import itertools
import random
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from classify_eval import get_rf_model, calculate_metrics


def prepare_features_targets(df):
    """Separates features (X) and target (Y) from the dataframe."""
    cols_to_drop = ['oxidation_zone', 'target_var']
    # Drop columns if they exist in df
    X = df.drop([c for c in cols_to_drop if c in df.columns], axis=1)
    Y = df['oxidation_zone']
    return X, Y

def run_fixed_group_generalization(
    df,
    train_group_size=3,
    per_test_group=126,
    random_state=0
):
    """
    Strategy: Fixed Test Group + Randomized Training Group Combinations
    (Classification version of the previously agreed strategy)
    """

    print(
        f"\n--- Fixed Group Generalization "
        f"(Train on {train_group_size}, Test on 1) ---"
    )

    random.seed(random_state)

    target_col = "oxidation_zone"
    unique_groups = sorted(df["group_id"].unique())
    results = []

    for test_group in unique_groups:

        # All other groups available for training
        train_pool = [g for g in unique_groups if g != test_group]

        # All possible training combinations
        all_combos = list(
            itertools.combinations(train_pool, train_group_size)
        )

        random.shuffle(all_combos)
        selected_combos = all_combos[:per_test_group]

        test_df = df[df["group_id"] == test_group]
        test_group_name = (
            test_df["file_name"].iloc[0]
            if "file_name" in test_df.columns
            else f"Group {test_group}"
        )

        print(
            f"Testing on {test_group_name} "
            f"({len(selected_combos)} combos)"
        )

        for train_groups in selected_combos:

            train_df = df[df["group_id"].isin(train_groups)]

            # Prepare features
            X_train, y_train = prepare_features_targets(train_df)
            X_test, y_test = prepare_features_targets(test_df)

            model = get_rf_model(random_state)

            # 5-fold CV on training groups ONLY
            kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
            cv_scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=kf,
                scoring="accuracy",
                n_jobs=-1
            )

            # Train on full training groups
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            test_metrics = calculate_metrics(y_test, y_pred)

            results.append({
                "train_group_size": train_group_size,
                "test_group": test_group,
                "test_group_name": test_group_name,
                "train_groups": train_groups,
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
                **{f"test_{k}": v for k, v in test_metrics.items()}
            })

    return (
        pd.DataFrame(results)
        .sort_values("test_f1", ascending=False)
        .reset_index(drop=True)
    )
