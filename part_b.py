"""Part B: leakage-safe churn prediction, tuning, thresholding and scoring."""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from features import (
    CAT_FEATURES,
    LEAKAGE_CATEGORICAL,
    LEAKAGE_NUMERIC,
    NUM_FEATURES,
    build_features,
    modelling_frame,
)

SEED = 42
MISSING_CHURN_COST = 1200
NEEDLESS_OFFER_COST = 300


def make_preprocessor(numeric, categorical) -> ColumnTransformer:
    """Create a leakage-safe preprocessing transformer."""
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )


def make_pipeline(model, numeric=NUM_FEATURES, categorical=CAT_FEATURES) -> Pipeline:
    """Return a pipeline containing preprocessing and an estimator."""
    return Pipeline(
        [
            ("preprocessor", make_preprocessor(numeric, categorical)),
            ("model", model),
        ]
    )


def classification_report(y_true, y_pred) -> dict:
    """Return the metrics required for model comparison."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def threshold_cost(y_true, probability, threshold) -> int:
    """Return the business cost of alerts produced at one threshold."""
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction).ravel()
    return MISSING_CHURN_COST * fn + NEEDLESS_OFFER_COST * fp


def evaluate_at_threshold(y_true, probability, threshold) -> dict:
    """Return confusion-matrix counts and metrics at one threshold."""
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction).ravel()
    metrics = classification_report(y_true, prediction)
    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        **metrics,
        "cost": threshold_cost(y_true, probability, threshold),
    }


def main() -> None:
    os.makedirs("outputs", exist_ok=True)

    subscribers = pd.read_csv("data/clean/subscribers.csv")
    recharges = pd.read_csv("data/clean/recharges.csv")

    X, y, feature_frame = modelling_frame(subscribers, recharges)

    print("=== PROBLEM FRAMING ===")
    print("class balance:", y.value_counts().to_dict())
    print("churn rate: {:.2%}".format(y.mean()))

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=0.25,
        stratify=y_trainval,
        random_state=SEED,
    )

    print("\n=== SPLITS ===")
    print("train:", X_train.shape[0], "churners:", int(y_train.sum()))
    print("validation:", X_val.shape[0], "churners:", int(y_val.sum()))
    print("test:", X_test.shape[0], "churners:", int(y_test.sum()))

    print("\n=== LEAKAGE EXPERIMENT ===")
    leaky_numeric = NUM_FEATURES + LEAKAGE_NUMERIC
    leaky_categorical = CAT_FEATURES + LEAKAGE_CATEGORICAL

    leaky_model = make_pipeline(
        RandomForestClassifier(n_estimators=100, random_state=SEED),
        numeric=leaky_numeric,
        categorical=leaky_categorical,
    )
    leaky_model.fit(X_train, y_train)
    leaky_prediction = leaky_model.predict(X_val)
    leaky_metrics = classification_report(y_val, leaky_prediction)
    print("with leakage columns:", leaky_metrics)

    clean_model = make_pipeline(
        RandomForestClassifier(n_estimators=100, random_state=SEED)
    )
    clean_model.fit(X_train, y_train)
    clean_prediction = clean_model.predict(X_val)
    clean_metrics = classification_report(y_val, clean_prediction)
    print("without leakage columns:", clean_metrics)

    preprocessor = make_preprocessor(NUM_FEATURES, CAT_FEATURES)
    preprocessor.fit(X_train)
    transformed = preprocessor.transform(X_train)

    print("\n=== PREPROCESSING ===")
    print("output columns:", transformed.shape[1])
    print("learned median age:", preprocessor.named_transformers_["num"]
          .named_steps["imputer"].statistics_[NUM_FEATURES.index("age")])
    print("scaler mean tenure_months:",
          preprocessor.named_transformers_["num"]
          .named_steps["scaler"].mean_[NUM_FEATURES.index("tenure_months")])
    print("full-data mean tenure_months:",
          X["tenure_months"].mean())
    print("missing values after transform:", int(np.isnan(transformed).sum()))

    majority_prediction = np.zeros(len(y_val), dtype=int)
    majority_metrics = classification_report(y_val, majority_prediction)
    print("\n=== BASELINE ===")
    print("majority class:", majority_metrics)

    logistic = make_pipeline(LogisticRegression(max_iter=1000))
    logistic.fit(X_train, y_train)
    logistic_prediction = logistic.predict(X_val)
    print("\nlogistic regression:", classification_report(y_val, logistic_prediction))

    tree = make_pipeline(DecisionTreeClassifier(random_state=SEED))
    tree.fit(X_train, y_train)
    print("\n unrestricted decision tree")
    print("train accuracy:", accuracy_score(y_train, tree.predict(X_train)))
    print("validation churn F1:", f1_score(y_val, tree.predict(X_val), zero_division=0))

    forest = make_pipeline(RandomForestClassifier(n_estimators=100, random_state=SEED))
    forest.fit(X_train, y_train)
    print("\ndefault random forest")
    print("train accuracy:", accuracy_score(y_train, forest.predict(X_train)))
    print("validation churn F1:", f1_score(y_val, forest.predict(X_val), zero_division=0))

    print("\n=== DECISION TREE DEPTH SWEEP ===")
    depth_results = []
    for depth in range(1, 11):
        depth_model = make_pipeline(
            DecisionTreeClassifier(max_depth=depth, random_state=SEED)
        )
        depth_model.fit(X_train, y_train)
        depth_results.append(
            {
                "max_depth": depth,
                "train_accuracy": accuracy_score(y_train, depth_model.predict(X_train)),
                "validation_f1": f1_score(
                    y_val, depth_model.predict(X_val), zero_division=0
                ),
            }
        )
    print(pd.DataFrame(depth_results).to_string(index=False))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    logistic_grid = GridSearchCV(
        make_pipeline(LogisticRegression(max_iter=1000)),
        param_grid={"model__C": [0.01, 0.1, 1, 10, 100]},
        scoring="f1",
        cv=cv,
        n_jobs=-1,
    )
    logistic_grid.fit(X_train, y_train)

    forest_grid = GridSearchCV(
        make_pipeline(RandomForestClassifier(random_state=SEED)),
        param_grid={
            "model__n_estimators": [50, 100, 150],
            "model__max_depth": [3, 4, 5, 6, 8],
            "model__max_features": [3, 4, 5],
        },
        scoring="f1",
        cv=cv,
        n_jobs=-1,
    )
    forest_grid.fit(X_train, y_train)

    print("\n=== GRID SEARCH ===")
    print("logistic regression combinations: 5; fits behind the scenes: 25")
    print("random forest combinations: 45; fits behind the scenes: 225")

    for name, search in [("logistic regression", logistic_grid), ("random forest", forest_grid)]:
        print(f"\n{name}")
        print("best parameters:", search.best_params_)
        print("best mean CV F1: {:.3f}".format(search.best_score_))
        print("CV F1 standard deviation: {:.3f}".format(search.cv_results_["std_test_score"][search.best_index_]))

    logistic_best = logistic_grid.best_estimator_
    forest_best = forest_grid.best_estimator_

    logistic_probability = logistic_best.predict_proba(X_val)[:, 1]
    forest_probability = forest_best.predict_proba(X_val)[:, 1]

    thresholds = np.arange(0.10, 0.91, 0.05)

    logistic_threshold_costs = [
        threshold_cost(y_val, logistic_probability, threshold)
        for threshold in thresholds
    ]
    forest_threshold_costs = [
        threshold_cost(y_val, forest_probability, threshold)
        for threshold in thresholds
    ]

    logistic_threshold = float(thresholds[int(np.argmin(logistic_threshold_costs))])
    forest_threshold = float(thresholds[int(np.argmin(forest_threshold_costs))])

    print("\n=== THRESHOLD SELECTION ON VALIDATION ===")
    print("logistic regression best threshold:", logistic_threshold)
    print("logistic regression validation cost: INR",
          min(logistic_threshold_costs))
    print("random forest best threshold:", forest_threshold)
    print("random forest validation cost: INR", min(forest_threshold_costs))

    logistic_test_probability = logistic_best.predict_proba(X_test)[:, 1]
    forest_test_probability = forest_best.predict_proba(X_test)[:, 1]

    logistic_test = evaluate_at_threshold(
        y_test, logistic_test_probability, logistic_threshold
    )
    forest_test = evaluate_at_threshold(
        y_test, forest_test_probability, forest_threshold
    )

    print("\n=== TEST EVALUATION ===")
    print("logistic regression:", logistic_test)
    print("random forest:", forest_test)

    test_churners = int(y_test.sum())
    no_action_cost = MISSING_CHURN_COST * test_churners
    offer_everybody_cost = (
        NEEDLESS_OFFER_COST * (len(y_test) - test_churners)
        + MISSING_CHURN_COST * test_churners
    )
    print("no-action cost: INR", no_action_cost)
    print("offer-everybody cost: INR", offer_everybody_cost)

    selected_is_forest = forest_test["cost"] <= logistic_test["cost"]
    selected_model = forest_best if selected_is_forest else logistic_best
    selected_threshold = forest_threshold if selected_is_forest else logistic_threshold
    selected_probability = (
        forest_test_probability if selected_is_forest else logistic_test_probability
    )
    selected_name = "random_forest" if selected_is_forest else "logistic_regression"

    print("\nselected model:", selected_name)

    if selected_is_forest:
        importances = selected_model.named_steps["model"].feature_importances_
        names = selected_model.named_steps["preprocessor"].get_feature_names_out()
        importance_frame = (
            pd.DataFrame({"feature": names, "importance": importances})
            .sort_values("importance", ascending=False)
        )
        print("\n=== TOP FIVE RANDOM FOREST FEATURES ===")
        print(importance_frame.head(5).to_string(index=False))

    logistic_coefficients = logistic_best.named_steps["model"].coef_[0]
    logistic_names = logistic_best.named_steps["preprocessor"].get_feature_names_out()
    coefficient_frame = (
        pd.DataFrame({"feature": logistic_names, "coefficient": logistic_coefficients})
        .sort_values("coefficient", key=abs, ascending=False)
    )
    print("\n=== LOGISTIC REGRESSION COEFFICIENTS ===")
    print(coefficient_frame.head(10).to_string(index=False))

    joblib.dump(selected_model, "outputs/churn_model.joblib")

    reloaded_model = joblib.load("outputs/churn_model.joblib")
    reloaded_probability = reloaded_model.predict_proba(X_test)[:, 1]
    original_prediction = (selected_probability >= selected_threshold).astype(int)
    reloaded_prediction = (reloaded_probability >= selected_threshold).astype(int)
    print("\nreloaded model reproduces original predictions:",
          bool(np.array_equal(original_prediction, reloaded_prediction)))

    test_ids = X_test.index
    scores = pd.DataFrame(
        {
            "subscriber_id": subscribers.loc[test_ids, "subscriber_id"].values,
            "churn_probability": selected_probability,
            "churned": y_test.values,
        }
    )
    scores.to_csv("outputs/test_scores.csv", index=False)

    print("\nPart B complete.")


if __name__ == "__main__":
    main()
