"""Part C: segmentation, spend modelling and revenue at risk."""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
CLUSTER_FEATURES = [
    "tenure_months",
    "hours_watched_per_week",
    "monthly_recharge_inr",
]

SPEND_FEATURES = [
    "monthly_recharge_inr",
    "tenure_months",
    "hours_watched_per_week",
    "hd_hours_per_week",
    "sports_hours_per_week",
    "channels_watched",
    "support_calls_90d",
]

ALPHAS = [0.001, 0.01, 0.1, 0.5, 1, 2, 3, 10, 30, 100]


def regression_metrics(y_true, y_pred) -> dict:
    """Return R-squared, MAE and RMSE for a regression model."""
    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
    }


def main() -> None:
    os.makedirs("outputs", exist_ok=True)

    subscribers = pd.read_csv("data/clean/subscribers.csv")
    test_scores = pd.read_csv("outputs/test_scores.csv")

    cluster_data = subscribers[CLUSTER_FEATURES].copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(cluster_data)

    print("=== K-MEANS MODEL SELECTION ===")
    sweep = []
    for k in range(2, 9):
        model = KMeans(n_clusters=k, n_init=10, random_state=SEED)
        labels = model.fit_predict(scaled)
        sweep.append(
            {
                "k": k,
                "inertia": model.inertia_,
                "silhouette": metrics_silhouette(scaled, labels),
            }
        )
    sweep_frame = pd.DataFrame(sweep)
    print(sweep_frame.to_string(index=False))

    chosen_k = 4
    print("\nchosen K:", chosen_k)

    kmeans = KMeans(n_clusters=chosen_k, n_init=10, random_state=SEED)
    segment_ids = kmeans.fit_predict(scaled)

    profile = subscribers.copy()
    profile["segment_id"] = segment_ids
    profile_summary = (
        profile.groupby("segment_id")
        .agg(
            size=("subscriber_id", "size"),
            mean_tenure_months=("tenure_months", "mean"),
            mean_hours_watched=("hours_watched_per_week", "mean"),
            mean_monthly_recharge=("monthly_recharge_inr", "mean"),
            churn_rate=("churned", "mean"),
        )
        .reset_index()
    )
    print("\n=== SEGMENT PROFILE ===")
    print(profile_summary.to_string(index=False))

    premium_segment = int(
        profile_summary.loc[
            profile_summary["mean_monthly_recharge"].idxmax(), "segment_id"
        ]
    )
    at_risk_segment = int(
        profile_summary.loc[profile_summary["churn_rate"].idxmax(), "segment_id"]
    )

    segment_names = {}
    for segment_id in range(chosen_k):
        if segment_id == premium_segment:
            segment_names[segment_id] = "Premium"
        elif segment_id == at_risk_segment:
            segment_names[segment_id] = "At-Risk"
        else:
            segment_names[segment_id] = f"Segment {segment_id}"

    print("\nsegment names:", segment_names)

    segments = pd.DataFrame(
        {
            "subscriber_id": subscribers["subscriber_id"],
            "segment_id": segment_ids,
            "segment_name": [segment_names[value] for value in segment_ids],
        }
    )
    segments.to_csv("outputs/segments.csv", index=False)

    new_subscriber = pd.DataFrame(
        [
            {
                "tenure_months": 4,
                "hours_watched_per_week": 15,
                "monthly_recharge_inr": 520,
            }
        ]
    )
    new_scaled = scaler.transform(new_subscriber)
    distances = np.linalg.norm(kmeans.cluster_centers_ - new_scaled[0], axis=1)
    nearest_segment = int(np.argmin(distances))

    print("\n=== NEW SUBSCRIBER ASSIGNMENT ===")
    for segment_id, distance in enumerate(distances):
        print(
            f"segment {segment_id} ({segment_names[segment_id]}): "
            f"distance {distance:.3f}"
        )
    print("nearest segment:", nearest_segment, segment_names[nearest_segment])
    print("kmeans.predict:", int(kmeans.predict(new_scaled)[0]))

    correlation_columns = [
        "hours_watched_per_week",
        "hd_hours_per_week",
        "sports_hours_per_week",
        "channels_watched",
    ]
    correlation = subscribers[correlation_columns].corr()
    print("\n=== CORRELATION MATRIX ===")
    print(correlation.round(3).to_string())

    high_correlation = []
    for first in range(len(correlation_columns)):
        for second in range(first + 1, len(correlation_columns)):
            value = correlation.iloc[first, second]
            if abs(value) > 0.9:
                high_correlation.append(
                    (correlation_columns[first], correlation_columns[second], value)
                )
    print("\npairs with absolute correlation above 0.9:")
    for pair in high_correlation:
        print(pair)

    stayers = subscribers[subscribers["churned"] == 0].copy()
    X_spend = stayers[SPEND_FEATURES]
    y_spend = stayers["next_cycle_recharge_inr"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_spend, y_spend, test_size=0.20, random_state=SEED
    )

    print("\n=== SPEND MODEL SPLITS ===")
    print("train rows:", len(X_train))
    print("test rows:", len(X_test))

    linear = Pipeline(
        [("scaler", StandardScaler()), ("model", LinearRegression())]
    )
    ridge_search = GridSearchCV(
        Pipeline([("scaler", StandardScaler()), ("model", Ridge())]),
        param_grid={"model__alpha": ALPHAS},
        scoring="r2",
        cv=5,
    )
    lasso_search = GridSearchCV(
        Pipeline([("scaler", StandardScaler()), ("model", Lasso(max_iter=10000))]),
        param_grid={"model__alpha": ALPHAS},
        scoring="r2",
        cv=5,
    )

    linear.fit(X_train, y_train)
    ridge_search.fit(X_train, y_train)
    lasso_search.fit(X_train, y_train)

    ridge = ridge_search.best_estimator_
    lasso = lasso_search.best_estimator_

    print("\n=== SPEND MODEL RESULTS ===")
    for name, model in [
        ("linear regression", linear),
        ("ridge", ridge),
        ("lasso", lasso),
    ]:
        prediction = model.predict(X_test)
        metrics = regression_metrics(y_test, prediction)
        print(name, metrics)

    print("ridge best alpha:", ridge_search.best_params_["model__alpha"])
    print("ridge best CV R2:", ridge_search.best_score_)
    print("lasso best alpha:", lasso_search.best_params_["model__alpha"])
    print("lasso best CV R2:", lasso_search.best_score_)

    lasso_coefficients = lasso.named_steps["model"].coef_
    zero_features = [
        feature
        for feature, coefficient in zip(SPEND_FEATURES, lasso_coefficients)
        if coefficient == 0
    ]
    print("\nlasso zero coefficients:", zero_features)

    print("\n=== COEFFICIENT STABILITY ===")
    linear_coefficients = []
    ridge_coefficients = []

    for seed in range(5):
        sample = X_train.sample(frac=0.8, random_state=seed)
        sample_target = y_train.loc[sample.index]

        linear_sample = Pipeline(
            [("scaler", StandardScaler()), ("model", LinearRegression())]
        )
        ridge_sample = Pipeline(
            [("scaler", StandardScaler()), ("model", Ridge(alpha=30))]
        )

        linear_sample.fit(sample, sample_target)
        ridge_sample.fit(sample, sample_target)

        linear_coefficients.append(
            linear_sample.named_steps["model"].coef_[
                SPEND_FEATURES.index("hours_watched_per_week")
            ]
        )
        ridge_coefficients.append(
            ridge_sample.named_steps["model"].coef_[
                SPEND_FEATURES.index("hours_watched_per_week")
            ]
        )

    print("linear regression coefficient population std:",
          np.std(linear_coefficients, ddof=0))
    print("ridge coefficient population std:", np.std(ridge_coefficients, ddof=0))

    test_ids = test_scores["subscriber_id"]
    test_features = subscribers.set_index("subscriber_id").loc[test_ids]
    test_prediction = ridge.predict(test_features[SPEND_FEATURES])

    revenue = pd.DataFrame(
        {
            "subscriber_id": test_ids.values,
            "churn_probability": test_scores["churn_probability"].values,
            "predicted_next_recharge_inr": test_prediction,
        }
    )
    revenue["revenue_at_risk_inr"] = (
        revenue["churn_probability"] * revenue["predicted_next_recharge_inr"]
    )
    revenue = revenue.merge(segments, on="subscriber_id", how="left")
    revenue = revenue.merge(
        subscribers[["subscriber_id", "churned"]], on="subscriber_id", how="left"
    )

    priority = revenue.sort_values(
        "revenue_at_risk_inr", ascending=False
    ).head(15)

    priority = priority[
        [
            "subscriber_id",
            "churn_probability",
            "predicted_next_recharge_inr",
            "revenue_at_risk_inr",
            "segment_name",
            "churned",
        ]
    ]
    priority.to_csv("outputs/retention_priority.csv", index=False)

    print("\n=== RETENTION PRIORITY ===")
    print(priority.to_string(index=False))
    print("total revenue at risk: INR",
          round(priority["revenue_at_risk_inr"].sum(), 2))
    print("subscribers in top 15 who churned:", int(priority["churned"].sum()))

    joblib.dump(ridge, "outputs/spend_model.joblib")
    print("\nPart C complete.")


def metrics_silhouette(data, labels) -> float:
    """Compute the silhouette score without importing it at module level."""
    from sklearn.metrics import silhouette_score

    return silhouette_score(data, labels)


if __name__ == "__main__":
    main()
