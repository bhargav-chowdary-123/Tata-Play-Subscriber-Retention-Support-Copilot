"""Shared feature engineering for Part B and the Retention Copilot."""

import numpy as np
import pandas as pd

SNAPSHOT_DATE = pd.Timestamp("2026-03-31")

NUM_FEATURES = [
    "tenure_months", "age", "add_on_count", "monthly_recharge_inr",
    "hours_watched_per_week", "hd_hours_per_week", "sports_hours_per_week",
    "channels_watched", "support_calls_90d",
    "n_success_6m", "n_failed_6m", "recency_days",
]

CAT_FEATURES = ["city", "plan_id", "payment_mode", "auto_renew"]

LEAKAGE_NUMERIC = ["next_cycle_recharge_inr"]
LEAKAGE_CATEGORICAL = ["closure_request_logged"]


def build_features(subscribers: pd.DataFrame, recharges: pd.DataFrame) -> pd.DataFrame:
    """Return one modelling row per subscriber using only pre-2026-04-01 data.

    Recharge aggregates are computed from clean recharge records as of the
    snapshot date. Subscribers with no recharge rows receive zero counts and a
    missing recency value.
    """
    df = subscribers.copy()

    successful = recharges[recharges["status"] == "success"]
    failed = recharges[recharges["status"] == "failed"]

    success_counts = successful.groupby("subscriber_id").size()
    failed_counts = failed.groupby("subscriber_id").size()
    last_success = successful.groupby("subscriber_id")["recharge_date"].max()

    df["n_success_6m"] = df["subscriber_id"].map(success_counts).fillna(0).astype(int)
    df["n_failed_6m"] = df["subscriber_id"].map(failed_counts).fillna(0).astype(int)

    recency = SNAPSHOT_DATE - df["subscriber_id"].map(last_success)
    df["recency_days"] = recency.dt.days

    df["n_success_6m"] = df["n_success_6m"].astype(float)
    df["n_failed_6m"] = df["n_failed_6m"].astype(float)

    return df


def modelling_frame(subscribers: pd.DataFrame, recharges: pd.DataFrame):
    """Return X, y and the full feature frame for churn modelling."""
    df = build_features(subscribers, recharges)
    y = df["churned"].astype(int)
    X = df[NUM_FEATURES + CAT_FEATURES].copy()
    return X, y, df
