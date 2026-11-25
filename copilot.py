"""Retention Copilot: build validated policy-aware retention briefs."""

import json
import os
import sqlite3

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from typing import Literal

from features import CAT_FEATURES, NUM_FEATURES, build_features
from llm_client import call_llm
from part_d import build_collection, retrieve

DB_PATH = "data/tataplay.db"
MODEL_PATH = "outputs/churn_model.joblib"


class RetentionBrief(BaseModel):
    """Validated retention brief returned by the language model."""

    subscriber_id: str
    risk_band: Literal["High", "Medium", "Low"]
    recommended_action: str
    policy_source: str


def read_subscriber(subscriber_id: str, subscribers: pd.DataFrame, recharges: pd.DataFrame):
    """Read one subscriber and compute its churn-model feature row."""
    subscriber = subscribers[subscribers["subscriber_id"] == subscriber_id].copy()

    if subscriber.empty:
        raise ValueError(f"Unknown subscriber: {subscriber_id}")

    features = build_features(subscriber, recharges)
    return features.iloc[0]


def derive_risk_band(probability: float) -> str:
    """Map a churn probability to a retention risk band."""
    if probability >= 0.40:
        return "High"
    if probability >= 0.20:
        return "Medium"
    return "Low"


def build_brief(
    subscriber_id: str,
    subscribers: pd.DataFrame,
    recharges: pd.DataFrame,
    segments: pd.DataFrame,
    collection,
    model,
) -> dict:
    """Create and validate one retention brief for a subscriber."""
    feature_row = read_subscriber(subscriber_id, subscribers, recharges)

    stored_probability = pd.read_csv("outputs/test_scores.csv")
    stored_probability = stored_probability[
        stored_probability["subscriber_id"] == subscriber_id
    ]

    if stored_probability.empty:
        raise ValueError(f"No test score found for {subscriber_id}")

    probability = float(stored_probability["churn_probability"].iloc[0])

    X = pd.DataFrame([feature_row[NUM_FEATURES + CAT_FEATURES]])
    X = X.replace({None: np.nan})
    rescored_probability = float(model.predict_proba(X)[:, 1][0])

    assert abs(probability - rescored_probability) < 0.0001

    risk_band = derive_risk_band(probability)
    tenure_months = int(feature_row["tenure_months"])
    failed_recharges = int(feature_row["n_failed_6m"])

    segment_name = segments.loc[
        segments["subscriber_id"] == subscriber_id, "segment_name"
    ].iloc[0]

    query = (
        f"retention offer for a {risk_band} risk subscriber with "
        f"{tenure_months} months tenure and {failed_recharges} failed recharges"
    )
    retrieved = retrieve(collection, query, k=3)

    context = "\n\n".join(
        f"[{item['source']}]\n{item['text']}" for item in retrieved
    )

    user_prompt = (
        f"FACTS:\n"
        f"subscriber_id: {subscriber_id}\n"
        f"risk_band: {risk_band}\n"
        f"tenure_months: {tenure_months}\n"
        f"failed_recharges_6m: {failed_recharges}\n"
        f"segment: {segment_name}\n\n"
        f"POLICY CONTEXT:\n{context}"
    )

    messages = [
        {
            "role": "system",
            "content": (
                "RETENTION BRIEF.\n"
                "Reply with one JSON object only.\n"
                "Use only the policy context."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]

    reply = call_llm(messages)

    try:
        data = json.loads(reply)
        brief = RetentionBrief(**data)
    except (json.JSONDecodeError, ValidationError):
        retry_messages = messages + [
            {"role": "assistant", "content": reply},
            {
                "role": "user",
                "content": (
                    "VALIDATION ERROR: Return one valid JSON object with "
                    "subscriber_id, risk_band, recommended_action and "
                    "policy_source."
                ),
            },
        ]
        reply = call_llm(retry_messages)
        data = json.loads(reply)
        brief = RetentionBrief(**data)

    assert brief.subscriber_id == subscriber_id
    assert brief.risk_band == risk_band
    assert brief.policy_source == "retention_offers_policy.txt"

    return {
        "subscriber_id": subscriber_id,
        "churn_probability": probability,
        "risk_band": risk_band,
        "tenure_months": tenure_months,
        "failed_recharges_6m": failed_recharges,
        "segment_name": segment_name,
        "brief": brief.model_dump(),
    }


def main() -> None:
    os.makedirs("outputs", exist_ok=True)

    subscribers = pd.read_csv("data/clean/subscribers.csv")
    recharges = pd.read_csv("data/clean/recharges.csv")
    segments = pd.read_csv("outputs/segments.csv")
    priority = pd.read_csv("outputs/retention_priority.csv")

    model = joblib.load(MODEL_PATH)
    collection = build_collection()

    output = []

    for subscriber_id in priority["subscriber_id"]:
        output.append(
            build_brief(
                subscriber_id,
                subscribers,
                recharges,
                segments,
                collection,
                model,
            )
        )

    with open("outputs/retention_briefs.jsonl", "w", encoding="utf-8") as f:
        for item in output:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    briefs = pd.DataFrame(
        [
            {
                "subscriber_id": item["subscriber_id"],
                "risk_band": item["risk_band"],
                "recommended_action": item["brief"]["recommended_action"],
            }
            for item in output
        ]
    )

    print("\n=== RETENTION BRIEFS ===")
    print("valid briefs:", len(briefs))
    print("risk band counts:")
    print(briefs["risk_band"].value_counts().to_string())
    print("recommended action counts:")
    print(briefs["recommended_action"].value_counts().to_string())

    print("\nRetention Copilot complete.")


if __name__ == "__main__":
    main()
