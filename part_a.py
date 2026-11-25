"""Part A: audit, clean, store and query the raw exports."""

import os
import sqlite3

import numpy as np
import pandas as pd

RAW = "data/raw"
CLEAN = "data/clean"
DB_PATH = "data/tataplay.db"

CITY_MAP = {
    "bangalore": "Bengaluru",
    "bombay": "Mumbai",
    "madras": "Chennai",
    "calcutta": "Kolkata",
    "new delhi": "Delhi",
    "hyd": "Hyderabad",
}

VALID_CITIES = {
    "Chennai", "Mumbai", "Bengaluru", "Delhi", "Hyderabad", "Pune", "Kolkata"
}


def audit_frame(df: pd.DataFrame, name: str) -> None:
    """Print shape, dtypes and missing-value statistics for a raw table."""
    print(f"\n=== AUDIT: {name} ===")
    print("shape:", df.shape)
    print("\ndtypes:")
    print(df.dtypes.to_string())

    missing_count = df.isna().sum()
    missing_pct = (missing_count / len(df) * 100).round(2)
    audit = pd.DataFrame({"missing_count": missing_count, "missing_pct": missing_pct})
    print("\nmissing values:")
    print(audit.to_string())


def clean_subscribers(df: pd.DataFrame) -> pd.DataFrame:
    """Clean subscriber records without imputing missing values."""
    df = df.drop_duplicates().copy()

    city = df["city"].astype(str).str.strip().str.lower()
    df["city"] = city.map(CITY_MAP).fillna(city.str.title())
    df.loc[~df["city"].isin(VALID_CITIES), "city"] = np.nan

    auto = df["auto_renew"].astype(str).str.strip().str.lower()
    df["auto_renew"] = auto.map({"y": "Yes", "yes": "Yes", "n": "No", "no": "No"})

    age = pd.to_numeric(df["age"], errors="coerce")
    age = age.where(age.between(10, 100))
    df["age"] = age

    return df


def clean_recharges(df: pd.DataFrame, subscribers: pd.DataFrame) -> pd.DataFrame:
    """Clean recharge records without imputing unrelated missing values."""
    df = df.drop_duplicates().copy()

    valid_ids = set(subscribers["subscriber_id"])
    before = len(df)
    df = df[df["subscriber_id"].isin(valid_ids)].copy()
    print(f"recharges: removed {before - len(df)} rows with unknown subscriber_id")

    df["status"] = df["status"].astype(str).str.strip().str.lower()

    raw_amount = df["amount_inr"].astype(str).str.strip()
    numeric_amount = pd.to_numeric(raw_amount, errors="coerce")

    monthly_by_id = subscribers.set_index("subscriber_id")["monthly_recharge_inr"]
    rebuilt = df["subscriber_id"].map(monthly_by_id)
    rebuilt_count = int(numeric_amount.isna().sum())

    df["amount_inr"] = numeric_amount.fillna(rebuilt)
    df["recharge_date"] = pd.to_datetime(df["recharge_date"], errors="coerce")

    print(f"recharges: rebuilt {rebuilt_count} non-numeric amount_inr values")
    return df


def run_query(conn: sqlite3.Connection, query: str, title: str) -> pd.DataFrame:
    """Execute one SQL query and print its result."""
    result = pd.read_sql_query(query, conn)
    print(f"\n=== {title} ===")
    print(result.to_string(index=False))
    return result


def main() -> None:
    os.makedirs(CLEAN, exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    subscribers_raw = pd.read_csv(os.path.join(RAW, "subscribers.csv"))
    recharges_raw = pd.read_csv(os.path.join(RAW, "recharges.csv"))
    plans = pd.read_csv(os.path.join(RAW, "plans.csv"))

    audit_frame(subscribers_raw, "subscribers.csv")
    audit_frame(recharges_raw, "recharges.csv")

    print("\n=== RAW AUDIT FIGURES ===")
    print("subscriber rows:", len(subscribers_raw))
    print("exact duplicate subscriber rows:", len(subscribers_raw) - len(subscribers_raw.drop_duplicates()))
    print("distinct subscriber_id:", subscribers_raw["subscriber_id"].nunique())
    print("blank age:", int(subscribers_raw["age"].isna().sum()))
    age_num = pd.to_numeric(subscribers_raw["age"], errors="coerce")
    print("age outside 10-100:", int(((age_num < 10) | (age_num > 100)).sum()))
    print("blank payment_mode:", int(subscribers_raw["payment_mode"].isna().sum()))
    print("distinct raw city spellings:", subscribers_raw["city"].astype(str).str.strip().nunique())
    print("distinct raw auto_renew spellings:", subscribers_raw["auto_renew"].astype(str).str.strip().nunique())
    print("blank next_cycle_recharge_inr:", int(subscribers_raw["next_cycle_recharge_inr"].isna().sum()))

    print("recharge rows:", len(recharges_raw))
    print("exact duplicate recharge rows:", len(recharges_raw) - len(recharges_raw.drop_duplicates()))
    print("distinct recharge_id:", recharges_raw["recharge_id"].nunique())
    amount_numeric = pd.to_numeric(recharges_raw["amount_inr"].astype(str).str.strip(), errors="coerce")
    print("non-plain-number amount_inr values:", int(amount_numeric.isna().sum()))
    print("distinct raw status spellings:", recharges_raw["status"].astype(str).str.strip().nunique())
    known_ids = set(subscribers_raw["subscriber_id"])
    print("recharges with unknown subscriber_id:", int((~recharges_raw["subscriber_id"].isin(known_ids)).sum()))

    subscribers = clean_subscribers(subscribers_raw)
    recharges = clean_recharges(recharges_raw, subscribers)

    print("\n=== CLEANED SUBSCRIBERS ===")
    print("rows:", len(subscribers))
    print("city counts:")
    print(subscribers["city"].value_counts().sort_index().to_string())
    print("auto_renew counts:")
    print(subscribers["auto_renew"].value_counts().to_string())
    print("missing age:", int(subscribers["age"].isna().sum()))
    print("missing payment_mode:", int(subscribers["payment_mode"].isna().sum()))
    print("churned subscribers:", int(subscribers["churned"].sum()))
    print("churn rate: {:.2%}".format(subscribers["churned"].mean()))

    print("\n=== CLEANED RECHARGES ===")
    print("rows:", len(recharges))
    print("status counts:")
    print(recharges["status"].value_counts().to_string())
    successful_total = recharges.loc[recharges["status"] == "success", "amount_inr"].sum()
    print(f"total successful recharge amount: INR {successful_total:,.0f}")

    subscribers.to_csv(os.path.join(CLEAN, "subscribers.csv"), index=False)
    recharges.to_csv(os.path.join(CLEAN, "recharges.csv"), index=False)
    plans.to_csv(os.path.join(CLEAN, "plans.csv"), index=False)

    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    subscribers.to_sql("subscribers", conn, if_exists="replace", index=False)
    recharges.to_sql("recharges", conn, if_exists="replace", index=False)
    plans.to_sql("plans", conn, if_exists="replace", index=False)

    with open("sql/queries.sql", "r", encoding="utf-8") as f:
        sql = f.read()

    queries = [q.strip() for q in sql.split(";") if q.strip() and not q.strip().startswith("--")]
    titles = [
        "Q1 revenue by plan",
        "Q2 subscribers who never successfully recharged",
        "Q3 ranking within a city",
        "Q4 running total for TP00010",
    ]

    results = {}
    for title, query in zip(titles, queries):
        results[title] = run_query(conn, query, title)

    q1 = results["Q1 revenue by plan"]
    pandas_check = (
        recharges[recharges["status"] == "success"]
        .merge(subscribers[["subscriber_id", "plan_id"]], on="subscriber_id")
        .merge(plans, on="plan_id")
        .groupby("plan_name")["amount_inr"]
        .agg(["sum", "mean", "nunique"])
        .sort_values("sum", ascending=False)
    )
    print("\n=== Q1 PANDAS CONFIRMATION ===")
    print(pandas_check.to_string())

    q2 = results["Q2 subscribers who never successfully recharged"]
    print("\nQ2 tenure_months = 0 count:",
          int((q2["tenure_months"] == 0).sum()))

    q3 = results["Q3 ranking within a city"]
    rank_one = q3[q3["rank_value"] == 1].groupby("city").size()
    print("\n=== Q3 SUBSCRIBERS WITH RANK() = 1 ===")
    print(rank_one.to_string())
    print("\n=== Q3 KOLKATA FIRST FOUR ROWS ===")
    print(q3[q3["city"] == "Kolkata"].head(4).to_string(index=False))

    conn.close()
    print("\nPart A complete.")


if __name__ == "__main__":
    main()
