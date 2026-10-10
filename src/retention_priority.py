import pandas as pd
import numpy as np


# ============================================================
# FILE PATHS
# ============================================================

INPUT_FILE = "data/risk_segmented_customers.csv"

OUTPUT_FILE = "data/retention_priority.csv"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    INPUT_FILE
)


# ============================================================
# ESTIMATED CUSTOMER VALUE
# ============================================================
#
# This is a student-project approximation.
#
# Estimated value is based on:
#
# Monthly Charges × Remaining Value Factor
#
# We use tenure to give longer-tenure customers
# additional value.
# ============================================================

df["estimated_customer_value"] = (
    df["monthly_charges"] *
    (1 + df["tenure_months"] / 12)
)


# ============================================================
# EXPECTED REVENUE AT RISK
# ============================================================

df["expected_revenue_at_risk"] = (
    df["estimated_customer_value"] *
    df["churn_probability"]
)


# ============================================================
# RETENTION PRIORITY SCORE
# ============================================================
#
# Higher churn probability + higher customer value
# = higher priority.
# ============================================================

df["retention_priority_score"] = (
    df["churn_probability"] *
    df["estimated_customer_value"]
)


# ============================================================
# PRIORITY CATEGORY
# ============================================================

def classify_priority(row):

    if (
        row["risk_level"] == "HIGH"
        and
        row["expected_revenue_at_risk"] >= 150
    ):

        return "CRITICAL"

    elif (
        row["risk_level"] == "HIGH"
    ):

        return "HIGH"

    elif (
        row["risk_level"] == "MEDIUM"
        and
        row["expected_revenue_at_risk"] >= 100
    ):

        return "MEDIUM-HIGH"

    elif (
        row["risk_level"] == "MEDIUM"
    ):

        return "MEDIUM"

    else:

        return "LOW"


df["retention_priority"] = (
    df.apply(
        classify_priority,
        axis=1
    )
)


# ============================================================
# RECOMMENDED ACTION
# ============================================================

def recommended_action(priority):

    if priority == "CRITICAL":

        return (
            "Immediate personalized retention intervention"
        )

    elif priority == "HIGH":

        return (
            "Priority retention campaign"
        )

    elif priority == "MEDIUM-HIGH":

        return (
            "Targeted retention campaign"
        )

    elif priority == "MEDIUM":

        return (
            "Monitor and engage customer"
        )

    else:

        return (
            "Normal customer engagement"
        )


df["retention_action"] = (
    df["retention_priority"]
    .apply(recommended_action)
)


# ============================================================
# SORT CUSTOMERS
# ============================================================

df = df.sort_values(
    "retention_priority_score",
    ascending=False
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print()
print("=" * 60)
print("RETENTION PRIORITY ANALYSIS")
print("=" * 60)

print()

print(
    "Priority distribution:"
)

print(
    df["retention_priority"]
    .value_counts()
)


print()

print(
    "Total estimated customer value:",
    f"${df['estimated_customer_value'].sum():,.2f}"
)


print()

print(
    "Total expected revenue at risk:",
    f"${df['expected_revenue_at_risk'].sum():,.2f}"
)


print()

print(
    "Top 10 retention-priority customers:"
)

print()

top_customers = df[
    [
        "customer_id",
        "churn_probability",
        "risk_level",
        "monthly_charges",
        "tenure_months",
        "estimated_customer_value",
        "expected_revenue_at_risk",
        "retention_priority",
        "retention_action"
    ]
].head(10)


print(
    top_customers.to_string(
        index=False
    )
)


print()

print("=" * 60)

print(
    f"Saved: {OUTPUT_FILE}"
)

print("=" * 60)