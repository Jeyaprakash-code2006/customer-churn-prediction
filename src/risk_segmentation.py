import os
import joblib
import pandas as pd
import numpy as np


# ============================================================
# FILE PATHS
# ============================================================

DATA_FILE = "data/customer_churn.csv"

MODEL_FILE = "models/logistic_regression.pkl"

PREPROCESSOR_FILE = "models/model_preprocessor.pkl"

OUTPUT_FILE = "data/risk_segmented_customers.csv"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_FILE)


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(MODEL_FILE)

preprocessor = joblib.load(
    PREPROCESSOR_FILE
)


# ============================================================
# FEATURES
# ============================================================

feature_columns = [
    "age",
    "gender",
    "senior_citizen",
    "tenure_months",
    "contract_type",
    "internet_service",
    "payment_method",
    "monthly_charges",
    "total_charges",
    "auto_payment",
    "discount",
    "support_tickets_30d",
    "logins_30d",
    "usage_hours_30d",
    "last_login_days",
    "satisfaction_score"
]


X = df[feature_columns].copy()


# ============================================================
# TRANSFORM DATA
# ============================================================

X_transformed = preprocessor.transform(X)


# ============================================================
# PREDICT CHURN PROBABILITY
# ============================================================

churn_probability = model.predict_proba(
    X_transformed
)[:, 1]


df["churn_probability"] = churn_probability


# ============================================================
# RISK SEGMENT
# ============================================================

def classify_risk(probability):

    if probability < 0.30:

        return "LOW"

    elif probability < 0.60:

        return "MEDIUM"

    else:

        return "HIGH"


df["risk_level"] = (
    df["churn_probability"]
    .apply(classify_risk)
)


# ============================================================
# BUSINESS ACTION
# ============================================================

def recommended_action(risk):

    if risk == "HIGH":

        return (
            "Proactive retention intervention"
        )

    elif risk == "MEDIUM":

        return (
            "Targeted retention campaign"
        )

    else:

        return (
            "Normal customer engagement"
        )


df["recommended_action"] = (
    df["risk_level"]
    .apply(recommended_action)
)


# ============================================================
# SORT BY RISK
# ============================================================

df = df.sort_values(
    "churn_probability",
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
print("=" * 50)
print("RISK SEGMENTATION")
print("=" * 50)

print()

print(
    df["risk_level"]
    .value_counts()
)


print()

print(
    "Risk percentage:"
)

risk_percentage = (
    df["risk_level"]
    .value_counts(normalize=True)
    * 100
)

print(
    risk_percentage.round(2)
)


print()

print(
    "Average churn probability:",
    f"{df['churn_probability'].mean() * 100:.2f}%"
)


print()

print(
    "Highest-risk customers:"
)

print(
    df[
        [
            "customer_id",
            "churn_probability",
            "risk_level",
            "monthly_charges",
            "satisfaction_score"
        ]
    ].head(10).to_string(index=False)
)


print()

print("=" * 50)

print(
    f"Saved: {OUTPUT_FILE}"
)

print("=" * 50)