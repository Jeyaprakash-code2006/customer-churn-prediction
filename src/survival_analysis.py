import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    "data/customer_churn.csv"
)


# ============================================================
# COHORT ANALYSIS
# ============================================================

# Create tenure cohorts
df["tenure_cohort"] = pd.cut(
    df["tenure_months"],
    bins=[-1, 6, 12, 24, 36, 60, 120],
    labels=[
        "0-6 Months",
        "7-12 Months",
        "13-24 Months",
        "25-36 Months",
        "37-60 Months",
        "60+ Months"
    ]
)


cohort_analysis = (
    df.groupby(
        "tenure_cohort",
        observed=True
    )
    .agg(
        customers=("customer_id", "count"),
        churned=("churn", "sum"),
        avg_monthly_charges=(
            "monthly_charges",
            "mean"
        ),
        avg_satisfaction=(
            "satisfaction_score",
            "mean"
        )
    )
    .reset_index()
)


cohort_analysis["churn_rate"] = (
    cohort_analysis["churned"]
    / cohort_analysis["customers"]
    * 100
)


# ============================================================
# DISPLAY COHORT RESULTS
# ============================================================

print("\n======================================")
print("COHORT ANALYSIS")
print("======================================\n")

print(
    cohort_analysis.to_string(
        index=False
    )
)


# ============================================================
# SAVE COHORT RESULTS
# ============================================================

cohort_analysis.to_csv(
    "data/cohort_analysis.csv",
    index=False
)


# ============================================================
# COHORT CHURN CHART
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    cohort_analysis[
        "tenure_cohort"
    ].astype(str),
    cohort_analysis[
        "churn_rate"
    ]
)

plt.xlabel(
    "Customer Tenure Cohort"
)

plt.ylabel(
    "Churn Rate (%)"
)

plt.title(
    "Churn Rate by Tenure Cohort"
)

plt.xticks(
    rotation=30
)

plt.tight_layout()

plt.savefig(
    "data/cohort_churn_rate.png",
    dpi=300
)

plt.close()


# ============================================================
# KAPLAN-MEIER SURVIVAL ANALYSIS
# ============================================================

kmf = KaplanMeierFitter()


# In survival analysis:
#
# duration = customer's observed tenure
# event = whether churn occurred
#
# churn = 1 means event occurred
# churn = 0 means customer did not churn


duration = df[
    "tenure_months"
]

event = df[
    "churn"
]


kmf.fit(
    durations=duration,
    event_observed=event,
    label="Customer Survival"
)


# ============================================================
# SURVIVAL CURVE
# ============================================================

plt.figure(
    figsize=(10, 6)
)

kmf.plot_survival_function()

plt.title(
    "Kaplan-Meier Customer Survival Curve"
)

plt.xlabel(
    "Tenure (Months)"
)

plt.ylabel(
    "Probability of Remaining Active"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "data/kaplan_meier_curve.png",
    dpi=300
)

plt.close()


# ============================================================
# MEDIAN SURVIVAL
# ============================================================

print("\n======================================")
print("KAPLAN-MEIER ANALYSIS")
print("======================================\n")

print(
    "Median survival time:",
    kmf.median_survival_time_
)


print(
    "\nSurvival probability at selected months:"
)


for month in [6, 12, 24, 36, 48, 60]:

    try:

        probability = (
            kmf
            .survival_function_at_times(
                month
            )
            .iloc[0]
        )

        print(
            f"{month} months: "
            f"{probability:.4f}"
        )

    except Exception:

        pass


# ============================================================
# SAVE SURVIVAL TABLE
# ============================================================

survival_table = (
    kmf
    .survival_function_
    .reset_index()
)


survival_table.columns = [
    "tenure_months",
    "survival_probability"
]


survival_table.to_csv(
    "data/kaplan_meier_survival.csv",
    index=False
)


print("\nAnalysis completed.")

print(
    "\nFiles created:"
)

print(
    "data/cohort_analysis.csv"
)

print(
    "data/cohort_churn_rate.png"
)

print(
    "data/kaplan_meier_curve.png"
)

print(
    "data/kaplan_meier_survival.csv"
)