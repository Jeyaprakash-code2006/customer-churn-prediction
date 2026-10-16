
import os
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Customer Retention Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# 2. CUSTOM DESIGN
# ==========================================================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f1f5f9, #eef2ff);
}
.block-container {
    padding-top: 1.5rem;
}
[data-testid="stMetric"] {
    background: white;
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #dbeafe;
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06);
}
div[data-testid="stMetricValue"] {
    color: #3730a3;
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# 3. LOAD DATA
# ==========================================================

DATA_FILE = "data/customer_churn.csv"

@st.cache_data
def load_data(file_path):
    if not os.path.exists(file_path):
        return None

    return pd.read_csv(file_path)

df = load_data(DATA_FILE)

if df is None:
    st.error(f"Dataset not found: {DATA_FILE}")
    st.info("Check that customer_churn.csv is inside the data folder.")
    st.stop()

required_columns = [
    "churn",
    "contract_type",
    "internet_service",
    "monthly_charges",
    "satisfaction_score",
    "tenure_months",
    "support_tickets_30d",
    "logins_30d",
    "usage_hours_30d"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    st.error("Missing required columns: " + ", ".join(missing_columns))
    st.stop()

if df.empty:
    st.warning("The customer dataset is empty.")
    st.stop()

# Prepare numeric columns safely
numeric_columns = [
    "churn",
    "monthly_charges",
    "satisfaction_score",
    "tenure_months",
    "support_tickets_30d",
    "logins_30d",
    "usage_hours_30d"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["churn"]).copy()
df["churn"] = df["churn"].astype(int)

# ==========================================================
# 4. DASHBOARD HEADING
# ==========================================================

st.title("📈 Customer Retention Intelligence")
st.markdown(
    "Explore customer behaviour, churn patterns, "
    "revenue exposure and retention opportunities."
)

st.caption("Business Analytics | Customer Churn Prediction")

# ==========================================================
# 5. SIDEBAR FILTERS
# ==========================================================

st.sidebar.title("🔎 Dashboard Filters")

contract_options = sorted(
    df["contract_type"].dropna().unique().tolist()
)

internet_options = sorted(
    df["internet_service"].dropna().unique().tolist()
)

selected_contracts = st.sidebar.multiselect(
    "Contract Type",
    options=contract_options,
    default=contract_options
)

selected_internet = st.sidebar.multiselect(
    "Internet Service",
    options=internet_options,
    default=internet_options
)

selected_status = st.sidebar.multiselect(
    "Customer Status",
    options=[0, 1],
    default=[0, 1],
    format_func=lambda value: (
        "Churned" if value == 1 else "Active"
    )
)

# Apply filters
filtered_df = df[
    df["contract_type"].isin(selected_contracts)
    & df["internet_service"].isin(selected_internet)
    & df["churn"].isin(selected_status)
].copy()

st.sidebar.divider()
st.sidebar.write(f"Filtered records: {len(filtered_df):,}")
st.sidebar.write(f"Original records: {len(df):,}")

if filtered_df.empty:
    st.warning("No customers match the selected filters.")
    st.stop()

# ==========================================================
# 6. CALCULATE BUSINESS METRICS
# ==========================================================

total_customers = len(filtered_df)

churned_df = filtered_df[filtered_df["churn"] == 1]
active_df = filtered_df[filtered_df["churn"] == 0]

churned_customers = len(churned_df)
active_customers = len(active_df)

churn_rate = (
    churned_customers / total_customers * 100
)

active_rate = (
    active_customers / total_customers * 100
)

avg_satisfaction = filtered_df[
    "satisfaction_score"
].mean()

avg_tenure = filtered_df["tenure_months"].mean()

avg_monthly_charges = filtered_df[
    "monthly_charges"
].mean()

total_monthly_revenue = filtered_df[
    "monthly_charges"
].sum()

revenue_at_risk = churned_df[
    "monthly_charges"
].sum()

# ==========================================================
# 7. TOP KPI CARDS
# ==========================================================

st.subheader("📌 Business Performance")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Total Customers",
    f"{total_customers:,}"
)

c2.metric(
    "Churn Rate",
    f"{churn_rate:.1f}%",
    delta=f"{churn_rate:.1f}% churn",
    delta_color="inverse"
)

c3.metric(
    "Active Customers",
    f"{active_customers:,}"
)

c4.metric(
    "Average Satisfaction",
    f"{avg_satisfaction:.2f}/10"
    if pd.notna(avg_satisfaction) else "N/A"
)

c5, c6, c7, c8 = st.columns(4)

c5.metric(
    "Average Tenure",
    f"{avg_tenure:.1f} months"
    if pd.notna(avg_tenure) else "N/A"
)

c6.metric(
    "Average Monthly Charges",
    f"${avg_monthly_charges:,.2f}"
    if pd.notna(avg_monthly_charges) else "N/A"
)

c7.metric(
    "Monthly Revenue",
    f"${total_monthly_revenue:,.2f}"
)

c8.metric(
    "Revenue from Churned Customers",
    f"${revenue_at_risk:,.2f}"
)

st.divider()

# ==========================================================
# 8. CUSTOMER STATUS VISUALIZATION
# ==========================================================

st.subheader("👥 Customer Retention Overview")

left, right = st.columns(2)

with left:
    status_data = pd.DataFrame({
        "Status": ["Active", "Churned"],
        "Customers": [active_customers, churned_customers]
    })

    fig = px.pie(
        status_data,
        names="Status",
        values="Customers",
        hole=0.62,
        color="Status",
        color_discrete_map={
            "Active": "#10b981",
            "Churned": "#f43f5e"
        },
        title="Customer Status Breakdown"
    )

    fig.update_traces(
        textinfo="label+percent",
        hovertemplate="%{label}<br>Customers: %{value}<extra></extra>"
    )

    st.plotly_chart(fig, use_container_width=True)

with right:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=churn_rate,
        number={"suffix": "%"},
        title={"text": "Current Churn Rate"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#f43f5e"},
            "steps": [
                {"range": [0, 20], "color": "#d1fae5"},
                {"range": [20, 40], "color": "#fef3c7"},
                {"range": [40, 100], "color": "#ffe4e6"}
            ]
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# 9. CONTRACT PERFORMANCE
# ==========================================================

st.subheader("📄 Contract Performance")

contract_data = (
    filtered_df.groupby("contract_type", dropna=False)
    .agg(
        customers=("churn", "size"),
        churned=("churn", "sum"),
        average_charges=("monthly_charges", "mean")
    )
    .reset_index()
)

contract_data["churn_rate"] = (
    contract_data["churned"]
    / contract_data["customers"].replace(0, float("nan"))
    * 100
)

left, right = st.columns(2)

with left:
    fig = px.bar(
        contract_data,
        x="contract_type",
        y="churn_rate",
        color="churn_rate",
        text="churn_rate",
        color_continuous_scale="Reds",
        title="Churn Rate by Contract",
        labels={
            "contract_type": "Contract",
            "churn_rate": "Churn Rate (%)"
        }
    )

    fig.update_traces(texttemplate="%{text:.1f}%")
    st.plotly_chart(fig, use_container_width=True)

with right:
    fig = px.bar(
        contract_data,
        x="contract_type",
        y="customers",
        color="contract_type",
        title="Customer Count by Contract",
        labels={
            "contract_type": "Contract",
            "customers": "Customers"
        }
    )

    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# 10. INTERNET SERVICE ANALYSIS
# ==========================================================

st.subheader("🌐 Internet Service Insights")

internet_data = (
    filtered_df.groupby("internet_service", dropna=False)
    .agg(
        customers=("churn", "size"),
        churned=("churn", "sum")
    )
    .reset_index()
)

internet_data["churn_rate"] = (
    internet_data["churned"]
    / internet_data["customers"].replace(0, float("nan"))
    * 100
)

fig = px.bar(
    internet_data,
    x="internet_service",
    y="churn_rate",
    text="churn_rate",
    color="internet_service",
    title="Churn Percentage by Internet Service",
    labels={
        "internet_service": "Internet Service",
        "churn_rate": "Churn Rate (%)"
    }
)

fig.update_traces(texttemplate="%{text:.1f}%")
st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# 11. NEW METHOD: REVENUE EXPOSURE
# ==========================================================

st.subheader("💰 Revenue Exposure Analysis")

revenue_data = pd.DataFrame({
    "Customer Group": ["Active Customers", "Churned Customers"],
    "Monthly Charges": [
        active_df["monthly_charges"].sum(),
        churned_df["monthly_charges"].sum()
    ]
})

left, right = st.columns([2, 1])

with left:
    fig = px.bar(
        revenue_data,
        x="Customer Group",
        y="Monthly Charges",
        color="Customer Group",
        text="Monthly Charges",
        color_discrete_map={
            "Active Customers": "#0ea5e9",
            "Churned Customers": "#f97316"
        },
        title="Monthly Charges by Customer Status"
    )

    fig.update_traces(
        texttemplate="$%{text:,.2f}",
        textposition="outside"
    )

    st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown("### Business Interpretation")

    if total_monthly_revenue > 0:
        exposure = revenue_at_risk / total_monthly_revenue * 100
    else:
        exposure = 0

    st.metric(
        "Share of Charges from Churned Customers",
        f"{exposure:.1f}%"
    )

    st.write(
        "This estimates the monthly charges associated with "
        "customers who have churned. It is not a forecast of "
        "future lost revenue."
    )

# ==========================================================
# 12. NEW METHOD: CUSTOMER RISK SEGMENTATION
# ==========================================================

st.subheader("🚦 Customer Risk Segmentation")

risk_df = filtered_df.copy()

# These are simple business rules, not ML predictions.
risk_df["risk_score"] = 0

risk_df.loc[
    risk_df["satisfaction_score"] <= 3,
    "risk_score"
] += 2

risk_df.loc[
    risk_df["support_tickets_30d"] >= 3,
    "risk_score"
] += 2

risk_df.loc[
    risk_df["tenure_months"] <= 6,
    "risk_score"
] += 1

risk_df.loc[
    risk_df["logins_30d"] <= 2,
    "risk_score"
] += 1

risk_df["risk_segment"] = pd.cut(
    risk_df["risk_score"],
    bins=[-1, 1, 3, 6],
    labels=["Low", "Medium", "High"]
)

risk_summary = (
    risk_df.groupby("risk_segment", observed=False)
    .agg(
        customers=("churn", "size"),
        actual_churn_rate=("churn", "mean")
    )
    .reset_index()
)

risk_summary["actual_churn_rate"] *= 100

left, right = st.columns(2)

with left:
    fig = px.bar(
        risk_summary,
        x="risk_segment",
        y="customers",
        color="risk_segment",
        text="customers",
        title="Customers by Rule-Based Risk Segment",
        labels={
            "risk_segment": "Risk Segment",
            "customers": "Customers"
        }
    )

    st.plotly_chart(fig, use_container_width=True)

with right:
    fig = px.bar(
        risk_summary,
        x="risk_segment",
        y="actual_churn_rate",
        color="risk_segment",
        text="actual_churn_rate",
        title="Observed Churn by Risk Segment",
        labels={
            "risk_segment": "Risk Segment",
            "actual_churn_rate": "Observed Churn Rate (%)"
        }
    )

    fig.update_traces(texttemplate="%{text:.1f}%")
    st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Risk segments are heuristic categories based on satisfaction, "
    "support tickets, tenure and logins. They are not generated "
    "by the trained machine-learning model."
)

# ==========================================================
# 13. NEW METHOD: ENGAGEMENT ANALYSIS
# ==========================================================

st.subheader("📱 Customer Engagement")

fig = px.scatter(
    filtered_df,
    x="logins_30d",
    y="usage_hours_30d",
    color=filtered_df["churn"].map({
        0: "Active",
        1: "Churned"
    }),
    size="monthly_charges",
    hover_data=[
        "satisfaction_score",
        "tenure_months",
        "support_tickets_30d"
    ],
    opacity=0.7,
    title="Login Frequency vs Usage Hours",
    labels={
        "logins_30d": "Logins in Last 30 Days",
        "usage_hours_30d": "Usage Hours in Last 30 Days",
        "monthly_charges": "Monthly Charges",
        "color": "Customer Status"
    }
)

st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# 14. SATISFACTION ANALYSIS
# ==========================================================

st.subheader("😊 Satisfaction and Churn")

satisfaction_data = (
    filtered_df.groupby("satisfaction_score", dropna=True)
    .agg(
        customers=("churn", "size"),
        churn_rate=("churn", "mean")
    )
    .reset_index()
)

satisfaction_data["churn_rate"] *= 100

fig = px.line(
    satisfaction_data,
    x="satisfaction_score",
    y="churn_rate",
    markers=True,
    title="Observed Churn Rate by Satisfaction Score",
    labels={
        "satisfaction_score": "Satisfaction Score",
        "churn_rate": "Churn Rate (%)"
    }
)

st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# 15. TENURE ANALYSIS
# ==========================================================

st.subheader("⏳ Customer Tenure")

tenure_df = filtered_df.dropna(
    subset=["tenure_months"]
).copy()

tenure_df["tenure_group"] = pd.cut(
    tenure_df["tenure_months"],
    bins=[-1, 6, 12, 24, 36, 60, float("inf")],
    labels=[
        "0-6 Months",
        "7-12 Months",
        "13-24 Months",
        "25-36 Months",
        "37-60 Months",
        "60+ Months"
    ]
)

tenure_data = (
    tenure_df.groupby("tenure_group", observed=False)
    .agg(
        customers=("churn", "size"),
        churn_rate=("churn", "mean")
    )
    .reset_index()
)

tenure_data["churn_rate"] *= 100

fig = px.bar(
    tenure_data,
    x="tenure_group",
    y="churn_rate",
    text="churn_rate",
    color="churn_rate",
    color_continuous_scale="Oranges",
    title="Churn Rate Across Tenure Groups",
    labels={
        "tenure_group": "Tenure Group",
        "churn_rate": "Churn Rate (%)"
    }
)

fig.update_traces(texttemplate="%{text:.1f}%")
st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# 16. SUPPORT TICKET ANALYSIS
# ==========================================================

st.subheader("🎧 Customer Support Analysis")

support_data = (
    filtered_df.groupby("support_tickets_30d", dropna=True)
    .agg(
        customers=("churn", "size"),
        churn_rate=("churn", "mean")
    )
    .reset_index()
)

support_data["churn_rate"] *= 100

fig = px.bar(
    support_data,
    x="support_tickets_30d",
    y="churn_rate",
    text="churn_rate",
    title="Observed Churn by Recent Support Tickets",
    labels={
        "support_tickets_30d": "Support Tickets in 30 Days",
        "churn_rate": "Churn Rate (%)"
    }
)

st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# 17. AUTOMATIC BUSINESS SUMMARY
# ==========================================================

st.subheader("💡 Business Recommendations")

if not contract_data.empty:
    highest_contract = contract_data.sort_values(
        "churn_rate", ascending=False
    ).iloc[0]

    st.write(
        f"- **Contract review:** "
        f"{highest_contract['contract_type']} has the highest "
        f"observed contract churn rate "
        f"({highest_contract['churn_rate']:.1f}%) in this filtered data."
    )

if not internet_data.empty:
    highest_internet = internet_data.sort_values(
        "churn_rate", ascending=False
    ).iloc[0]

    st.write(
        f"- **Service review:** "
        f"{highest_internet['internet_service']} has the highest "
        f"observed internet-service churn rate "
        f"({highest_internet['churn_rate']:.1f}%)."
    )

if pd.notna(avg_satisfaction) and avg_satisfaction < 6:
    st.write(
        "- **Satisfaction:** Review customer feedback and improve "
        "support for customers reporting low satisfaction."
    )
else:
    st.write(
        "- **Satisfaction:** Continue monitoring satisfaction scores "
        "and compare them with actual churn."
    )

st.write(
    "- **Retention:** Use the trained ML model to identify customers "
    "with high predicted churn probability before selecting retention actions."
)

st.write(
    "- **Revenue:** Evaluate retention offers using customer value, "
    "predicted churn probability and campaign cost."
)

# ==========================================================
# 18. CUSTOMER DATA TABLE
# ==========================================================

st.subheader("📋 Filtered Customer Records")

with st.expander("Open Customer Data"):
    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )

# ==========================================================
# 19. DOWNLOAD FILTERED DATA
# ==========================================================

csv_file = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Download Filtered Customer Data",
    data=csv_file,
    file_name="filtered_customer_churn.csv",
    mime="text/csv"
)

# ==========================================================
# 20. FOOTER
# ==========================================================

st.divider()

st.caption(
    "Customer Retention Intelligence Dashboard | "
    "Customer Churn Prediction Project"
)

st.caption(
    "All rates describe the selected dataset. "
    "Associations do not establish causation."
)
