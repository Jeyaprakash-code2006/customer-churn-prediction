import os
import pandas as pd
import streamlit as st
import plotly.express as px


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn - Business Dashboard",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f8fafc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.dashboard-title {
    font-size: 36px;
    font-weight: 700;
}

.dashboard-subtitle {
    font-size: 17px;
    color: #64748b;
    margin-bottom: 25px;
}

[data-testid="stMetric"] {
    background-color: white;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 18px;
    box-shadow: 0px 3px 10px rgba(0,0,0,0.05);
}

.insight-box {
    background-color: white;
    padding: 22px;
    border-radius: 14px;
    border: 1px solid #e2e8f0;
    margin-bottom: 15px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA FILE
# ============================================================

DATA_FILE = "data/customer_churn.csv"


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    if not os.path.exists(DATA_FILE):
        return None

    return pd.read_csv(DATA_FILE)


df = load_data()


# ============================================================
# CHECK DATA
# ============================================================

if df is None:

    st.error(
        "❌ customer_churn.csv was not found."
    )

    st.info(
        "Make sure this file exists:\n\n"
        "data/customer_churn.csv"
    )

    st.stop()


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="dashboard-title">'
    '📊 Customer Churn Business Dashboard'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="dashboard-subtitle">'
    'Executive overview of customer churn, customer behavior, '
    'revenue exposure, and retention opportunities.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# BASIC CALCULATIONS
# ============================================================

total_customers = len(df)

churned_customers = int(
    df["churn"].sum()
)

active_customers = (
    total_customers -
    churned_customers
)

churn_rate = (
    churned_customers /
    total_customers
) * 100

active_rate = (
    active_customers /
    total_customers
) * 100

avg_satisfaction = (
    df["satisfaction_score"].mean()
)

avg_monthly_charges = (
    df["monthly_charges"].mean()
)

total_monthly_revenue = (
    df["monthly_charges"].sum()
)

churned_monthly_revenue = (
    df.loc[
        df["churn"] == 1,
        "monthly_charges"
    ].sum()
)

avg_tenure = (
    df["tenure_months"].mean()
)


# ============================================================
# KPI SECTION
# ============================================================

st.subheader("📌 Business Overview")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:

    st.metric(
        "Total Customers",
        f"{total_customers:,}"
    )

with col2:

    st.metric(
        "Churned Customers",
        f"{churned_customers:,}",
        delta=f"{churn_rate:.2f}% churn"
    )

with col3:

    st.metric(
        "Active Customers",
        f"{active_customers:,}",
        delta=f"{active_rate:.2f}% active"
    )

with col4:

    st.metric(
        "Avg Satisfaction",
        f"{avg_satisfaction:.2f}/10"
    )

with col5:

    st.metric(
        "Avg Tenure",
        f"{avg_tenure:.1f} months"
    )


# ============================================================
# REVENUE KPI
# ============================================================

st.divider()

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Average Monthly Charges",
        f"${avg_monthly_charges:,.2f}"
    )

with col2:

    st.metric(
        "Total Monthly Revenue",
        f"${total_monthly_revenue:,.2f}"
    )

with col3:

    st.metric(
        "Revenue from Churned Customers",
        f"${churned_monthly_revenue:,.2f}"
    )


# ============================================================
# CHURN DISTRIBUTION
# ============================================================

st.divider()

st.subheader("📊 Customer Churn Distribution")

col1, col2 = st.columns(2)


# ------------------------------------------------------------
# PIE CHART
# ------------------------------------------------------------

with col1:

    churn_distribution = pd.DataFrame({
        "Customer Status": [
            "Active",
            "Churned"
        ],
        "Customers": [
            active_customers,
            churned_customers
        ]
    })

    fig = px.pie(
        churn_distribution,
        names="Customer Status",
        values="Customers",
        hole=0.5,
        title="Active vs Churned Customers"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ------------------------------------------------------------
# CHURN RATE
# ------------------------------------------------------------

with col2:

    churn_summary = pd.DataFrame({
        "Status": [
            "Active",
            "Churned"
        ],
        "Percentage": [
            active_rate,
            churn_rate
        ]
    })

    fig = px.bar(
        churn_summary,
        x="Status",
        y="Percentage",
        text="Percentage",
        title="Customer Percentage"
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    fig.update_layout(
        yaxis_title="Percentage (%)",
        xaxis_title=""
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# CONTRACT ANALYSIS
# ============================================================

st.divider()

st.subheader("📄 Churn by Contract Type")

contract_analysis = (
    df.groupby(
        "contract_type"
    )
    .agg(
        customers=("churn", "count"),
        churned=("churn", "sum"),
        churn_rate=("churn", "mean")
    )
    .reset_index()
)

contract_analysis[
    "churn_rate"
] *= 100

col1, col2 = st.columns(2)


with col1:

    fig = px.bar(
        contract_analysis,
        x="contract_type",
        y="churn_rate",
        text="churn_rate",
        title="Churn Rate by Contract Type"
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    fig.update_layout(
        xaxis_title="Contract Type",
        yaxis_title="Churn Rate (%)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


with col2:

    fig = px.bar(
        contract_analysis,
        x="contract_type",
        y="customers",
        text="customers",
        title="Customers by Contract Type"
    )

    fig.update_traces(
        textposition="outside"
    )

    fig.update_layout(
        xaxis_title="Contract Type",
        yaxis_title="Customers"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# INTERNET SERVICE ANALYSIS
# ============================================================

st.divider()

st.subheader("🌐 Churn by Internet Service")

internet_analysis = (
    df.groupby(
        "internet_service"
    )
    .agg(
        customers=("churn", "count"),
        churned=("churn", "sum"),
        churn_rate=("churn", "mean")
    )
    .reset_index()
)

internet_analysis[
    "churn_rate"
] *= 100

fig = px.bar(
    internet_analysis,
    x="internet_service",
    y="churn_rate",
    text="churn_rate",
    title="Churn Rate by Internet Service"
)

fig.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)

fig.update_layout(
    xaxis_title="Internet Service",
    yaxis_title="Churn Rate (%)"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# MONTHLY CHARGES VS CHURN
# ============================================================

st.divider()

st.subheader("💳 Monthly Charges vs Churn")

fig = px.box(
    df,
    x="churn",
    y="monthly_charges",
    points="outliers",
    title="Monthly Charges Distribution by Churn",
    labels={
        "churn": "Churn Status",
        "monthly_charges": "Monthly Charges"
    }
)

fig.update_xaxes(
    tickvals=[0, 1],
    ticktext=[
        "Active",
        "Churned"
    ]
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# SATISFACTION ANALYSIS
# ============================================================

st.divider()

st.subheader("😊 Satisfaction Score vs Churn")

satisfaction_analysis = (
    df.groupby(
        "satisfaction_score"
    )
    .agg(
        customers=("churn", "count"),
        churn_rate=("churn", "mean")
    )
    .reset_index()
)

satisfaction_analysis[
    "churn_rate"
] *= 100

fig = px.line(
    satisfaction_analysis,
    x="satisfaction_score",
    y="churn_rate",
    markers=True,
    title="Churn Rate by Satisfaction Score"
)

fig.update_layout(
    xaxis_title="Satisfaction Score",
    yaxis_title="Churn Rate (%)"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# TENURE ANALYSIS
# ============================================================

st.divider()

st.subheader("⏳ Tenure vs Churn")

tenure_bins = [
    0,
    6,
    12,
    24,
    36,
    60,
    float("inf")
]

tenure_labels = [
    "0-6 Months",
    "7-12 Months",
    "13-24 Months",
    "25-36 Months",
    "37-60 Months",
    "60+ Months"
]

tenure_df = df.copy()

tenure_df["tenure_group"] = pd.cut(
    tenure_df["tenure_months"],
    bins=tenure_bins,
    labels=tenure_labels,
    include_lowest=True
)

tenure_analysis = (
    tenure_df.groupby(
        "tenure_group",
        observed=False
    )["churn"]
    .mean()
    .reset_index()
)

tenure_analysis[
    "churn"
] *= 100

fig = px.bar(
    tenure_analysis,
    x="tenure_group",
    y="churn",
    text="churn",
    title="Churn Rate by Customer Tenure"
)

fig.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)

fig.update_layout(
    xaxis_title="Tenure Group",
    yaxis_title="Churn Rate (%)"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# SUPPORT TICKETS
# ============================================================

st.divider()

st.subheader("🎧 Support Tickets vs Churn")

support_analysis = (
    df.groupby(
        "support_tickets_30d"
    )["churn"]
    .mean()
    .reset_index()
)

support_analysis[
    "churn"
] *= 100

fig = px.line(
    support_analysis,
    x="support_tickets_30d",
    y="churn",
    markers=True,
    title="Churn Rate by Support Tickets"
)

fig.update_layout(
    xaxis_title="Support Tickets in Last 30 Days",
    yaxis_title="Churn Rate (%)"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# LOGIN ACTIVITY
# ============================================================

st.divider()

st.subheader("🔐 Login Activity vs Churn")

fig = px.scatter(
    df,
    x="logins_30d",
    y="usage_hours_30d",
    color="churn",
    opacity=0.65,
    title="Customer Usage and Login Activity",
    labels={
        "logins_30d": "Logins in Last 30 Days",
        "usage_hours_30d": "Usage Hours in Last 30 Days",
        "churn": "Churn"
    }
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# BUSINESS INSIGHTS
# ============================================================

st.divider()

st.subheader("💡 Key Business Insights")


# Highest contract churn

highest_contract_row = (
    contract_analysis
    .sort_values(
        "churn_rate",
        ascending=False
    )
    .iloc[0]
)

highest_contract = (
    highest_contract_row[
        "contract_type"
    ]
)

highest_contract_rate = (
    highest_contract_row[
        "churn_rate"
    ]
)


# Highest internet churn

highest_internet_row = (
    internet_analysis
    .sort_values(
        "churn_rate",
        ascending=False
    )
    .iloc[0]
)

highest_internet = (
    highest_internet_row[
        "internet_service"
    ]
)

highest_internet_rate = (
    highest_internet_row[
        "churn_rate"
    ]
)


# Highest satisfaction-related churn

highest_satisfaction_row = (
    satisfaction_analysis
    .sort_values(
        "churn_rate",
        ascending=False
    )
    .iloc[0]
)

highest_satisfaction_churn_score = (
    highest_satisfaction_row[
        "satisfaction_score"
    ]
)

highest_satisfaction_churn_rate = (
    highest_satisfaction_row[
        "churn_rate"
    ]
)


# Highest tenure churn

highest_tenure_row = (
    tenure_analysis
    .sort_values(
        "churn",
        ascending=False
    )
    .iloc[0]
)

highest_tenure_group = (
    highest_tenure_row[
        "tenure_group"
    ]
)

highest_tenure_rate = (
    highest_tenure_row[
        "churn"
    ]
)


col1, col2 = st.columns(2)


with col1:

    st.markdown(
        f"""
        <div class="insight-box">

        ### 🔎 Major Findings

        **Overall churn:**  
        {churn_rate:.2f}%

        **Highest-churn contract:**  
        {highest_contract} — {highest_contract_rate:.2f}%

        **Highest-churn internet service:**  
        {highest_internet} — {highest_internet_rate:.2f}%

        **Highest observed churn satisfaction group:**  
        Score {highest_satisfaction_churn_score} —
        {highest_satisfaction_churn_rate:.2f}%

        **Highest-churn tenure group:**  
        {highest_tenure_group} —
        {highest_tenure_rate:.2f}%

        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        """
        <div class="insight-box">

        ### 🎯 Recommended Business Actions

        **1. Identify high-risk customers**

        Use the ML model to identify customers with
        high predicted churn probability.

        **2. Prioritize valuable customers**

        Combine churn probability with estimated
        customer value.

        **3. Improve satisfaction**

        Investigate customers with low satisfaction
        and provide targeted support.

        **4. Monitor customer engagement**

        Low login and usage activity can be used
        as an additional warning signal.

        **5. Reduce revenue exposure**

        Focus retention resources on customers
        with higher expected revenue at risk.

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# DATA TABLE
# ============================================================

st.divider()

st.subheader("📋 Customer Dataset")

with st.expander(
    "View Customer Data"
):

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# DOWNLOAD DATA
# ============================================================

csv_data = (
    df
    .to_csv(index=False)
    .encode("utf-8")
)

st.download_button(
    label="⬇️ Download Customer Dataset",
    data=csv_data,
    file_name="customer_churn.csv",
    mime="text/csv"
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Customer Churn Prediction • Business Analytics Dashboard"
)

st.caption(
    "Note: Findings are based on the project dataset and "
    "should be validated before production business decisions."
)