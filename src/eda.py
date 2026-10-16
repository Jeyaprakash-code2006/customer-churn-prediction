
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -----------------------------------
# Load Dataset
# -----------------------------------

df = pd.read_csv("data/customer_churn.csv")

print("Dataset loaded successfully!")
print("Dataset Shape:", df.shape)

# Create folder to save charts
os.makedirs("data", exist_ok=True)

# Set chart design
sns.set_theme(style="darkgrid", palette="viridis")

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicate records
print("\nDuplicate Records:", df.duplicated().sum())

# -----------------------------------
# 1. Churn Percentage
# -----------------------------------

churn_counts = df["churn"].value_counts().sort_index()

plt.figure(figsize=(7, 5))
plt.pie(
    churn_counts,
    labels=["Active Customers", "Churned Customers"],
    autopct="%1.1f%%",
    startangle=90,
    explode=(0, 0.1),
    shadow=True
)
plt.title("Customer Churn Percentage")
plt.tight_layout()
plt.savefig("data/churn_percentage.png", dpi=300)
plt.show()

# -----------------------------------
# 2. Churn by Contract Type
# -----------------------------------

contract_data = pd.crosstab(
    df["contract_type"],
    df["churn"],
    normalize="index"
) * 100

contract_data.plot(
    kind="bar",
    figsize=(9, 5),
    colormap="viridis"
)

plt.title("Churn Rate by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Percentage of Customers")
plt.xticks(rotation=20)
plt.legend(["Active", "Churned"])
plt.tight_layout()
plt.savefig("data/contract_churn_rate.png", dpi=300)
plt.show()

# -----------------------------------
# 3. Monthly Charges Distribution
# -----------------------------------

plt.figure(figsize=(8, 5))
sns.histplot(
    data=df,
    x="monthly_charges",
    hue="churn",
    kde=True,
    bins=25,
    multiple="layer"
)

plt.title("Monthly Charges Distribution by Churn")
plt.xlabel("Monthly Charges")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("data/monthly_charges_distribution.png", dpi=300)
plt.show()

# -----------------------------------
# 4. Customer Satisfaction Analysis
# -----------------------------------

plt.figure(figsize=(8, 5))
sns.violinplot(
    data=df,
    x="churn",
    y="satisfaction_score",
    inner="quartile"
)

plt.title("Customer Satisfaction by Churn Status")
plt.xlabel("Churn (0 = Active, 1 = Churned)")
plt.ylabel("Satisfaction Score")
plt.tight_layout()
plt.savefig("data/satisfaction_violin.png", dpi=300)
plt.show()

# -----------------------------------
# 5. Churn by Internet Service
# -----------------------------------

internet_data = pd.crosstab(
    df["internet_service"],
    df["churn"],
    normalize="index"
) * 100

internet_data.plot(
    kind="barh",
    figsize=(9, 5),
    colormap="plasma"
)

plt.title("Churn Percentage by Internet Service")
plt.xlabel("Percentage of Customers")
plt.ylabel("Internet Service")
plt.legend(["Active", "Churned"])
plt.tight_layout()
plt.savefig("data/internet_churn_percentage.png", dpi=300)
plt.show()

# -----------------------------------
# 6. Correlation Heatmap
# -----------------------------------

numeric_df = df.select_dtypes(include="number")

if numeric_df.shape[1] > 1:
    plt.figure(figsize=(10, 7))

    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5
    )

    plt.title("Customer Data Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("data/correlation_heatmap.png", dpi=300)
    plt.show()
else:
    print("Not enough numerical columns for correlation analysis.")

# -----------------------------------
# 7. Average Customer Charges
# -----------------------------------

if "churn" in df.columns and "monthly_charges" in df.columns:
    avg_charges = df.groupby("churn")["monthly_charges"].mean()

    print("\nAverage Monthly Charges by Churn Status:")
    print(avg_charges)

# -----------------------------------
# Summary
# -----------------------------------

print("\nEDA completed successfully!")

print("\nGenerated Charts:")
print("1. churn_percentage.png")
print("2. contract_churn_rate.png")
print("3. monthly_charges_distribution.png")
print("4. satisfaction_violin.png")
print("5. internet_churn_percentage.png")
print("6. correlation_heatmap.png")

print("\nDataset analysis completed.")
