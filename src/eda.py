import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load dataset
df = pd.read_csv("data/customer_churn.csv")

print("Dataset loaded successfully!")
print("Shape:", df.shape)


# -------------------------------
# 1. Churn Distribution
# -------------------------------

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="churn"
)

plt.title("Customer Churn Distribution")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.savefig("data/churn_distribution.png")
plt.show()


# -------------------------------
# 2. Churn by Contract Type
# -------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="contract_type",
    hue="churn"
)

plt.title("Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.savefig("data/churn_by_contract.png")
plt.show()


# -------------------------------
# 3. Churn by Internet Service
# -------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="internet_service",
    hue="churn"
)

plt.title("Churn by Internet Service")
plt.xlabel("Internet Service")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.savefig("data/churn_by_internet.png")
plt.show()


# -------------------------------
# 4. Monthly Charges vs Churn
# -------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="churn",
    y="monthly_charges"
)

plt.title("Monthly Charges vs Churn")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Monthly Charges")

plt.tight_layout()
plt.savefig("data/monthly_charges_vs_churn.png")
plt.show()


# -------------------------------
# 5. Satisfaction Score vs Churn
# -------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="churn",
    y="satisfaction_score"
)

plt.title("Satisfaction Score vs Churn")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Satisfaction Score")

plt.tight_layout()
plt.savefig("data/satisfaction_vs_churn.png")
plt.show()


print("\nEDA completed successfully!")

print("\nGenerated charts:")
print("1. churn_distribution.png")
print("2. churn_by_contract.png")
print("3. churn_by_internet.png")
print("4. monthly_charges_vs_churn.png")
print("5. satisfaction_vs_churn.png")