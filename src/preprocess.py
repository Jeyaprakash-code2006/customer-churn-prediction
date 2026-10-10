import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

DATA_PATH = "data/customer_churn.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully")
print("Shape:", df.shape)


# --------------------------------------------------
# 2. Check missing values
# --------------------------------------------------

print("\nMissing values:")
print(df.isnull().sum())


# --------------------------------------------------
# 3. Separate features and target
# --------------------------------------------------

X = df.drop(columns=["customer_id", "churn"])
y = df["churn"]


# --------------------------------------------------
# 4. Identify categorical and numerical columns
# --------------------------------------------------

categorical_columns = [
    "gender",
    "contract_type",
    "internet_service",
    "payment_method"
]

numerical_columns = [
    "age",
    "senior_citizen",
    "tenure_months",
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


# --------------------------------------------------
# 5. Create preprocessing pipeline
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical_columns
        ),
        (
            "numerical",
            "passthrough",
            numerical_columns
        )
    ]
)


# --------------------------------------------------
# 6. Split data
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 7. Fit preprocessing on training data
# --------------------------------------------------

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)


# --------------------------------------------------
# 8. Save preprocessing object
# --------------------------------------------------

joblib.dump(
    preprocessor,
    "models/preprocessor.pkl"
)


# --------------------------------------------------
# 9. Display results
# --------------------------------------------------

print("\nPreprocessing completed successfully!")

print("\nTraining data:")
print("X_train:", X_train.shape)
print("Processed X_train:", X_train_processed.shape)
print("y_train:", y_train.shape)

print("\nTesting data:")
print("X_test:", X_test.shape)
print("Processed X_test:", X_test_processed.shape)
print("y_test:", y_test.shape)

print("\nPreprocessor saved to:")
print("models/preprocessor.pkl")