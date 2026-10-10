import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)


# ==================================================
# 1. LOAD DATASET
# ==================================================

df = pd.read_csv("data/customer_churn.csv")

print("Dataset loaded successfully!")
print("Shape:", df.shape)


# ==================================================
# 2. SEPARATE FEATURES AND TARGET
# ==================================================

# customer_id is removed because it is only an identifier.
# churn is our target variable.

X = df.drop(columns=["customer_id", "churn"])
y = df["churn"]


# ==================================================
# 3. DEFINE COLUMNS
# ==================================================

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


# ==================================================
# 4. CREATE PREPROCESSING PIPELINE
# ==================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        ),
        (
            "numerical",
            StandardScaler(),
            numerical_columns
        )
    ]
)


# ==================================================
# 5. SPLIT DATA INTO TRAINING AND TESTING
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nData split completed!")

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# ==================================================
# 6. PREPROCESS THE DATA
# ==================================================

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)


print("\nPreprocessing completed!")

print("Processed training shape:", X_train_processed.shape)
print("Processed testing shape :", X_test_processed.shape)


# ==================================================
# 7. CREATE LOGISTIC REGRESSION MODEL
# ==================================================

model = LogisticRegression(
    max_iter=2000,
    random_state=42
)


# ==================================================
# 8. TRAIN MODEL
# ==================================================

print("\nTraining Logistic Regression model...")

model.fit(
    X_train_processed,
    y_train
)

print("Model training completed!")


# ==================================================
# 9. MAKE PREDICTIONS
# ==================================================

y_pred = model.predict(X_test_processed)

y_probability = model.predict_proba(
    X_test_processed
)[:, 1]


# ==================================================
# 10. CALCULATE EVALUATION METRICS
# ==================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


# ==================================================
# 11. DISPLAY RESULTS
# ==================================================

print("\n")
print("=" * 50)
print("LOGISTIC REGRESSION RESULTS")
print("=" * 50)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")


# ==================================================
# 12. CLASSIFICATION REPORT
# ==================================================

print("\nClassification Report:")
print("-" * 50)

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ==================================================
# 13. SAVE MODEL
# ==================================================

joblib.dump(
    model,
    "models/logistic_regression.pkl"
)

joblib.dump(
    preprocessor,
    "models/model_preprocessor.pkl"
)


# ==================================================
# 14. FINAL MESSAGE
# ==================================================

print("=" * 50)
print("MODEL SAVED SUCCESSFULLY!")
print("=" * 50)

print("\nSaved files:")

print("1. models/logistic_regression.pkl")
print("2. models/model_preprocessor.pkl")