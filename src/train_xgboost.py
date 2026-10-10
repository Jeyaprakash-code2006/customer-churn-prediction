import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

from xgboost import XGBClassifier


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("data/customer_churn.csv")

print("Dataset loaded successfully!")
print("Shape:", df.shape)


# ============================================================
# 2. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(columns=["customer_id", "churn"])
y = df["churn"]

print("\nFeatures and target separated!")
print("Features:", X.shape)
print("Target  :", y.shape)


# ============================================================
# 3. DEFINE COLUMNS
# ============================================================

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


# ============================================================
# 4. CREATE PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        ),
        (
            "numerical",
            "passthrough",
            numerical_columns
        )
    ]
)


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

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


# ============================================================
# 6. PREPROCESS DATA
# ============================================================

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("\nPreprocessing completed!")
print("Processed training shape:", X_train_processed.shape)
print("Processed testing shape :", X_test_processed.shape)


# ============================================================
# 7. CALCULATE CLASS IMBALANCE
# ============================================================

negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

scale_pos_weight = negative_count / positive_count

print("\nClass distribution in training data:")
print("Not Churn:", negative_count)
print("Churn    :", positive_count)
print("scale_pos_weight:", round(scale_pos_weight, 4))


# ============================================================
# 8. CREATE XGBOOST MODEL
# ============================================================

model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 9. TRAIN MODEL
# ============================================================

print("\nTraining XGBoost model...")

model.fit(
    X_train_processed,
    y_train
)

print("XGBoost training completed!")


# ============================================================
# 10. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test_processed)

y_probability = model.predict_proba(
    X_test_processed
)[:, 1]


# ============================================================
# 11. CALCULATE METRICS
# ============================================================

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


# ============================================================
# 12. DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 50)
print("XGBOOST RESULTS")
print("=" * 50)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")


# ============================================================
# 13. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")
print("-" * 50)

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ============================================================
# 14. SAVE MODEL
# ============================================================

joblib.dump(
    model,
    "models/xgboost.pkl"
)

joblib.dump(
    preprocessor,
    "models/xgboost_preprocessor.pkl"
)


# ============================================================
# 15. FINAL MESSAGE
# ============================================================

print("=" * 50)
print("XGBOOST MODEL SAVED SUCCESSFULLY!")
print("=" * 50)

print("\nSaved files:")
print("1. models/xgboost.pkl")
print("2. models/xgboost_preprocessor.pkl")