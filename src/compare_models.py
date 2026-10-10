import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("data/customer_churn.csv")

print("Dataset loaded successfully!")
print("Shape:", df.shape)


# ============================================================
# 2. FEATURES AND TARGET
# ============================================================

X = df.drop(columns=["customer_id", "churn"])
y = df["churn"]

print("\nFeatures:", X.shape)
print("Target:", y.shape)


# ============================================================
# 3. COLUMN DEFINITIONS
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
# 4. TRAIN / TEST SPLIT
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
# 5. PREPROCESSING FOR LOGISTIC REGRESSION
# ============================================================

logistic_preprocessor = ColumnTransformer(
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


# ============================================================
# 6. PREPROCESSING FOR TREE MODELS
# ============================================================

tree_preprocessor = ColumnTransformer(
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
# 7. TRANSFORM DATA
# ============================================================

X_train_logistic = logistic_preprocessor.fit_transform(X_train)
X_test_logistic = logistic_preprocessor.transform(X_test)

X_train_tree = tree_preprocessor.fit_transform(X_train)
X_test_tree = tree_preprocessor.transform(X_test)

print("\nPreprocessing completed!")
print("Logistic Regression shape:", X_train_logistic.shape)
print("Tree model shape:", X_train_tree.shape)


# ============================================================
# 8. TRAIN LOGISTIC REGRESSION
# ============================================================

print("\nTraining Logistic Regression...")

logistic_model = LogisticRegression(
    max_iter=2000,
    random_state=42
)

logistic_model.fit(
    X_train_logistic,
    y_train
)

print("Logistic Regression completed!")


# ============================================================
# 9. TRAIN RANDOM FOREST
# ============================================================

print("\nTraining Random Forest...")

random_forest_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

random_forest_model.fit(
    X_train_tree,
    y_train
)

print("Random Forest completed!")


# ============================================================
# 10. TRAIN XGBOOST
# ============================================================

print("\nTraining XGBoost...")

negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

scale_pos_weight = negative_count / positive_count

xgboost_model = XGBClassifier(
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

xgboost_model.fit(
    X_train_tree,
    y_train
)

print("XGBoost completed!")


# ============================================================
# 11. PREDICTIONS
# ============================================================

logistic_pred = logistic_model.predict(X_test_logistic)
logistic_prob = logistic_model.predict_proba(
    X_test_logistic
)[:, 1]


random_forest_pred = random_forest_model.predict(X_test_tree)
random_forest_prob = random_forest_model.predict_proba(
    X_test_tree
)[:, 1]


xgboost_pred = xgboost_model.predict(X_test_tree)
xgboost_prob = xgboost_model.predict_proba(
    X_test_tree
)[:, 1]


# ============================================================
# 12. FUNCTION TO CALCULATE METRICS
# ============================================================

def calculate_metrics(model_name, y_true, predictions, probabilities):

    return {
        "Model": model_name,

        "Accuracy": accuracy_score(
            y_true,
            predictions
        ),

        "Precision": precision_score(
            y_true,
            predictions
        ),

        "Recall": recall_score(
            y_true,
            predictions
        ),

        "F1 Score": f1_score(
            y_true,
            predictions
        ),

        "ROC-AUC": roc_auc_score(
            y_true,
            probabilities
        ),

        "PR-AUC": average_precision_score(
            y_true,
            probabilities
        )
    }


# ============================================================
# 13. CALCULATE RESULTS
# ============================================================

results = []

results.append(
    calculate_metrics(
        "Logistic Regression",
        y_test,
        logistic_pred,
        logistic_prob
    )
)

results.append(
    calculate_metrics(
        "Random Forest",
        y_test,
        random_forest_pred,
        random_forest_prob
    )
)

results.append(
    calculate_metrics(
        "XGBoost",
        y_test,
        xgboost_pred,
        xgboost_prob
    )
)


# ============================================================
# 14. CREATE COMPARISON TABLE
# ============================================================

results_df = pd.DataFrame(results)

print("\n")
print("=" * 90)
print("MODEL COMPARISON")
print("=" * 90)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 15. FIND BEST MODELS
# ============================================================

best_recall = results_df.loc[
    results_df["Recall"].idxmax()
]

best_roc_auc = results_df.loc[
    results_df["ROC-AUC"].idxmax()
]

best_pr_auc = results_df.loc[
    results_df["PR-AUC"].idxmax()
]

best_f1 = results_df.loc[
    results_df["F1 Score"].idxmax()
]


print("\n")
print("=" * 90)
print("BEST MODEL BY METRIC")
print("=" * 90)

print(
    f"Best Recall : {best_recall['Model']} "
    f"({best_recall['Recall']:.4f})"
)

print(
    f"Best ROC-AUC: {best_roc_auc['Model']} "
    f"({best_roc_auc['ROC-AUC']:.4f})"
)

print(
    f"Best PR-AUC : {best_pr_auc['Model']} "
    f"({best_pr_auc['PR-AUC']:.4f})"
)

print(
    f"Best F1     : {best_f1['Model']} "
    f"({best_f1['F1 Score']:.4f})"
)


# ============================================================
# 16. SAVE COMPARISON TABLE
# ============================================================

results_df.to_csv(
    "models/model_comparison.csv",
    index=False
)

print("\n")
print("=" * 90)
print("MODEL COMPARISON SAVED!")
print("=" * 90)

print("Saved file:")
print("models/model_comparison.csv")