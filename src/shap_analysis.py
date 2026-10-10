import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


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


# ============================================================
# 3. LOAD TRAINED LOGISTIC REGRESSION MODEL
# ============================================================

model = joblib.load(
    "models/logistic_regression.pkl"
)

preprocessor = joblib.load(
    "models/model_preprocessor.pkl"
)

print("Logistic Regression model loaded!")
print("Preprocessor loaded!")


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


# ============================================================
# 5. PREPROCESS TEST DATA
# ============================================================

X_test_processed = preprocessor.transform(X_test)

print("Test data preprocessing completed!")
print(
    "Processed test shape:",
    X_test_processed.shape
)


# ============================================================
# 6. GET FEATURE NAMES
# ============================================================

feature_names = preprocessor.get_feature_names_out()

print(
    "Number of features:",
    len(feature_names)
)


# ============================================================
# 7. CONVERT TO DATAFRAME
# ============================================================

X_test_processed_df = pd.DataFrame(
    X_test_processed,
    columns=feature_names,
    index=X_test.index
)


# ============================================================
# 8. CREATE SHAP EXPLAINER
# ============================================================

explainer = shap.LinearExplainer(
    model,
    X_test_processed_df
)

shap_values = explainer(
    X_test_processed_df
)

print("\nSHAP explainer created successfully!")


# ============================================================
# 9. CREATE SHAP OUTPUT FOLDER
# ============================================================

import os

os.makedirs(
    "data/shap",
    exist_ok=True
)


# ============================================================
# 10. GLOBAL SHAP BAR PLOT
# ============================================================

print("\nCreating global SHAP bar plot...")

plt.figure()

shap.plots.bar(
    shap_values,
    max_display=15,
    show=False
)

plt.title(
    "Global Feature Importance - SHAP"
)

plt.tight_layout()

plt.savefig(
    "data/shap/shap_global_bar.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved: data/shap/shap_global_bar.png"
)


# ============================================================
# 11. GLOBAL SHAP BEESWARM PLOT
# ============================================================

print("Creating SHAP beeswarm plot...")

plt.figure()

shap.plots.beeswarm(
    shap_values,
    max_display=15,
    show=False
)

plt.title(
    "SHAP Feature Impact on Churn"
)

plt.tight_layout()

plt.savefig(
    "data/shap/shap_beeswarm.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved: data/shap/shap_beeswarm.png"
)


# ============================================================
# 12. LOCAL EXPLANATION
# ============================================================

sample_index = 0

print("\nCreating local explanation...")
print(
    "Explaining test customer:",
    sample_index
)


# ============================================================
# 13. LOCAL WATERFALL PLOT
# ============================================================

plt.figure()

shap.plots.waterfall(
    shap_values[sample_index],
    max_display=12,
    show=False
)

plt.title(
    "Local SHAP Explanation - Customer"
)

plt.tight_layout()

plt.savefig(
    "data/shap/shap_local_waterfall.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved: data/shap/shap_local_waterfall.png"
)


# ============================================================
# 14. PREDICT SAMPLE CUSTOMER
# ============================================================

sample_data = X_test.iloc[[sample_index]]

sample_processed = preprocessor.transform(
    sample_data
)

prediction = model.predict(
    sample_processed
)[0]

probability = model.predict_proba(
    sample_processed
)[0][1]


# ============================================================
# 15. DISPLAY PREDICTION
# ============================================================

print("\n")
print("=" * 60)
print("CUSTOMER CHURN PREDICTION")
print("=" * 60)

print(
    "Prediction:",
    "CHURN" if prediction == 1 else "NOT CHURN"
)

print(
    f"Churn Probability: {probability:.4f}"
)

print(
    f"Churn Probability: {probability * 100:.2f}%"
)


# ============================================================
# 16. DISPLAY CUSTOMER DATA
# ============================================================

print("\nCustomer information:")
print("-" * 60)

print(
    sample_data.to_string(
        index=False
    )
)


# ============================================================
# 17. DISPLAY TOP SHAP FEATURES
# ============================================================

print("\nTop factors influencing this prediction:")
print("-" * 60)

sample_shap_values = shap_values[
    sample_index
].values

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "SHAP Value": sample_shap_values
})

feature_importance["Absolute SHAP"] = (
    feature_importance["SHAP Value"]
    .abs()
)

feature_importance = (
    feature_importance
    .sort_values(
        "Absolute SHAP",
        ascending=False
    )
    .head(10)
)

print(
    feature_importance[
        ["Feature", "SHAP Value"]
    ].to_string(
        index=False
    )
)


# ============================================================
# 18. SAVE SHAP FEATURE IMPORTANCE
# ============================================================

feature_importance.to_csv(
    "data/shap/shap_local_features.csv",
    index=False
)

print(
    "\nSaved: data/shap/shap_local_features.csv"
)


# ============================================================
# 19. FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 60)
print("SHAP ANALYSIS COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nGenerated files:")

print(
    "1. data/shap/shap_global_bar.png"
)

print(
    "2. data/shap/shap_beeswarm.png"
)

print(
    "3. data/shap/shap_local_waterfall.png"
)

print(
    "4. data/shap/shap_local_features.csv"
)