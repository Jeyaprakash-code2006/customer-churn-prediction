# Customer Churn Prediction for Subscription Businesses

## Project Overview

This project uses machine learning to predict customer churn for subscription-based businesses. It analyzes customer information, estimates churn probability, identifies high-risk customers, and supports data-driven customer retention decisions.

**Note:** The included dataset is synthetic and intended for educational and demonstration purposes.

## Features

- **Business Dashboard:** View customer statistics, churn rates, revenue metrics, and interactive charts.
- **Churn Prediction:** Estimate the probability that an individual customer will churn.
- **Batch Prediction:** Upload a CSV or Excel file to predict churn for multiple customers.
- **Model Analysis:** Compare Logistic Regression, Random Forest, and XGBoost.
- **Customer Survival:** Explore simplified cohort and survival-analysis results.
- **Risk Segmentation:** Group customers into low-, medium-, and high-risk categories.
- **Retention Priority:** Identify customers for potential retention activities.
- **Data Visualization:** Explore charts, model metrics, and feature explanations.

## Technologies Used

- Python
- Streamlit
- Pandas and NumPy
- Scikit-learn
- XGBoost
- Plotly
- SHAP
- Joblib
- Matplotlib

## Machine Learning Models

Three classification models are evaluated:

1. Logistic Regression
2. Random Forest
3. XGBoost

### Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 76.50% | 64.29% | 45.76% | 53.47% | 0.8033 |
| Random Forest | 75.00% | 57.89% | 55.93% | 56.90% | 0.7462 |
| XGBoost | 74.00% | 56.86% | 49.15% | 52.73% | 0.7556 |

Logistic Regression achieved the highest ROC-AUC in the recorded evaluation. Random Forest achieved the highest recall and F1 score. The preferred model depends on the business objective and the cost of missed churn cases versus false alarms.

## Project Structure

```text
customer-churn-prediction/
├── app.py
├── README.md
├── requirements.txt
├── data/
│   ├── customer_churn.csv
│   ├── cohort_analysis.csv
│   ├── kaplan_meier_survival.csv
│   ├── risk_segmented_customers.csv
│   ├── retention_priority.csv
│   └── shap/
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   └── model_comparison.csv
├── notebooks/
├── src/
└── venv/
```

## Installation and Setup

### 1. Open the project directory

```powershell
cd C:\customer-churn-prediction
```

### 2. Activate the virtual environment

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Run the Streamlit application

```powershell
python -m streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser to use the application.

## Risk Categories

The application uses the following probability thresholds:

- **Low Risk:** Churn probability below 30%.
- **Medium Risk:** Churn probability from 30% to below 60%.
- **High Risk:** Churn probability of 60% or higher.

These thresholds are configurable business rules, not guarantees of customer behavior.

## Limitations

- The dataset is synthetic and may not reflect real customer behavior.
- Model performance on real-world data may differ.
- Revenue-at-risk and retention-priority figures are simplified estimates.
- Survival analysis is illustrative and does not replace analysis using validated time-to-event and censoring data.
- Predictions support decision-making but do not guarantee that a customer will churn.

## Future Enhancements

- Evaluate models on real, appropriately authorized customer data.
- Add model monitoring and periodic retraining.
- Tune decision thresholds based on retention costs and business goals.
- Integrate a database and authenticated user access.
- Add deployment and automated testing.

## Author

Developed as a machine learning and business analytics project.
