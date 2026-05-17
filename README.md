# 📡 Telecom Customer Churn Prediction

> **Which customers are about to leave and why?**  
> An end-to-end ML pipeline using XGBoost + SHAP to predict and explain customer churn for telecom companies like Jio / Airtel etc.

---

## 🎯 Business Problem

Telecom companies lose revenue every month when customers cancel their subscriptions ("churn"). Acquiring a new customer costs **5–7× more** than retaining an existing one.

**Goal:** Build a system that:
1. Predicts which customers are likely to churn in the next 30 days
2. Explains *why* each customer is flagged (so teams can act on it)
3. Prioritizes outreach so retention teams call the right people first

---

## 📊 Dataset

| Field | Description |
|-------|-------------|
| `customerID` | Unique customer identifier |
| `tenure` | Months with the company |
| `Contract` | Month-to-month / One year / Two year |
| `MonthlyCharges` | Current monthly bill (₹ or $) |
| `TotalCharges` | Total amount paid |
| `InternetService` | DSL / Fiber optic / None |
| `PaymentMethod` | How they pay |
| `Churn` | **Target** — Yes / No |

~7,000 customers · 21 features · Source: [IBM Telco Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## 🏗️ Project Structure

```
telecom-churn-prediction/
│
├── data/
│   ├── raw/                    # Original CSV files (do not modify)
│   └── processed/              # Cleaned & feature-engineered data
│
├── sql/
│   ├── 01_create_tables.sql    # Schema creation
│   ├── 02_load_data.sql        # Data ingestion
│   ├── 03_eda_queries.sql      # Exploratory analysis
│   ├── 04_feature_engineering.sql  # SQL-based feature creation
│   └── 05_churn_reporting.sql  # Business reporting queries
│
├── src/
│   ├── data_cleaning.py        # Preprocessing pipeline
│   ├── feature_engineering.py  # Feature creation
│   ├── train_model.py          # XGBoost training + tuning
│   ├── evaluate_model.py       # Metrics + confusion matrix
│   ├── shap_explainer.py       # SHAP value computation
│   └── predict.py              # Inference on new customers
│
├── notebooks/
│   ├── 01_EDA.ipynb            # Exploratory Data Analysis
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_Model_Training.ipynb
│   └── 04_SHAP_Explainability.ipynb
│
├── models/
│   └── xgboost_churn_model.pkl # Saved model artifact
│
├── reports/
│   └── model_performance.md    # Accuracy, precision, recall
│
├── tests/
│   └── test_pipeline.py        # Unit tests
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ How It Works

### Step 1 — Data Cleaning
- Remove nulls (especially in `TotalCharges`)
- Encode categorical variables (contract type, payment method)
- Scale numerical features

### Step 2 — Feature Engineering
- `tenure_group`: Bucketed tenure (new / mid / loyal)
- `avg_monthly_bill`: `TotalCharges / tenure`
- `charge_to_tenure_ratio`: Spend intensity signal
- `is_month_to_month`: High-risk contract flag

### Step 3 — Model Training (XGBoost)
XGBoost builds hundreds of decision trees, each correcting the previous one's errors. It asks questions like:
- Is contract month-to-month? → higher churn risk
- Is tenure < 6 months? → higher churn risk
- Is monthly bill < ₹500? → lower churn risk

### Step 4 — Explainability (SHAP)
SHAP values answer: *"For THIS specific customer, why did the model predict churn?"*

```
Customer #4821 — Churn Probability: 78%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Contract: Month-to-month    → +0.42 (pushes toward churn)
Tenure: 2 months            → +0.31 (pushes toward churn)
InternetService: Fiber      → +0.18 (pushes toward churn)
MonthlyCharges: ₹450        → -0.12 (pushes away from churn)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Action: Offer annual plan upgrade with 20% discount
```

---

## 📈 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 91% |
| **Precision** | 88% |
| **Recall** | 84% |
| **F1 Score** | 86% |
| **ROC-AUC** | 0.94 |

Models compared: Logistic Regression · Decision Tree · Random Forest · **XGBoost ✓**

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/telecom-churn-prediction.git
cd telecom-churn-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline
python src/train_model.py

# 4. Predict on new data
python src/predict.py --input data/raw/new_customers.csv

# 5. Generate SHAP explanations
python src/shap_explainer.py --customer_id 4821
```

---

## 🛢️ SQL Usage

```bash
# Set up the database and load data
psql -U postgres -d telecom_db -f sql/01_create_tables.sql
psql -U postgres -d telecom_db -f sql/02_load_data.sql

# Run EDA queries
psql -U postgres -d telecom_db -f sql/03_eda_queries.sql

# Generate churn report
psql -U postgres -d telecom_db -f sql/05_churn_reporting.sql
```

---

## 💡 Business Impact

- Retention team gets a **daily ranked list** of at-risk customers
- Each customer comes with a **reason code** (contract type, tenure, billing)
- Estimated **15–20% reduction in monthly churn** with targeted offers

---

## 🛠️ Tech Stack

`Python 3.10` · `XGBoost` · `SHAP` · `Scikit-learn` · `PostgreSQL` · `Matplotlib` 

---

## 👤 Author

**Samiksha** — [GitHub](https://github.com/sammi2213) · [LinkedIn](www.linkedin.com/in/
samiksha-portfolio)
