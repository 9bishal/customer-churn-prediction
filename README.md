# Customer Churn Prediction System

An end-to-end machine learning system that predicts whether a subscription
customer is likely to churn (cancel their subscription), and explains *why*
— deployed as an interactive Streamlit dashboard.

![Architecture Diagram](images/architecture_diagram.png)

## Problem Statement

Subscription businesses (SaaS, telecom, streaming, fintech) lose recurring
revenue every time a customer cancels. Acquiring a new customer typically
costs 5-7x more than retaining an existing one. This project predicts churn
risk **before it happens**, so retention teams can intervene with the right
customers at the right time.

## Dataset

[IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) —
7,043 customers, 21 original features (demographics, account info, services
subscribed), ~26.5% churn rate.

## Key Insights from EDA

![Churn Rate by Contract Type](images/churn_by_contract.png)

- **Month-to-month contracts churn far more** than one-year or two-year contracts — the single strongest signal in the data.
- **Fiber optic customers churn more** than DSL or no-internet customers, possibly due to pricing or service quality.
- **Low-tenure customers (0-12 months) churn the most** — motivated the `tenure_group` feature.
- Class imbalance (~26.5% churn) means **accuracy is the wrong metric** — precision, recall, F1, and ROC-AUC are used instead.

Full analysis: [`notebooks/eda.ipynb`](notebooks/eda.ipynb)

## Approach

1. **Data cleaning**: fixed blank `TotalCharges` values (new customers with tenure=0), dropped `customerID`, encoded target as binary.
2. **Feature engineering**: added `tenure_group` (New/Medium/Long-term buckets) and `avg_monthly_spend` (lifetime average spend) — both with direct business rationale.
3. **Encoding**: one-hot encoded all categorical features (`drop_first=True` to avoid the dummy variable trap).
4. **Modeling**: built `sklearn.Pipeline` (ColumnTransformer for scaling + model) to prevent train/serve skew. Trained and compared Logistic Regression vs Random Forest, both with `class_weight="balanced"` to handle imbalance.
5. **Model selection**: chose the model with the higher **recall**, since for this business problem, missing a churner (false negative) is far costlier than flagging a loyal customer (false positive).
6. **Deployment**: Streamlit app where a user inputs customer details and gets a churn probability, risk tier, and a chart of the top global churn drivers.

## Results

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.503 | **0.778** | 0.611 | 0.846 |
| Random Forest | 0.553 | 0.749 | 0.636 | 0.842 |

**Selected model: Logistic Regression** — catches 78% of customers who will
actually churn, with a ROC-AUC of 0.846 (ranks churners above non-churners
85% of the time).

## Project Structure

```
customer-churn-prediction/
├── data/
│   ├── raw/telco_churn.csv
│   └── processed/churn_clean.csv
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── data_preprocessing.py   # cleaning, feature engineering, encoding
│   ├── train_model.py          # trains, evaluates, saves the model
│   └── predict.py               # reusable prediction function (used by app)
├── models/
│   └── churn_model.pkl
├── app/
│   └── streamlit_app.py        # interactive dashboard
├── images/
├── requirements.txt
└── README.md
```

## How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/customer-churn-prediction.git
cd customer-churn-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the preprocessing + training pipeline (regenerates the model)
python src/data_preprocessing.py
python src/train_model.py

# 4. Launch the dashboard
python -m streamlit run app/streamlit_app.py
```

## Tech Stack

Python · Pandas · NumPy · Scikit-learn · Matplotlib · Streamlit

## Future Improvements

- Tune the classification decision threshold based on retention team capacity (precision/recall tradeoff)
- Add SHAP for per-prediction (local) explanations, not just global feature importance
- Hyperparameter tuning via `GridSearchCV`
- Expose predictions via a FastAPI endpoint for integration with a CRM
