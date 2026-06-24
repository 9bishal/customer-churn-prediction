"""
data_preprocessing.py

Purpose:
    Load the raw Telco Customer Churn dataset, clean it, engineer a couple
    of simple features, and save a "model-ready" version to data/processed/.

Run this file directly to produce data/processed/churn_clean.csv:
    python src/data_preprocessing.py
"""

import os

import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "telco_churn.csv")
PROCESSED_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "churn_clean.csv")


def reset_processed_file() -> None:
    if os.path.exists(PROCESSED_PATH):
        os.remove(PROCESSED_PATH)


def load_data(path: str = RAW_PATH) -> pd.DataFrame:
    """Load the raw CSV into a pandas DataFrame."""
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix data quality issues in the raw data.

    Issues handled:
    1. 'TotalCharges' is stored as text and has 11 rows with blank strings
       (these are new customers with tenure = 0, so TotalCharges should be 0).
    2. 'customerID' is just an identifier — it has no predictive value and
       must be dropped before modeling.
    3. 'Churn' (target column) is "Yes"/"No" text — convert to 1/0 so models
       can use it.
    """
    df = df.copy()

    # 1. Fix TotalCharges: blank strings -> 0, then convert column to numeric
    df["TotalCharges"] = df["TotalCharges"].replace(" ", "0")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])

    # 2. Drop the ID column — it's a unique identifier, not a feature
    df = df.drop(columns=["customerID"])

    # 3. Convert target column to binary (1 = churned, 0 = stayed)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create two simple, business-meaningful features.

    1. tenure_group: buckets customers into New / Medium / Long-term.
       Why: churn behaviour is very different for brand-new customers
       (still in "trial" mindset) vs long-term customers (more loyal).
       A tree-based model can learn this from raw 'tenure' too, but an
       explicit bucket makes the relationship easier to inspect and explain.

    2. avg_monthly_spend: TotalCharges / (tenure + 1).
       Why: two customers can have the same MonthlyCharges today but very
       different historical spending patterns (e.g. plan changes, discounts).
       This feature captures their *actual* average spend over their
       lifetime with the company. We add 1 to tenure to avoid dividing by
       zero for brand-new customers (tenure = 0).
    """
    df = df.copy()

    # Feature 1: tenure_group
    # Fixed bin edges (not data-dependent) so this works identically for
    # both the full training dataset AND a single customer record at
    # prediction time. 9999 is a safe upper bound (max real tenure is 72).
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 48, 9999],
        labels=["New (0-12 mo)", "Medium (13-48 mo)", "Long-term (49+ mo)"],
    )

    # Feature 2: avg_monthly_spend
    df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"] + 1)

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert categorical (text) columns into numeric columns using one-hot
    encoding, so the model can use them.

    pd.get_dummies() creates a new 0/1 column for each category
    (e.g. 'Contract' becomes 'Contract_One year', 'Contract_Two year', etc.,
    with 'Month-to-month' as the implied baseline via drop_first=True).

    drop_first=True avoids redundant columns: if you know a customer is
    NOT 'One year' and NOT 'Two year', they must be 'Month-to-month' —
    so we don't need a separate column for it.
    """
    df = df.copy()

    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    # tenure_group is a category dtype (from pd.cut), include it too
    categorical_cols += ["tenure_group"]

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return df


def run_pipeline(raw_path: str = RAW_PATH, save_path: str = PROCESSED_PATH) -> pd.DataFrame:
    """Run the full preprocessing pipeline and save the result."""
    reset_processed_file()
    df = load_data(raw_path)
    df = clean_data(df)
    df = add_features(df)
    df = encode_categoricals(df)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Processed data saved to {save_path}")
    print(f"Final shape: {df.shape}")

    return df


if __name__ == "__main__":
    run_pipeline()
