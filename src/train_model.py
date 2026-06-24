"""
train_model.py

Purpose:
    Load the processed churn data, split it into train/test sets, train two
    candidate models inside scikit-learn Pipelines, evaluate them with metrics
    appropriate for an imbalanced classification problem, and save the
    best-performing pipeline to disk.

Run this file directly to train and save the model:
    python src/train_model.py
"""

import os

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "churn_clean.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "churn_model.pkl")

# Numeric columns that need scaling. Everything else in the dataset is
# already 0/1 (from one-hot encoding), so only these need StandardScaler.
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "avg_monthly_spend", "SeniorCitizen"]


def load_processed_data(path: str = PROCESSED_PATH):
    """Load the processed dataset and split into features (X) and target (y)."""
    df = pd.read_csv(path)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return X, y


def build_pipeline(model) -> Pipeline:
    """
    Build a scikit-learn Pipeline that:
    1. Scales the numeric columns with StandardScaler
    2. Passes the (already one-hot-encoded) categorical columns through unchanged
    3. Feeds everything into the given model

    Why a Pipeline instead of scaling manually?
    - It bundles preprocessing + model into ONE object. When we save this
      pipeline with joblib, predict.py can call .predict() on raw feature
      rows and get correct results — no risk of forgetting to scale, or
      scaling with different parameters than training.
    - StandardScaler's mean/std are learned ONLY from the training data when
      we call pipeline.fit(X_train, y_train) — this prevents data leakage
      from the test set into training.
    """
    from sklearn.compose import ColumnTransformer

    preprocessor = ColumnTransformer(
        transformers=[("scale", StandardScaler(), NUMERIC_COLS)],
        remainder="passthrough",  # leave one-hot columns as-is
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    return pipeline


def evaluate_model(pipeline: Pipeline, X_test, y_test, name: str) -> dict:
    """
    Evaluate a trained pipeline using metrics suited for imbalanced data.

    Why not accuracy?
    - Our dataset is ~73% "no churn" / 27% "churn". A model that always
      predicts "no churn" would score 73% accuracy while being USELESS.
    - Precision: of customers we flagged as "will churn", how many actually did?
    - Recall: of customers who actually churned, how many did we catch?
    - F1: harmonic mean of precision and recall — a single balanced score.
    - ROC-AUC: how well the model ranks churners above non-churners,
      independent of the chosen decision threshold (0.5).

    For THIS business problem, Recall matters most: missing a churner
    (false negative) means losing a customer permanently. A false positive
    (flagging a loyal customer) just costs a discount/retention email.
    """
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "precision": round(precision_score(y_test, y_pred), 3),
        "recall": round(recall_score(y_test, y_pred), 3),
        "f1": round(f1_score(y_test, y_pred), 3),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 3),
    }

    cm = confusion_matrix(y_test, y_pred)
    print(f"\n--- {name} ---")
    for k, v in metrics.items():
        if k != "model":
            print(f"{k:>10}: {v}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(cm)

    return metrics


def run_training(processed_path: str = PROCESSED_PATH, save_path: str = MODEL_PATH):
    """Train both candidate models, compare them, and save the better one."""
    X, y = load_processed_data(processed_path)

    # Stratified split: keeps the ~27% churn ratio consistent in both
    # train and test sets. Without stratify, a random split could give us
    # a test set with a very different churn rate, making evaluation noisy.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Model 1: Logistic Regression (simple, interpretable baseline) ---
    log_reg = build_pipeline(
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    )
    log_reg.fit(X_train, y_train)
    log_reg_metrics = evaluate_model(log_reg, X_test, y_test, "Logistic Regression")

    # --- Model 2: Random Forest (better at capturing non-linear patterns) ---
    rf = build_pipeline(
        RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42, max_depth=10
        )
    )
    rf.fit(X_train, y_train)
    rf_metrics = evaluate_model(rf, X_test, y_test, "Random Forest")

    # Pick the model with the higher Recall (our priority metric for churn)
    if rf_metrics["recall"] >= log_reg_metrics["recall"]:
        best_pipeline, best_name = rf, "Random Forest"
    else:
        best_pipeline, best_name = log_reg, "Logistic Regression"

    print(f"\nSelected best model: {best_name}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(best_pipeline, save_path)
    print(f"Saved pipeline to {save_path}")

    return best_pipeline, X_train.columns.tolist()


if __name__ == "__main__":
    run_training()