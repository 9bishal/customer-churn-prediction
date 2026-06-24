"""
predict.py

Purpose:
    Provide a single, reusable function that takes a RAW customer record
    (as a plain Python dict, using human-readable values like "Yes"/"No"
    and "Month-to-month") and returns a churn prediction.

    This is the file the Streamlit app calls. By centralizing prediction
    logic here, both the app and any future API (FastAPI) use the exact
    same transformation steps that were used during training.
"""

import os
import logging
import warnings
import joblib
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.exceptions import InconsistentVersionWarning

from data_preprocessing import clean_data, add_features, encode_categoricals

logger = logging.getLogger(__name__)


# Resolve paths relative to the project root (one level above src/), so this
# module works whether it's run from the project root or from inside src/.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "churn_model.pkl")

# The exact column order the model was trained on. We reindex incoming
# data to match this, filling any missing one-hot columns with 0.
# (Generated from data/processed/churn_clean.csv, excluding 'Churn')
TRAINING_COLUMNS = [
    "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges", "avg_monthly_spend",
    "gender_Male", "Partner_Yes", "Dependents_Yes", "PhoneService_Yes",
    "MultipleLines_No phone service", "MultipleLines_Yes",
    "InternetService_Fiber optic", "InternetService_No",
    "OnlineSecurity_No internet service", "OnlineSecurity_Yes",
    "OnlineBackup_No internet service", "OnlineBackup_Yes",
    "DeviceProtection_No internet service", "DeviceProtection_Yes",
    "TechSupport_No internet service", "TechSupport_Yes",
    "StreamingTV_No internet service", "StreamingTV_Yes",
    "StreamingMovies_No internet service", "StreamingMovies_Yes",
    "Contract_One year", "Contract_Two year", "PaperlessBilling_Yes",
    "PaymentMethod_Credit card (automatic)", "PaymentMethod_Electronic check",
    "PaymentMethod_Mailed check",
    "tenure_group_Medium (13-48 mo)", "tenure_group_Long-term (49+ mo)",
]


def load_model(path: str = MODEL_PATH):
    """
    Load the trained pipeline (preprocessing + model) from disk.
    
    Includes robust checks for scikit-learn version compatibility
    and validates that the loaded object is a valid scikit-learn Pipeline.
    """
    current_version = sklearn.__version__
    logger.info("Current scikit-learn version: %s", current_version)
    print(f"[INFO] Current scikit-learn version: {current_version}")
    
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found at {path}. "
            "Please run 'python src/train_model.py' to train and save the model."
        )
        
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InconsistentVersionWarning)
            model = joblib.load(path)
    except Exception as e:
        logger.error("Failed to load model file from %s: %s", path, str(e))
        raise RuntimeError(
            f"Error loading model from {path}. This can happen due to an incompatible pickle format. "
            "Please retrain the model by running:\n"
            "    python src/train_model.py"
        ) from e
        
    # Verify loaded object is a valid Pipeline
    if not isinstance(model, Pipeline):
        raise ValueError(
            f"Loaded model is of type {type(model).__name__}, but a scikit-learn Pipeline was expected. "
            "Please retrain the model by running:\n"
            "    python src/train_model.py"
        )
        
    # Check if 'model' step exists in the pipeline
    if "model" not in model.named_steps:
        raise ValueError(
            "Loaded Pipeline is missing the required 'model' step. "
            "Please retrain the model by running:\n"
            "    python src/train_model.py"
        )
        
    final_estimator = model.named_steps["model"]
    estimator_type = type(final_estimator).__name__
    logger.info("Loaded final estimator of type: %s", estimator_type)
    print(f"[INFO] Loaded final estimator of type: {estimator_type}")
    
    # Check if the model has _sklearn_version attribute
    saved_version = getattr(final_estimator, "_sklearn_version", None) or getattr(model, "_sklearn_version", None)
    if saved_version:
        logger.info("Model was saved with scikit-learn version: %s", saved_version)
        print(f"[INFO] Model was saved with scikit-learn version: {saved_version}")
        if saved_version != current_version:
            logger.warning(
                "Version mismatch! Model was saved with version %s but running on %s.",
                saved_version,
                current_version
            )
            print(f"[WARNING] Version mismatch! Model was saved with version {saved_version} but running on {current_version}.")
    else:
        logger.info("Model does not contain scikit-learn version metadata.")
        print("[INFO] Model does not contain scikit-learn version metadata (likely trained on an older scikit-learn version).")
        
    # Backward compatibility patch for LogisticRegression unpickled in different scikit-learn versions
    if estimator_type == "LogisticRegression" and not hasattr(final_estimator, "multi_class"):
        logger.info("Patching missing 'multi_class' attribute on LogisticRegression for compatibility.")
        print("[INFO] Patching missing 'multi_class' attribute on LogisticRegression for compatibility.")
        final_estimator.multi_class = "auto"
        
    return model


def preprocess_single_record(customer: dict) -> pd.DataFrame:
    """
    Apply the SAME cleaning + feature engineering + encoding steps used
    during training, but to a single customer record.

    Why reuse clean_data / add_features / encode_categoricals from
    data_preprocessing.py instead of rewriting the logic here?
    - If preprocessing logic ever changes (e.g. a new feature is added),
      we only need to change it in ONE place. Both the training script
      and this prediction function automatically stay in sync.
    - This directly answers the interview question: "how do you prevent
      training/serving skew?"
    """
    df = pd.DataFrame([customer])

    # clean_data() expects a 'Churn' and 'customerID' column to exist
    # (it drops/transforms them). For a single prediction record neither
    # is relevant, so we add dummy placeholders before cleaning, and the
    # function will drop/convert them safely.
    df["customerID"] = "temp"
    df["Churn"] = "No"

    df = clean_data(df)
    df = add_features(df)
    df = encode_categoricals(df)

    # Drop the target column (not a feature for prediction)
    df = df.drop(columns=["Churn"])

    # Align columns with training data:
    # - any one-hot column that didn't appear for this single record
    #   (e.g. customer's PaymentMethod category) gets added with value 0
    # - columns are ordered exactly as the model expects
    df = df.reindex(columns=TRAINING_COLUMNS, fill_value=0)

    return df


def predict_churn(customer: dict, model=None) -> dict:
    """
    Main prediction function.

    Args:
        customer: dict with raw, human-readable keys/values (see example below)
        model: optional pre-loaded pipeline. If None, loads from disk.
               Passing a pre-loaded model avoids repeated disk reads
               (e.g. the Streamlit app loads it once and reuses it).
            {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 5,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.0,
                "TotalCharges": 350.0,
            }

    Output:
        dict with churn_probability (float 0-1), risk_tier (str),
        and prediction (0 or 1)
    """
    if model is None:
        model = load_model()
    X = preprocess_single_record(customer)

    try:
        probability = model.predict_proba(X)[0][1]
    except AttributeError as e:
        current_version = sklearn.__version__
        logger.error("Prediction failed: %s", str(e))
        raise RuntimeError(
            f"Prediction failed due to model/library compatibility issue (scikit-learn={current_version}). "
            f"AttributeError: {str(e)}\n"
            "This typically happens when using a model file serialized with a different scikit-learn version.\n"
            "Please retrain and re-save the model using:\n"
            "    python src/train_model.py"
        ) from e

    prediction = int(probability >= 0.5)

    if probability < 0.30:
        risk_tier = "Low"
    elif probability < 0.60:
        risk_tier = "Medium"
    else:
        risk_tier = "High"

    return {
        "churn_probability": round(float(probability), 3),
        "risk_tier": risk_tier,
        "prediction": prediction,
    }


def get_feature_importance(top_n: int = 10, model=None) -> pd.DataFrame:
    """
    Return the top N most influential features for the loaded model.

    For Logistic Regression, the model's coefficients indicate direction
    and strength of each feature's effect on churn probability (after
    scaling, so coefficients are comparable). A positive coefficient
    means the feature increases churn probability; negative means it
    decreases it.

    This is used by the Streamlit app to show "why" a prediction was made
    in general terms (global feature importance, not per-prediction
    explanation — which would require SHAP, an unnecessary addition here).
    """
    if model is None:
        model = load_model()
    final_estimator = model.named_steps["model"]

    if hasattr(final_estimator, "coef_"):
        importances = final_estimator.coef_[0]
    elif hasattr(final_estimator, "feature_importances_"):
        importances = final_estimator.feature_importances_
    else:
        raise ValueError("Model has no interpretable importance attribute")

    importance_df = pd.DataFrame({
        "feature": TRAINING_COLUMNS,
        "importance": importances,
    })

    importance_df["abs_importance"] = importance_df["importance"].abs()
    importance_df = importance_df.sort_values("abs_importance", ascending=False).head(top_n)

    return importance_df[["feature", "importance"]]


if __name__ == "__main__":
    # Quick manual test
    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.0,
        "TotalCharges": 190.0,
    }

    result = predict_churn(sample_customer)
    print("Prediction result:", result)

    print("\nTop feature importances:")
    print(get_feature_importance())
