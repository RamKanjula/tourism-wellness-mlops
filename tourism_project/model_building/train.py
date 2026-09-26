"""
Model Training with Experiment Tracking
-----------------------------------------
Trains several candidate classifiers to predict ProdTaken (purchase of the
Wellness Tourism Package), logs every run (params + metrics + model
artifact) to MLflow, picks the best model by test-set ROC-AUC, and pushes
the winning pipeline to the Hugging Face Model Hub for deployment.

Run:
    python model_building/train.py
"""

import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

DATA_DIR = "tourism_project/data"
MODEL_DIR = "tourism_project/model_building"
TARGET_COL = "ProdTaken"
RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore", "OwnCar",
    "NumberOfChildrenVisiting", "MonthlyIncome",
]
CATEGORICAL_FEATURES = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched",
    "MaritalStatus", "Designation",
]


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def get_candidate_models():
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            eval_metric="logloss", random_state=RANDOM_STATE,
        ),
    }


def main():
    train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
    test_df = pd.read_csv(f"{DATA_DIR}/test.csv")

    X_train, y_train = train_df.drop(columns=TARGET_COL), train_df[TARGET_COL]
    X_test, y_test = test_df.drop(columns=TARGET_COL), test_df[TARGET_COL]

    mlflow.set_tracking_uri("sqlite:///tourism_project/model_building/mlflow.db")
    mlflow.set_experiment("wellness_tourism_package_prediction")

    results = []
    best_score = -1
    best_pipeline = None
    best_model_name = None

    for name, model in get_candidate_models().items():
        with mlflow.start_run(run_name=name):
            pipeline = Pipeline(steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", model),
            ])
            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_proba),
            }

            mlflow.log_param("model_type", name)
            mlflow.log_params({k: v for k, v in model.get_params().items()
                                if isinstance(v, (int, float, str, bool)) or v is None})
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                pipeline, name="model", serialization_format="cloudpickle"
            )

            results.append({"model": name, **metrics})
            print(f"{name}: {metrics}")

            if metrics["roc_auc"] > best_score:
                best_score = metrics["roc_auc"]
                best_pipeline = pipeline
                best_model_name = name

    results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    print("\n=== Model comparison (sorted by ROC-AUC) ===")
    print(results_df.to_string(index=False))
    print(f"\nBest model: {best_model_name} (ROC-AUC = {best_score:.4f})")

    # Save the winning pipeline locally (used by the Streamlit app / Docker image)
    os.makedirs(MODEL_DIR, exist_ok=True)
    local_model_path = f"{MODEL_DIR}/best_model.joblib"
    joblib.dump(best_pipeline, local_model_path)
    results_df.to_csv(f"{MODEL_DIR}/model_comparison.csv", index=False)
    print(f"Saved best model to {local_model_path}")

    # --- Register the best model on the Hugging Face Model Hub ---
    hf_token = os.getenv("HF_TOKEN")
    hf_username = os.getenv("HF_USERNAME")
    if hf_token and hf_username:
        from huggingface_hub import HfApi, create_repo

        model_repo_id = f"{hf_username}/tourism-wellness-package-model"
        create_repo(repo_id=model_repo_id, token=hf_token, exist_ok=True, private=False)
        api = HfApi(token=hf_token)
        api.upload_file(
            path_or_fileobj=local_model_path,
            path_in_repo="best_model.joblib",
            repo_id=model_repo_id,
        )
        print(f"Model registered at: https://huggingface.co/{model_repo_id}")
    else:
        print("HF_TOKEN/HF_USERNAME not set - skipping model registration (local run).")


if __name__ == "__main__":
    main()
