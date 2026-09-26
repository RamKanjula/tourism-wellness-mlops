"""
Data Preparation
------------------
Cleans the raw tourism dataset and produces train/test splits used for
model training. Also pushes the processed splits back to the Hugging Face
dataset repo so the training stage of the pipeline can pull them directly
(rather than depending on local files existing on the CI runner).

Run:
    python model_building/data_prep.py
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DATA_PATH = "tourism_project/data/tourism.csv"
OUT_DIR = "tourism_project/data"
TARGET_COL = "ProdTaken"
ID_COLS = ["Unnamed: 0", "CustomerID"]
RANDOM_STATE = 42
TEST_SIZE = 0.2


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop identifier / junk index columns - not predictive, and leaking
    # CustomerID into a model would be a data leakage risk.
    for col in ID_COLS:
        if col in df.columns:
            df = df.drop(columns=col)

    # Fix inconsistent category labels found during EDA.
    df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
    df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})

    # Drop exact duplicate rows, if any.
    df = df.drop_duplicates()

    # Impute any missing numeric values with the median (robust to outliers)
    # and any missing categorical values with the mode. Makes the pipeline
    # robust to future data refreshes that may contain nulls.
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.drop(TARGET_COL)
    cat_cols = df.select_dtypes(include=["object", "string"]).columns

    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


def main():
    df = pd.read_csv(RAW_DATA_PATH)
    df_clean = clean_data(df)

    X = df_clean.drop(columns=[TARGET_COL])
    y = df_clean[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    os.makedirs(OUT_DIR, exist_ok=True)
    train_df.to_csv(f"{OUT_DIR}/train.csv", index=False)
    test_df.to_csv(f"{OUT_DIR}/test.csv", index=False)

    print(f"Cleaned rows: {len(df_clean)} (raw had {len(df)})")
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    print(f"Train target balance:\n{y_train.value_counts(normalize=True)}")

    # --- Push processed splits to the HF dataset repo (pipeline/CI step) ---
    hf_token = os.getenv("HF_TOKEN")
    hf_username = os.getenv("HF_USERNAME")
    if hf_token and hf_username:
        from huggingface_hub import HfApi

        api = HfApi(token=hf_token)
        repo_id = f"{hf_username}/tourism-wellness-package-dataset"
        for fname in ["train.csv", "test.csv"]:
            api.upload_file(
                path_or_fileobj=f"{OUT_DIR}/{fname}",
                path_in_repo=fname,
                repo_id=repo_id,
                repo_type="dataset",
            )
        print(f"Train/test splits pushed to https://huggingface.co/datasets/{repo_id}")
    else:
        print("HF_TOKEN/HF_USERNAME not set - skipping upload (local run).")


if __name__ == "__main__":
    main()
