"""
Data Registration
------------------
Uploads the raw tourism.csv dataset to a Hugging Face Hub *Dataset* repository
so that it is version-controlled and can be pulled by later pipeline stages
(data preparation, training) instead of relying on a local file.

Requires environment variables:
    HF_TOKEN     - a Hugging Face access token with write permission
    HF_USERNAME  - your Hugging Face username

Run:
    python model_building/data_registration.py
"""

import os
from huggingface_hub import HfApi, create_repo

HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = os.getenv("HF_USERNAME")
DATASET_REPO_ID = f"{HF_USERNAME}/tourism-wellness-package-dataset"
LOCAL_DATA_PATH = "tourism_project/data/tourism.csv"


def main():
    if not HF_TOKEN or not HF_USERNAME:
        print("HF_TOKEN / HF_USERNAME not set - skipping registration (local/demo run).")
        return

    api = HfApi(token=HF_TOKEN)
    create_repo(
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        token=HF_TOKEN,
        exist_ok=True,
        private=False,
    )
    api.upload_file(
        path_or_fileobj=LOCAL_DATA_PATH,
        path_in_repo="tourism.csv",
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
    )
    print(f"Dataset registered at: https://huggingface.co/datasets/{DATASET_REPO_ID}")


if __name__ == "__main__":
    main()
