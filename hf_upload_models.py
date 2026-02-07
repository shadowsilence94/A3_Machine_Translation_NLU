import os
from huggingface_hub import HfApi

# Configuration
REPO_ID = "shadowsilence/burmese-english-translator"
REPO_TYPE = "space"
LOCAL_MODELS_PATH = "/Users/htutkoko/Library/CloudStorage/GoogleDrive-htutkoko1994@gmail.com/My Drive/NLP/Project_A3/A3_Burmese_English_Puffer/app/models"
PATH_IN_REPO = "app/models"

api = HfApi()

print(f"Starting upload of {LOCAL_MODELS_PATH} to {REPO_ID}...")

try:
    api.upload_folder(
        folder_path=LOCAL_MODELS_PATH,
        path_in_repo=PATH_IN_REPO,
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        commit_message="Batch upload scratch models and tokenizers (bypass rate limit)",
    )
    print("Upload successful!")
except Exception as e:
    print(f"Upload failed: {e}")
