
from huggingface_hub import snapshot_download

REPO_ID = "qcf568/TFR"
LOCAL_DIR = "./datasets/TFR"

path = snapshot_download(
    repo_id=REPO_ID,
    repo_type="model",
    local_dir=LOCAL_DIR,
    max_workers=4,
)

print(f"Downloaded to: {path}")