import os
import urllib.request
import zipfile
from pathlib import Path


REPO_OWNER = "wutdahack1337"
REPO_NAME = "tampered-text-detection-in-document-image"
RELEASE_TAG = "v1.0.0"

CHECKPOINTS = [
    "best_model.pth",
    "best_model_multi_classes.pth",
    "best_model_multi_label.pth",
]

BASE_URL = f"https://github.com/{REPO_OWNER}/{REPO_NAME}/releases/download/{RELEASE_TAG}"


def download_file(url: str, dest_path: Path) -> bool:
    print(f"Downloading {dest_path.name} from {url}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f" Saved to: {dest_path}")
        return True
    except Exception as e:
        print(f" Failed to download {dest_path.name}: {e}")
        return False


def main():
    dest_dir = Path("checkpoints")
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Check if all models already exist
    missing = [f for f in CHECKPOINTS if not (dest_dir / f).exists()]
    if not missing:
        print("All checkpoints already exist in checkpoints/ directory.")
        return

    print(f"Missing checkpoints: {missing}")

    # Try downloading checkpoints.zip first
    zip_path = Path("checkpoints.zip")
    zip_url = f"{BASE_URL}/checkpoints.zip"
    if download_file(zip_url, zip_path):
        print(f"Extracting {zip_path} to {dest_dir}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(dest_dir)
        zip_path.unlink(missing_ok=True)
        print("Extraction complete.")
        return

    # Fallback to individual .pth files
    for filename in missing:
        target_path = dest_dir / filename
        file_url = f"{BASE_URL}/{filename}"
        download_file(file_url, target_path)

    print("\nCheckpoints processing finished.")


if __name__ == "__main__":
    main()
