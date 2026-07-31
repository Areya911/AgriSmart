"""
model_downloader.py
-------------------
Downloads trained ML model weights from Google Drive on first startup.
Models are cached locally and reused on subsequent starts.
This ensures the GitHub repo stays lean (no large binaries).

USAGE: called automatically from app.py before model loading.

Google Drive file IDs — update these after uploading your model files:
  1. Open drive.google.com and upload each .h5 / .pt file
  2. Right-click → Share → "Anyone with the link"
  3. Copy the file ID from the URL:
        https://drive.google.com/file/d/<FILE_ID>/view
  4. Paste the FILE_ID below.
"""

import os
import sys

# ─────────────────────────────────────────────────────────────────────────────
# MODEL REGISTRY
# Update FILE_ID for each model after uploading to Google Drive.
# ─────────────────────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "soil_classifier.h5": {
        "gdrive_id": os.getenv("SOIL_MODEL_GDRIVE_ID", "YOUR_SOIL_MODEL_FILE_ID"),
        "description": "Soil type CNN classifier (Keras/H5)",
    },
    "leaf_disease_mobilenet_finetuned.h5": {
        "gdrive_id": os.getenv("LEAF_MODEL_GDRIVE_ID", "YOUR_LEAF_MODEL_FILE_ID"),
        "description": "Leaf disease MobileNetV2 classifier (Keras/H5)",
    },
    "yolov8n.pt": {
        "gdrive_id": os.getenv("YOLO_MODEL_GDRIVE_ID", "YOUR_YOLO_MODEL_FILE_ID"),
        "description": "YOLOv8 nano plant/leaf detector (PyTorch)",
    },
}


def _gdown_available() -> bool:
    try:
        import gdown  # noqa: F401
        return True
    except ImportError:
        return False


def download_models(base_dir: str = ".") -> None:
    """
    Download all models in MODEL_REGISTRY that are not already cached locally.
    Skips any model whose gdrive_id is still the placeholder string.
    """
    if not _gdown_available():
        print(
            "[model_downloader] WARNING: 'gdown' is not installed. "
            "Run: pip install gdown\n"
            "Skipping automatic model download."
        )
        return

    import gdown

    for filename, info in MODEL_REGISTRY.items():
        dest_path = os.path.join(base_dir, filename)
        gdrive_id = info["gdrive_id"]

        # Skip if already on disk
        if os.path.isfile(dest_path):
            print(f"[model_downloader] ✓ Found cached: {filename}")
            continue

        # Skip if placeholder not yet filled in
        if "YOUR_" in gdrive_id:
            print(
                f"[model_downloader] ⚠ Skipping '{filename}': "
                f"Google Drive file ID not configured.\n"
                f"  → Set env var or edit MODEL_REGISTRY in model_downloader.py"
            )
            continue

        url = f"https://drive.google.com/uc?id={gdrive_id}"
        print(f"[model_downloader] ⬇ Downloading {filename} ({info['description']}) …")
        try:
            gdown.download(url, dest_path, quiet=False, fuzzy=True)
            if os.path.isfile(dest_path):
                size_mb = os.path.getsize(dest_path) / (1024 * 1024)
                print(f"[model_downloader] ✓ Saved {filename} ({size_mb:.1f} MB)")
            else:
                print(f"[model_downloader] ✗ Download failed silently for {filename}")
        except Exception as exc:
            print(f"[model_downloader] ✗ Error downloading {filename}: {exc}")


if __name__ == "__main__":
    # Allow running standalone: python model_downloader.py
    base = sys.argv[1] if len(sys.argv) > 1 else "."
    download_models(base_dir=base)
