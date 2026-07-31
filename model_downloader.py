"""
model_downloader.py
-------------------
Downloads trained ML model weights from Google Drive / Direct URLs on startup.
Models are cached locally and reused on subsequent starts.
"""

import os
import sys
import requests

MODEL_REGISTRY = {
    "soil_classifier.h5": {
        "gdrive_id": os.getenv("SOIL_MODEL_GDRIVE_ID", ""),
        "direct_url": os.getenv("SOIL_MODEL_URL", ""),
        "description": "Soil type CNN classifier (Keras/H5)",
    },
    "leaf_disease_mobilenet_finetuned.h5": {
        "gdrive_id": os.getenv("LEAF_MODEL_GDRIVE_ID", ""),
        "direct_url": os.getenv("LEAF_MODEL_URL", ""),
        "description": "Leaf disease MobileNetV2 classifier (Keras/H5)",
    },
    "yolov8n.pt": {
        "gdrive_id": os.getenv("YOLO_MODEL_GDRIVE_ID", ""),
        "direct_url": os.getenv("YOLO_MODEL_URL", "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt"),
        "description": "YOLOv8 nano plant/leaf detector (PyTorch)",
    },
}

def _download_file_direct(url: str, dest_path: str) -> bool:
    try:
        print(f"[model_downloader] Downloading via HTTP: {url}")
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return os.path.isfile(dest_path) and os.path.getsize(dest_path) > 1000
    except Exception as e:
        print(f"[model_downloader] HTTP download failed: {e}")
        return False

def download_models(base_dir: str = ".") -> None:
    """Download all models in MODEL_REGISTRY if not already cached on disk."""
    gdown = None
    try:
        import gdown as _gdown
        gdown = _gdown
    except ImportError:
        pass

    for filename, info in MODEL_REGISTRY.items():
        dest_path = os.path.join(base_dir, filename)

        # Skip if already cached on disk and valid
        if os.path.isfile(dest_path) and os.path.getsize(dest_path) > 1000:
            print(f"[model_downloader] ✓ Cached model ready: {filename}")
            continue

        gdrive_id = (info.get("gdrive_id") or "").strip()
        direct_url = (info.get("direct_url") or "").strip()

        success = False

        # Try Google Drive if gdrive_id is set
        if gdrive_id and "YOUR_" not in gdrive_id:
            print(f"[model_downloader] ⬇ Fetching {filename} from Google Drive ID: {gdrive_id}")
            if gdown:
                try:
                    gdown.download(id=gdrive_id, output=dest_path, quiet=False)
                    if os.path.isfile(dest_path) and os.path.getsize(dest_path) > 1000:
                        success = True
                except Exception as e:
                    print(f"[model_downloader] gdown failed: {e}")
            if not success:
                # Fallback URL download for Google Drive
                uc_url = f"https://drive.google.com/uc?export=download&id={gdrive_id}"
                success = _download_file_direct(uc_url, dest_path)

        # Try Direct URL if gdrive_id wasn't set or failed
        if not success and direct_url:
            success = _download_file_direct(direct_url, dest_path)

        if success:
            size_mb = os.path.getsize(dest_path) / (1024 * 1024)
            print(f"[model_downloader] ✓ Successfully saved {filename} ({size_mb:.1f} MB)")
        else:
            print(f"[model_downloader] ℹ Model {filename} not configured/downloaded yet (will use robust instant fallback).")

if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "."
    download_models(base_dir=base)
