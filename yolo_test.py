# yolo_test.py
import os
import numpy as np
from PIL import Image
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "yolov8n.pt")
img_path = os.path.join(BASE_DIR, "diseased_leaf.png")  # replace with the image you used in UI

print("Model exists:", os.path.isfile(model_path))
print("Image exists:", os.path.isfile(img_path))

if not os.path.isfile(model_path):
    raise SystemExit("yolov8n.pt not found at: " + model_path)

model = YOLO(model_path)
print("YOLO model loaded:", model)

# Load image and print shape/dtype
img = Image.open(img_path).convert("RGB")
arr = np.array(img)
print("Image shape:", arr.shape, "dtype:", arr.dtype, "min/max:", arr.min(), arr.max())

# Run prediction with lower conf threshold for debugging
results = model.predict(arr, conf=0.2, imgsz=640)  # reduce conf so weaker boxes show
print("Number of results objects returned:", len(results))

for i, r in enumerate(results):
    print(f"--- Result {i} ---")
    # try to access boxes and names robustly
    try:
        names = r.names
        boxes = getattr(r, "boxes", None)
        if boxes is not None:
            print("Boxes count:", len(boxes))
            # boxes.cls may be a tensor-like - convert to list
            cls_list = [int(x) for x in boxes.cls] if hasattr(boxes, "cls") else []
            confs = [float(x) for x in boxes.conf] if hasattr(boxes, "conf") else []
            xyxy = [list(map(float, b)) for b in boxes.xyxy] if hasattr(boxes, "xyxy") else []
            for idx, (c, conf, xy) in enumerate(zip(cls_list, confs, xyxy)):
                label = names[c] if c < len(names) else str(c)
                print(f"Box {idx}: class={c} label={label} conf={conf:.3f} bbox={xy}")
        else:
            print("No boxes attribute on result object.")
    except Exception as e:
        print("Error parsing result:", e)
