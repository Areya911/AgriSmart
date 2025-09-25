# dump_misclassified.py
import json, shutil, os, numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from pathlib import Path

MODEL = "best_finetune.h5"   # change if needed
TARGET = "diseased_leafspot" # class you want to inspect
OUTDIR = Path("misclassified_samples") / TARGET
OUTDIR.mkdir(parents=True, exist_ok=True)

print("Loading", MODEL)
model = load_model(MODEL)
with open("class_map.json","r",encoding="utf-8") as fh:
    mapping = json.load(fh)
idx2name = [mapping[str(i)] for i in range(len(mapping))]
target_idx = idx2name.index(TARGET)

test_gen = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(
    "leaf_dataset/test", target_size=(128,128), batch_size=32, class_mode="categorical", shuffle=False
)

preds = model.predict(test_gen, steps=len(test_gen), verbose=1)
y_pred = preds.argmax(axis=1)
y_true = test_gen.classes
filenames = test_gen.filenames

count = 0
for i,(y_p,y_t) in enumerate(zip(y_pred,y_true)):
    if y_p == target_idx and y_t != target_idx:
        src = Path("leaf_dataset/test")/filenames[i]
        dst = OUTDIR / f"{i}_{idx2name[y_t]}__pred_{idx2name[y_p]}_{src.name}"
        shutil.copy2(src, dst)
        count += 1
        if count >= 200: break

print(f"Saved {count} misclassified examples for predicted class '{TARGET}' into {OUTDIR}")
