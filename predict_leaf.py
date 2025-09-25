# predict_leaf.py
import argparse, json, sys
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = (128, 128)
MODEL = "leaf_disease_mobilenet_finetuned.h5"   # use your fine-tuned model
CLASS_MAP = "class_map.json"                    # saved during training

def load_class_map(path):
    with open(path, "r", encoding="utf-8") as fh:
        idx2name = json.load(fh)
    # ensure it's a list-like mapping idx->name
    return [idx2name[str(i)] for i in range(len(idx2name))]

def predict(img_path):
    # load model + class mapping
    model = load_model(MODEL)
    idx2name = load_class_map(CLASS_MAP)

    # load & preprocess image
    img = image.load_img(img_path, target_size=IMG_SIZE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # predict
    preds = model.predict(x)
    idx = int(preds.argmax())
    print(f"\nPrediction: {idx2name[idx]} (confidence={preds[0][idx]:.4f})")

    # show all probabilities
    print("\nAll class probabilities:")
    for i, name in enumerate(idx2name):
        print(f"{name:25s}: {preds[0][i]:.4f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Leaf disease prediction")
    p.add_argument("--image", required=True, help="path to leaf image")
    args = p.parse_args()
    predict(args.image)
