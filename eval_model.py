# eval_model.py
import argparse, json, numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix

def run_eval(model_path, test_dir="leaf_dataset/test", img_size=(128,128), batch_size=32):
    print("Loading model:", model_path)
    model = load_model(model_path)
    with open("class_map.json","r",encoding="utf-8") as fh:
        mapping = json.load(fh)
        idx2name = [mapping[str(i)] for i in range(len(mapping))]
    test_gen = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(
        test_dir, target_size=img_size, batch_size=batch_size, class_mode="categorical", shuffle=False, color_mode='rgb'
    )
    preds = model.predict(test_gen, steps=len(test_gen), verbose=1)
    y_pred = preds.argmax(axis=1)
    y_true = test_gen.classes
    print("\nClassification report:\n")
    print(classification_report(y_true, y_pred, target_names=idx2name, digits=4))
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion matrix shape:", cm.shape)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="path to .h5 or .keras model")
    args = p.parse_args()
    run_eval(args.model)
