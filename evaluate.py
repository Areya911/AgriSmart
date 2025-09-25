# evaluate.py
import json, numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix

MODEL = "best_leaf_disease_mobilenet.h5"
CLASS_MAP = "class_map.json"
TEST_DIR = "leaf_dataset/test"
IMG_SIZE = (128,128)
BATCH_SIZE = 32

print("Loading model:", MODEL)
model = load_model(MODEL)

with open(CLASS_MAP, 'r', encoding='utf-8') as fh:
    mapping = json.load(fh)
    idx2name = [mapping[str(i)] for i in range(len(mapping))]

test_gen = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(
    TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False, color_mode='rgb'
)

print("Predicting on test set...")
preds = model.predict(test_gen, steps=len(test_gen), verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = test_gen.classes

print("\nClassification report:\n")
print(classification_report(y_true, y_pred, target_names=idx2name, digits=4))

cm = confusion_matrix(y_true, y_pred)
print("\nConfusion matrix shape:", cm.shape)
