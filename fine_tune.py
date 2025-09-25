# fine_tune.py
import json
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

MODEL_IN = "leaf_disease_mobilenet.h5"   # your saved model
OUT = "leaf_disease_mobilenet_finetuned.h5"
CLASS_MAP = "class_map.json"
LR = 1e-5
UNFREEZE_LAST = 60   # number of layers at the end to unfreeze
EPOCHS = 8
BATCH_SIZE = 8

print("Loading model:", MODEL_IN)
model = load_model(MODEL_IN)

# unfreeze last N layers
total = len(model.layers)
for layer in model.layers:
    layer.trainable = False
for layer in model.layers[-UNFREEZE_LAST:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=LR), loss="categorical_crossentropy", metrics=["accuracy"])

mc = ModelCheckpoint("best_finetune.h5", monitor="val_loss", save_best_only=True, verbose=1)
es = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1)
rlr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1)

# recreate generators (same as your train script)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE=(128,128)
train_gen = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(
    "leaf_dataset/train", target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", shuffle=True
)
val_gen = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(
    "leaf_dataset/val", target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False
)

print("Starting fine-tune")
model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=[mc, es, rlr])
print("Saving final:", OUT)
model.save(OUT)
