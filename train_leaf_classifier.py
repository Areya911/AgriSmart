# train_leaf_classifier.py  -- chunked full training (MobileNetV2)
import os, json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import CSVLogger
from sklearn.utils import class_weight
from tensorflow.keras.models import load_model

# === CONFIG ===
DATA_DIR = "leaf_dataset"
MODEL_OUT = "leaf_disease_mobilenet.h5"
BEST_MODEL = "best_" + MODEL_OUT
CLASS_MAP_JSON = "class_map.json"
IMG_SIZE = (128, 128)
BATCH_SIZE = 8
TOTAL_EPOCHS = 10   # total target epochs (we will run in chunks)
LR = 1e-4
PATIENCE = 5

# chunk config: number of batches per chunk-epoch
STEPS_PER_EPOCH_CHUNK = 500   # adjust: 2000*BATCH_SIZE ~ images processed per chunk-epoch
VAL_STEPS = 100                # validation batches per chunk
CHUNK_EPOCHS = 1               # run 1 "chunk-epoch" at a time, repeat as needed
# ============

train_dir = os.path.join(DATA_DIR, "train")
val_dir   = os.path.join(DATA_DIR, "val")

# data generators
train_aug = ImageDataGenerator(preprocessing_function=preprocess_input,
    rotation_range=20, width_shift_range=0.1, height_shift_range=0.1,
    shear_range=0.08, zoom_range=0.1, horizontal_flip=True, vertical_flip=True, fill_mode="reflect")
val_aug = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = train_aug.flow_from_directory(train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", color_mode='rgb', shuffle=True)
val_gen = val_aug.flow_from_directory(val_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", color_mode='rgb', shuffle=False)

print("generator image_shape:", train_gen.image_shape)
num_classes = len(train_gen.class_indices)
print("Detected classes:", train_gen.class_indices)

# class weights
y_train = train_gen.classes
cw = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = {int(i): float(cw[i]) for i in range(len(cw))}
print("Class weights:", class_weights)

# model
base = MobileNetV2(include_top=False, input_shape=(IMG_SIZE[0],IMG_SIZE[1],3), weights="imagenet", pooling='avg')
base.trainable = False
x = base.output
x = layers.Dropout(0.3)(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.25)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)
model = models.Model(inputs=base.input, outputs=outputs)

opt = tf.keras.optimizers.Adam(learning_rate=LR)
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

# callbacks
es = callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True, verbose=1)
mc = callbacks.ModelCheckpoint(BEST_MODEL, monitor="val_loss", save_best_only=True, verbose=1)
csv_logger = CSVLogger("training_log.csv", append=True)

# function to run a single chunk (call repeatedly)
def run_chunk(chunk_idx):
    print(f"=== Running chunk {chunk_idx} ===")
    history = model.fit(
        train_gen,
        steps_per_epoch=STEPS_PER_EPOCH_CHUNK,
        validation_data=val_gen,
        validation_steps=VAL_STEPS,
        epochs=CHUNK_EPOCHS,
        callbacks=[es, mc, csv_logger],
        class_weight=class_weights
    )
    return history

if __name__ == "__main__":
    
    # If a best checkpoint exists, load it (so we can resume)
    if os.path.exists(BEST_MODEL):
        print("Loading compatible weights from", BEST_MODEL, "(by_name=True, skip_mismatch=True)")
        try:
        # load_weights with by_name=True will only load layers with matching names/shapes
            model.load_weights(BEST_MODEL, by_name=True, skip_mismatch=True)
        except Exception as e:
            print("Warning: load_weights by_name failed:", e)
    # recompile after loading weights
        model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

    # Run chunks until TOTAL_EPOCHS reached (CHUNK_EPOCHS per chunk)
    chunks_to_run = (TOTAL_EPOCHS + CHUNK_EPOCHS - 1) // CHUNK_EPOCHS
    for i in range(chunks_to_run):
        hist = run_chunk(i+1)
        # optional: after some chunks, unfreeze part of the base for fine-tuning
        # you can add logic here to unfreeze after N chunks
    # Save final model
    model.save(MODEL_OUT)
    # save class_map
    class_map = {str(v): k for k, v in train_gen.class_indices.items()}
    with open(CLASS_MAP_JSON, "w", encoding="utf-8") as fh:
        json.dump(class_map, fh, ensure_ascii=False, indent=2)
    print("Training complete, saved:", MODEL_OUT)
