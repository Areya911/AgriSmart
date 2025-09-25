# quick_debug_fit.py  (removed workers/use_multiprocessing)
import os, json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight

DATA_DIR = "leaf_dataset"
IMG_SIZE = (128,128)
BATCH_SIZE = 8

train_gen = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(
    os.path.join(DATA_DIR,"train"), target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", color_mode='rgb', shuffle=True
)
val_gen = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(
    os.path.join(DATA_DIR,"val"), target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", color_mode='rgb', shuffle=False
)

print("generator image_shape:", train_gen.image_shape)
num_classes = len(train_gen.class_indices)
y_train = train_gen.classes
cw = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = {int(i): float(cw[i]) for i in range(len(cw))}
print("Classes:", train_gen.class_indices)
print("Class weights:", class_weights)

# Use pooling='avg' so base.output is already (None, features)
base = MobileNetV2(include_top=False, input_shape=(IMG_SIZE[0],IMG_SIZE[1],3), weights="imagenet", pooling='avg')
base.trainable = False

x = base.output                       # already pooled -> shape (None, 1280)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='relu')(x)
out = layers.Dense(num_classes, activation='softmax')(x)
model = models.Model(base.input, out)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

es = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
mc = callbacks.ModelCheckpoint("best_quick_debug.h5", monitor='val_loss', save_best_only=True)

# Quick, tiny run (no workers arg)
model.fit(train_gen, validation_data=val_gen,
          steps_per_epoch=200, validation_steps=50,
          epochs=3, callbacks=[es,mc], class_weight=class_weights)
print("Quick debug finished.")
