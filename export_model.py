from tensorflow.keras.models import load_model
import tensorflow as tf

model = load_model("leaf_disease_mobilenet_finetuned.h5")

# Save native format
model.save("leaf_disease_mobilenet_finetuned.keras")

# Export TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite = converter.convert()
open("leaf_disease_mobilenet_finetuned.tflite","wb").write(tflite)
print("Exported both .keras and .tflite")
