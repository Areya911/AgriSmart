from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# 1. Data Preparation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # 80% train, 20% val
    rotation_range=15,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    'soil_images',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)
print("Class indices:", train_data.class_indices)

val_data = datagen.flow_from_directory(
    'soil_images',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# 2. Model Architecture
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(),

    Flatten(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dense(4, activation='softmax')  # 4 soil types
])

# 3. Compile
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# 4. Train
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20
)

# 5. Save the model
model.save('soil_classifier.h5')
print("✅ Model saved as soil_classifier.h5")

# 6. Plot accuracy/loss (optional)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Training Accuracy")
plt.legend()
plt.show()
