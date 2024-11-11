import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Parameters
DATA_DIR = "Dataset/leapGestRecog"
IMG_SIZE = 64  # Reduced image size
BATCH_SIZE = 32
EPOCHS = 25

# Data Augmentation & Data Generators
def get_data_generators(data_dir, img_size, batch_size=32):
    datagen = ImageDataGenerator(
        rescale=1./255,  # Normalize images
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    train_gen = datagen.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    return train_gen

train_gen = get_data_generators(DATA_DIR, IMG_SIZE, BATCH_SIZE)

# Model Design with Transfer Learning (MobileNetV2)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(train_gen.class_indices), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    steps_per_epoch=train_gen.samples // BATCH_SIZE,
    validation_data=train_gen,
    validation_steps=train_gen.samples // BATCH_SIZE,
    callbacks=[early_stopping]
)

# Evaluate the model
test_loss, test_acc = model.evaluate(train_gen, verbose=2)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Visualize Training Progress
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Save the model
model.save("hand_gesture_model.h5")

print(f"Number of images loaded: {train_gen.samples}")
print(f"Number of unique labels: {len(train_gen.class_indices)}")
