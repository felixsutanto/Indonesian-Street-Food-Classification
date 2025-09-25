# src/model_training.py
import os
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt

# Correctly import from sibling module
from .data_preprocessing import DataPreprocessor

# --- Configuration (relative to project root) ---
RAW_DATA_DIR = "data/raw_images"
PROCESSED_DATA_DIR = "data/processed"
MODEL_SAVE_PATH = "models/indonesian_food_cnn.h5"
HISTORY_PLOT_PATH = "outputs/training_history.png"

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 50

def build_model(num_classes):
    """Builds the CNN architecture."""
    model = models.Sequential([
        layers.InputLayer(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def plot_history(history):
    """Saves a plot of training and validation accuracy/loss."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    os.makedirs(os.path.dirname(HISTORY_PLOT_PATH), exist_ok=True)
    plt.savefig(HISTORY_PLOT_PATH)
    print(f"Training history plot saved to {HISTORY_PLOT_PATH}")

def main():
    """Main training pipeline."""
    # 1. Preprocess Data
    preprocessor = DataPreprocessor(
        raw_data_dir=RAW_DATA_DIR,
        processed_data_dir=PROCESSED_DATA_DIR,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        batch_size=BATCH_SIZE
    )
    preprocessor.split_data()
    train_gen, val_gen = preprocessor.create_data_generators()

    # 2. Build Model
    num_classes = len(train_gen.class_indices)
    model = build_model(num_classes)
    model.summary()
    
    # 3. Train Model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model_callbacks = [
        callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
        callbacks.ModelCheckpoint(filepath=MODEL_SAVE_PATH, save_best_only=True, monitor='val_accuracy')
    ]
    
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=model_callbacks
    )

    # 4. Save results
    plot_history(history)
    print(f"Best model saved to {MODEL_SAVE_PATH}")
    print("Training pipeline completed!")

if __name__ == "__main__":
    main()
