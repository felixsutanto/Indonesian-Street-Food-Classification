# src/data_preprocessing.py
import os
import shutil
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DataPreprocessor:
    def __init__(self, raw_data_dir, processed_data_dir, img_height, img_width, batch_size, train_ratio=0.8):
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.train_dir = os.path.join(processed_data_dir, 'train')
        self.validation_dir = os.path.join(processed_data_dir, 'validation')
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.categories = [d for d in os.listdir(raw_data_dir) if os.path.isdir(os.path.join(raw_data_dir, d))]

    def split_data(self):
        """Splits raw images into train and validation sets."""
        if os.path.exists(self.processed_data_dir):
            print("Processed data directory already exists. Skipping split.")
            return

        print("Creating train/validation split...")
        for category in self.categories:
            print(f"  Processing category: {category}")
            os.makedirs(os.path.join(self.train_dir, category), exist_ok=True)
            os.makedirs(os.path.join(self.validation_dir, category), exist_ok=True)
            
            source_dir = os.path.join(self.raw_data_dir, category)
            images = [f for f in os.listdir(source_dir) if f.lower().endswith('.jpg')]
            random.shuffle(images)
            
            train_count = int(len(images) * self.train_ratio)
            train_images = images[:train_count]
            val_images = images[train_count:]
            
            for img in train_images:
                shutil.copy(os.path.join(source_dir, img), os.path.join(self.train_dir, category, img))
            for img in val_images:
                shutil.copy(os.path.join(source_dir, img), os.path.join(self.validation_dir, category, img))
            print(f"    - {len(train_images)} train, {len(val_images)} validation images.")

    def create_data_generators(self):
        """Creates data generators with augmentation for training and validation."""
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        validation_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )
        validation_generator = validation_datagen.flow_from_directory(
            self.validation_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        print(f"Found class indices: {train_generator.class_indices}")
        return train_generator, validation_generator
