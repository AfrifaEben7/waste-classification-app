"""
Data loading and preprocessing for waste classification
"""
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from . import config


def create_data_generators(data_dir=config.DATA_DIR, 
                          img_size=config.IMG_SIZE,
                          batch_size=config.BATCH_SIZE,
                          validation_split=config.VALIDATION_SPLIT):
    """
    Create training and validation data generators with augmentation
    
    Args:
        data_dir: Path to data directory containing class subdirectories
        img_size: Target image size
        batch_size: Batch size for training
        validation_split: Fraction of data to use for validation
        
    Returns:
        train_generator, validation_generator
    """
    
    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=config.AUGMENTATION_CONFIG['rotation_range'],
        width_shift_range=config.AUGMENTATION_CONFIG['width_shift_range'],
        height_shift_range=config.AUGMENTATION_CONFIG['height_shift_range'],
        shear_range=config.AUGMENTATION_CONFIG['shear_range'],
        zoom_range=config.AUGMENTATION_CONFIG['zoom_range'],
        horizontal_flip=config.AUGMENTATION_CONFIG['horizontal_flip'],
        fill_mode=config.AUGMENTATION_CONFIG['fill_mode'],
        validation_split=validation_split
    )
    
    # Validation data generator (only rescaling)
    validation_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split
    )
    
    # Training data generator
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=config.RANDOM_SEED
    )
    
    # Validation data generator
    validation_generator = validation_datagen.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=config.RANDOM_SEED
    )
    
    return train_generator, validation_generator


def get_class_names(data_dir=config.DATA_DIR):
    """
    Get class names from data directory
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        List of class names
    """
    class_names = sorted([d for d in os.listdir(data_dir) 
                         if os.path.isdir(os.path.join(data_dir, d))])
    return class_names


def count_images(data_dir=config.DATA_DIR):
    """
    Count total images in dataset
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        Dictionary with class names and image counts
    """
    class_counts = {}
    total_images = 0
    
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            count = len([f for f in os.listdir(class_path) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            class_counts[class_name] = count
            total_images += count
    
    class_counts['total'] = total_images
    return class_counts