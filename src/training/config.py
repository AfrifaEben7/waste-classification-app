"""
Configuration file for waste classification training
"""
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Data is in the parent directory of waste-classification-app
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Model Configuration
MODEL_NAME = 'EfficientNetB0'  # Options: 'EfficientNetB0', 'MobileNetV3Small', 'MobileNetV3Large'
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Classes
CLASSES = ['Cardboard', 'General-Waste', 'Glass', 'Metals', 'Paper', 'Plastic']
NUM_CLASSES = len(CLASSES)

# Training Configuration
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

# Data Augmentation
AUGMENTATION_CONFIG = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'shear_range': 0.2,
    'zoom_range': 0.2,
    'horizontal_flip': True,
    'fill_mode': 'nearest'
}

# Early Stopping
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5

# Model Saving
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'best_waste_classifier.keras')