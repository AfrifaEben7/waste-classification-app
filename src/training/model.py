"""
Model architecture for waste classification (EfficientNet & MobileNetV3)
"""
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0, MobileNetV3Small, MobileNetV3Large
from . import config


def create_model(num_classes=config.NUM_CLASSES, img_size=config.IMG_SIZE, model_name=config.MODEL_NAME):
    """
    Create model with custom classification head
    
    Args:
        num_classes: Number of output classes
        img_size: Input image size
        model_name: Model architecture ('EfficientNetB0', 'MobileNetV3Small', 'MobileNetV3Large')
        
    Returns:
        Compiled Keras model
    """
    # Input layer
    inputs = layers.Input(shape=(img_size, img_size, 3))
    
    # Preprocessing
    x = layers.Rescaling(1./255)(inputs)
    
    # Load pretrained base model
    if model_name == 'MobileNetV3Small':
        base_model = MobileNetV3Small(
            include_top=False,
            weights='imagenet',
            input_tensor=x,
            pooling='avg',
            minimalistic=False
        )
    elif model_name == 'MobileNetV3Large':
        base_model = MobileNetV3Large(
            include_top=False,
            weights='imagenet',
            input_tensor=x,
            pooling='avg',
            minimalistic=False
        )
    else:  # Default to EfficientNetB0
        base_model = EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_tensor=x,
            pooling='avg'
        )
    
    # Freeze base model layers initially
    base_model.trainable = False
    
    # Add custom classification head
    x = base_model.output
    x = layers.Dropout(0.3)(x)
    
    # Adjust dense layer size based on model
    dense_size = 128 if 'MobileNet' in model_name else 256
    x = layers.Dense(dense_size, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model


def compile_model(model, learning_rate=config.LEARNING_RATE):
    """
    Compile the model with optimizer, loss, and metrics
    
    Args:
        model: Keras model
        learning_rate: Initial learning rate
        
    Returns:
        Compiled model
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')]
    )
    return model


def unfreeze_model(model, unfreeze_layers=50):
    """
    Unfreeze the top layers of the base model for fine-tuning
    
    Args:
        model: Keras model
        unfreeze_layers: Number of layers to unfreeze from the top
        
    Returns:
        Model with unfrozen layers
    """
    # Find the base model layer (EfficientNet or MobileNet)
    base_model = None
    for layer in model.layers:
        if hasattr(layer, 'layers') and len(layer.layers) > 10:  # Base models have many layers
            base_model = layer
            break
    
    if base_model is None:
        print("Warning: Could not find base model layer. Skipping fine-tuning.")
        return model
    
    # Unfreeze the top layers
    print(f"Unfreezing top {unfreeze_layers} layers of {base_model.name}")
    for layer in base_model.layers[-unfreeze_layers:]:
        layer.trainable = True
    
    return model