"""
Training script for waste classification with EfficientNet
"""
import os
import argparse
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from datetime import datetime
from . import config
from .model import create_model, compile_model, unfreeze_model
from .data_loader import create_data_generators, count_images


def train_model(fine_tune=True, epochs=config.EPOCHS):
    """
    Train the waste classification model
    
    Args:
        fine_tune: Whether to fine-tune the base model
        epochs: Number of epochs to train
    """
    print("=" * 60)
    print(f"Waste Classification Training with {config.MODEL_NAME}")
    print("=" * 60)
    
    # Create model directory if it doesn't exist
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    # Display dataset statistics
    print("\nDataset Statistics:")
    image_counts = count_images()
    for class_name, count in image_counts.items():
        print(f"  {class_name}: {count} images")
    
    # Create data generators
    print("\nLoading data...")
    train_generator, validation_generator = create_data_generators()
    
    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {validation_generator.samples}")
    print(f"Classes: {list(train_generator.class_indices.keys())}")
    
    # Create model
    print(f"\nCreating {config.MODEL_NAME} model...")
    model = create_model()
    model = compile_model(model)
    
    print(f"\nModel Summary:")
    model.summary()
    
    # Callbacks
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(config.BASE_DIR, 'logs', timestamp)
    
    callbacks = [
        ModelCheckpoint(
            config.MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            verbose=1,
            min_lr=1e-7
        ),
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=1
        )
    ]
    
    # Phase 1: Train with frozen base model
    print("\n" + "=" * 60)
    print("Phase 1: Training with frozen base model")
    print("=" * 60)
    
    initial_epochs = epochs // 2 if fine_tune else epochs
    
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=initial_epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Phase 2: Fine-tuning
    if fine_tune:
        print("\n" + "=" * 60)
        print("Phase 2: Fine-tuning with unfrozen layers")
        print("=" * 60)
        
        # Unfreeze and recompile
        model = unfreeze_model(model, unfreeze_layers=50)
        model = compile_model(model, learning_rate=config.LEARNING_RATE / 10)
        
        print(f"\nUnfrozen top 50 layers for fine-tuning")
        
        # Continue training
        history_fine = model.fit(
            train_generator,
            validation_data=validation_generator,
            initial_epoch=history.epoch[-1] + 1,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
    
    # Evaluate final model
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    
    results = model.evaluate(validation_generator, verbose=1)
    print(f"\nValidation Loss: {results[0]:.4f}")
    print(f"Validation Accuracy: {results[1]:.4f}")
    print(f"Validation Top-2 Accuracy: {results[2]:.4f}")
    
    print(f"\nModel saved to: {config.MODEL_SAVE_PATH}")
    print(f"TensorBoard logs saved to: {log_dir}")
    print("\nTraining complete!")
    
    return model, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train waste classification model')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS, 
                       help='Number of epochs to train')
    parser.add_argument('--no-fine-tune', action='store_true',
                       help='Skip fine-tuning phase')
    
    args = parser.parse_args()
    
    # Train model
    train_model(fine_tune=not args.no_fine_tune, epochs=args.epochs)