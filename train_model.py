#!/usr/bin/env python3
"""
Training script for waste classification
Supports EfficientNetB0, MobileNetV3Small, and MobileNetV3Large
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from training import config
from training.train import train_model


def main():
    parser = argparse.ArgumentParser(description='Train waste classification model')
    parser.add_argument(
        '--model',
        type=str,
        default='MobileNetV3Small',
        choices=['EfficientNetB0', 'MobileNetV3Small', 'MobileNetV3Large'],
        help='Model architecture to train (default: MobileNetV3Small for Mac)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=30,
        help='Number of epochs to train (default: 30)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size (default: 32)'
    )
    parser.add_argument(
        '--no-finetune',
        action='store_true',
        help='Skip fine-tuning phase'
    )
    
    args = parser.parse_args()
    
    # Update config
    config.MODEL_NAME = args.model
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    
    # Adjust model save path
    config.MODEL_SAVE_PATH = str(
        Path(__file__).parent / 'models' / f'{args.model.lower()}_waste_classifier.keras'
    )
    
    print(f"Training Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Fine-tuning: {'Enabled' if not args.no_finetune else 'Disabled'}")
    print(f"  Save Path: {config.MODEL_SAVE_PATH}")
    print()
    
    # Train model
    train_model(fine_tune=not args.no_finetune, epochs=args.epochs)
    
    print("\n✅ Training complete!")
    print(f"Model saved to: {config.MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
