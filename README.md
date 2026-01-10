# Waste Classification System

An intelligent waste classification system powered by deep learning models (EfficientNet & MobileNetV3). This application helps classify waste items into six categories: Cardboard, General Waste, Glass, Metals, Paper, and Plastic.

## Features

- **Multiple Model Support**: Choose between EfficientNetB0, MobileNetV3Small, or MobileNetV3Large
- **Gradio Web Interface**: Simple, interactive web UI with drag-and-drop image upload
- **Top-3 Predictions**: Shows confidence scores for top predictions
- **ONNX Runtime**: Fast inference optimized for macOS ARM (Apple Silicon)
- **Educational**: Includes kid-friendly Jupyter notebook for teaching AI concepts
- **Responsive Design**: Works on desktop, tablet, and mobile devices

## Project Structure

```
waste-classification-app/
├── src/
│   ├── training/              # Model training
│   │   ├── config.py         # Training configuration
│   │   ├── model.py          # Model architectures (EfficientNet, MobileNet)
│   │   ├── data_loader.py    # Data loading and augmentation
│   │   └── train.py          # Training orchestration
│   ├── inference/             # Prediction module
│   │   ├── predictor.py      # ONNX Runtime predictor
│   │   └── utils.py          # Image preprocessing utilities
│   └── web/                   # Flask web app (legacy)
│       ├── app.py            # Flask application
│       ├── static/           # CSS and JavaScript
│       └── templates/        # HTML templates
├── models/                    # Saved models
│   ├── efficientnet_b4_final.onnx       # Pre-trained EfficientNet-B4
│   └── mobilenetv3small_waste_classifier.keras  # Trained MobileNetV3
├── data/                      # Training data (6 categories)
├── logs/                      # TensorBoard training logs
├── notebooks/                 # Jupyter notebooks
├── gradio_app.py             # Main Gradio web interface
├── train_model.py            # Easy training script
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Installation

### Prerequisites
- Python 3.10+
- macOS ARM (Apple Silicon) or other platforms
- 8GB+ RAM recommended

### 1. Clone the repository
```bash
cd waste-classification-app
```

### 2. Create conda environment (Recommended for macOS)

```bash
conda create -n waste-classifier python=3.10 -y
conda activate waste-classifier
```

### 3. Install dependencies

**For macOS ARM (Apple Silicon):**
```bash
# Install TensorFlow with Metal GPU acceleration
pip install tensorflow-macos tensorflow-metal

# Install other dependencies
pip install gradio onnxruntime pillow numpy matplotlib scipy

# For training
conda install scipy -y  # Use conda for better ARM compatibility
```

**For other platforms:**
```bash
pip install -r requirements.txt
```

## Quick Start

### Option 1: Use Pre-trained Model (Fastest!)

Run the Gradio web interface with the pre-trained EfficientNet-B4 model:

```bash
conda activate waste-classifier
python gradio_app.py
```

Then open your browser to: **http://127.0.0.1:7860**

That's it! Drag and drop images to classify waste!

### Option 2: Train Your Own Model

Train a custom model on your data:

```bash
# Train MobileNetV3Small (fast, good for Mac)
python train_model.py --model MobileNetV3Small --epochs 30

# Or train EfficientNetB0 (slower, more accurate)
python train_model.py --model EfficientNetB0 --epochs 40

# Or train MobileNetV3Large (balance of speed and accuracy)
python train_model.py --model MobileNetV3Large --epochs 35
```

**Training Options:**
- `--model`: Choose architecture (EfficientNetB0, MobileNetV3Small, MobileNetV3Large)
- `--epochs`: Number of training epochs (default: 30)
- `--batch-size`: Batch size (default: 32)
- `--no-finetune`: Skip fine-tuning phase

The model will be saved to `models/{model_name}_waste_classifier.keras`

### Running the Web Application

**Gradio Interface (Recommended):**
```bash
python gradio_app.py
```
- Automatically opens at http://127.0.0.1:7860
- Drag-and-drop interface
- Real-time predictions with confidence scores
- Mobile-friendly

**Flask Interface (Legacy):**
```bash
python src/web/app.py
```
- Available at http://127.0.0.1:8080
- Traditional web form interface

## Dataset Structure

Your training data should be organized as follows:

```
data/
├── Cardboard/       # Cardboard waste images
├── General-Waste/   # Non-recyclable waste
├── Glass/           # Glass bottles, jars
├── Metals/          # Aluminum cans, metal items
├── Paper/           # Paper, documents, newspapers
└── Plastic/         # Plastic bottles, containers
```

Each folder should contain at least 100+ images for effective training.

## Model Architectures

### Available Models

1. **EfficientNet-B4** (Pre-trained, ONNX)
   - Input: 380×380
   - Parameters: 4.4M
   - Accuracy: ~92% on test set
   - Best for: Production use
   - Format: ONNX (optimized for inference)

2. **MobileNetV3Small** (Trainable)
   - Input: 224×224
   - Parameters: ~2.5M
   - Speed: Very Fast
   - Best for: Mac training, mobile deployment

3. **MobileNetV3Large** (Trainable)
   - Input: 224×224
   - Parameters: ~5M
   - Speed: Fast
   - Best for: Balance of speed and accuracy

4. **EfficientNetB0** (Trainable)
   - Input: 224×224
   - Parameters: 5M+
   - Speed: Moderate
   - Best for: High accuracy requirements

### Model Components

All models include:
- **Preprocessing**: Image normalization (ImageNet mean/std)
- **Base Model**: Pre-trained on ImageNet (transfer learning)
- **Custom Head**:
  - Dropout layers (prevent overfitting)
  - Dense layers with BatchNormalization
  - Softmax output (6 classes)

## Educational Materials

### Interactive Jupyter Notebook

Located at `../hesam_data.ipynb` - A kid-friendly guide to training AI!

**Features:**
- Step-by-step tutorial with visual examples
- Builds a simple CNN from scratch
- Shows training progress with graphs
- Interactive prediction function
- Challenges for experimentation

**Perfect for:**
- Students learning AI/ML
- Teaching computer vision concepts
- Understanding neural networks
- Hands-on coding practice

**To use:**
```bash
jupyter notebook ../hesam_data.ipynb
```

## Training Strategy

**Two-Phase Training:**

1. **Phase 1 - Transfer Learning** (First 50% of epochs)
   - Freeze base model layers
   - Train only custom head
   - Fast convergence with ImageNet features

2. **Phase 2 - Fine-Tuning** (Last 50% of epochs)
   - Unfreeze top 50 layers of base model
   - Lower learning rate (0.0001)
   - Refine features for waste classification

**Optimization:**
- Optimizer: Adam
- Loss: Categorical Cross-Entropy
- Metrics: Accuracy, Top-2 Accuracy
- Callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

## Data Augmentation

- Rotation: ±20°
- Width/Height Shift: 20%
- Shear: 20%
- Zoom: 20%
- Horizontal Flip
- Random adjustments during training

## Performance

**EfficientNet-B4 (Pre-trained):**
- Overall Accuracy: **92.7%**
- Test Results:
  - Cardboard: 91.9%
  - Plastic: 92.6%
  - Glass: 92.7%
  - Metals: High 80s-90s%
  - Paper: High 80s-90s%
  - General-Waste: High 80s-90s%

**Training Metrics:**
- Model converges in 20-30 epochs
- Validation accuracy typically reaches 85-95%
- Top-2 accuracy: 95%+

**Performance on macOS ARM:**
- Training: ~70-100s per epoch (MobileNetV3Small, batch=32)
- Inference: <100ms per image (ONNX Runtime)
- GPU Acceleration: TensorFlow Metal (automatic)

## API Usage

### Using the ONNX Predictor

```python
from src.inference.predictor import WasteClassifier

# Initialize with ONNX model
classifier = WasteClassifier('models/efficientnet_b4_final.onnx')

# Predict from file path
results = classifier.predict('path/to/image.jpg')
print(f"Class: {results['predicted_class']}")
print(f"Confidence: {results['confidence']:.2%}")

# Get all top-3 predictions
for pred in results['top_predictions']:
    print(f"{pred['class']}: {pred['confidence']:.2%}")
```

### Using Keras Model (After Training)

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image

# Load model
model = keras.models.load_model('models/mobilenetv3small_waste_classifier.keras')

# Prepare image
img = Image.open('image.jpg').resize((224, 224))
img_array = np.array(img).astype('float32') / 255.0
img_array = np.expand_dims(img_array, 0)

# Predict
predictions = model.predict(img_array)
class_names = ['Cardboard', 'General-Waste', 'Glass', 'Metals', 'Paper', 'Plastic']
predicted_class = class_names[np.argmax(predictions[0])]
confidence = np.max(predictions[0])

print(f"Predicted: {predicted_class} ({confidence:.2%})")
```

## Requirements

### Core Dependencies
- **Python**: 3.10+
- **TensorFlow**: 2.16+ (with tensorflow-macos for Apple Silicon)
- **ONNX Runtime**: 1.23+ (for inference)
- **Gradio**: 6.2+ (web interface)
- **NumPy**: 1.26.4 (ARM-compatible version)
- **Scipy**: Via conda (for ARM compatibility)

### macOS ARM (Apple Silicon) Notes
- Use conda for scipy installation: `conda install scipy`
- TensorFlow Metal provides GPU acceleration
- NumPy 1.26.4 required (2.x has ARM issues)

See `requirements.txt` for complete list.

## Development

### Running Tests
```bash
python -m pytest tests/
```

### TensorBoard
Monitor training progress:
```bash
tensorboard --logdir=logs/
```

## Troubleshooting

### macOS ARM (Apple Silicon) Issues

**"Symbol not found: _dstevr$NEWLAPACK" (scipy error):**
```bash
conda activate waste-classifier
pip uninstall scipy
conda install scipy -y
```

**NumPy compatibility issues:**
```bash
pip install numpy==1.26.4
```

**TensorFlow Metal not working:**
```bash
pip install tensorflow-macos tensorflow-metal
```

### Model Issues

**Model not found:**
- Download the pre-trained model or train your own
- Check that `models/efficientnet_b4_final.onnx` exists
- For ONNX models, ensure both `.onnx` and `.onnx.data` files are present

**Low accuracy during training:**
- Increase number of epochs: `--epochs 40`
- Ensure sufficient training data (100+ images per class)
- Try different model architecture
- Check data quality and labeling

### Web Application Issues

**Port already in use:**
```bash
# For Gradio (default: 7860)
python gradio_app.py  # Will auto-select available port

# For Flask (default: 8080)
# Change port in src/web/app.py
```

**Gradio interface not loading:**
- Check firewall settings
- Try accessing from different browser
- Verify all dependencies installed: `pip install gradio`

**Memory errors:**
- Reduce batch size in `config.py`
- Use smaller image size (128x128 instead of 224x224)
- Close other applications to free RAM

## Development

### Using TensorBoard

Monitor training progress in real-time:
```bash
tensorboard --logdir=logs/
# Open browser to http://localhost:6006
```

### Testing the Model

Test inference with Python:
```python
from src.inference.predictor import WasteClassifier

# Test ONNX model
predictor = WasteClassifier('models/efficientnet_b4_final.onnx')
result = predictor.predict('test_image.jpg')
print(f"Prediction: {result['predicted_class']} ({result['confidence']:.1%})")

# Test Keras model
predictor = WasteClassifier('models/mobilenetv3small_waste_classifier.keras')
result = predictor.predict('test_image.jpg')
```

### Running Tests

```bash
python -m pytest tests/
```

## Future Enhancements

- [ ] Add more waste categories (Electronics, Organic, etc.)
- [ ] Implement multi-label classification (items with mixed materials)
- [ ] Deploy to cloud platforms (Azure, AWS, GCP)
- [ ] Mobile app integration (iOS/Android)
- [ ] Real-time video classification
- [ ] Data augmentation techniques for better accuracy
- [ ] Model explainability (Grad-CAM visualizations)

## References

- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- [MobileNetV3 Paper](https://arxiv.org/abs/1905.02244)
- [ONNX Runtime Documentation](https://onnxruntime.ai/docs/)
- [TensorFlow Transfer Learning Guide](https://www.tensorflow.org/guide/keras/transfer_learning)

## License

This project is for educational purposes.

## Acknowledgments

- Pre-trained models from TensorFlow/Keras Applications
- Dataset structure compatible with standard image classification datasets
- Educational content designed for students and beginners

---

**Note:** This project is optimized for macOS ARM (Apple Silicon) but should work on other platforms with minimal modifications.

**Happy Recycling!**