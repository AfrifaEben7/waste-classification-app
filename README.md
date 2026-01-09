# Waste Classification System 🗑️♻️

An intelligent waste classification system powered by EfficientNet deep learning model. This application helps classify waste items into six categories: Cardboard, General Waste, Glass, Metals, Paper, and Plastic.

## Features

- 🤖 **EfficientNet-based Classification**: State-of-the-art deep learning model for accurate waste classification
- 🌐 **Web Interface**: User-friendly web application for image upload and prediction
- 📊 **Multiple Predictions**: Shows top-3 predictions with confidence scores
- 💡 **Disposal Information**: Provides recycling guidelines for each waste category
- 📱 **Responsive Design**: Works on desktop and mobile devices

## Project Structure

```
waste-classification-app/
├── src/
│   ├── training/           # Training scripts
│   │   ├── config.py      # Configuration parameters
│   │   ├── model.py       # EfficientNet model architecture
│   │   ├── data_loader.py # Data loading and preprocessing
│   │   └── train.py       # Training script
│   ├── inference/          # Prediction module
│   │   ├── predictor.py   # Predictor class
│   │   └── utils.py       # Utility functions
│   └── web/               # Web application
│       ├── app.py         # Flask application
│       ├── static/        # CSS and JavaScript
│       └── templates/     # HTML templates
├── data/                  # Training data (6 categories)
├── models/               # Saved models
├── notebooks/            # Jupyter notebooks for exploration
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Installation

1. **Clone the repository**
```bash
cd waste-classification-app
```

2. **Create a conda environment** (recommended)

**Option A: From environment.yml (easiest)**
```bash
conda env create -f environment.yml
conda activate waste-classifier
```

**Option B: Manual setup**
```bash
# Create new conda environment with Python 3.10
conda create -n waste-classifier python=3.10 -y
conda activate waste-classifier
pip install -r requirements.txt
```

*Alternative: Using venv*
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

*Note: Conda handles TensorFlow and CUDA dependencies better, making it the preferred option for ML projects.*

## Usage

### Training the Model

1. **Prepare your data**: Ensure the `data/` directory contains subdirectories for each class with images:
```
data/
├── Cardboard/
├── General-Waste/
├── Glass/
├── Metals/
├── Paper/
└── Plastic/
```

2. **Train the model**:
```bash
python -m src.training.train
```

Optional arguments:
- `--epochs N`: Number of training epochs (default: 50)
- `--no-fine-tune`: Skip fine-tuning phase

The trained model will be saved to `models/best_waste_classifier.keras`

### Running the Web Application

1. **Start the Flask server**:
```bash
python -m src.web.app
```

2. **Open your browser** and navigate to:
```
http://localhost:5000
```

3. **Upload an image** and get instant classification results!

## Model Architecture

- **Base Model**: EfficientNetB0 (pre-trained on ImageNet)
- **Input Size**: 224x224x3
- **Custom Head**:
  - Global Average Pooling
  - Dropout (0.3)
  - Dense Layer (256 units, ReLU)
  - Batch Normalization
  - Dropout (0.4)
  - Output Layer (6 units, Softmax)

## Training Strategy

1. **Phase 1**: Train with frozen EfficientNet base (transfer learning)
2. **Phase 2**: Fine-tune top 50 layers with reduced learning rate

## Data Augmentation

- Rotation: ±20°
- Width/Height Shift: 20%
- Shear: 20%
- Zoom: 20%
- Horizontal Flip
- Random adjustments during training

## Performance Metrics

The model is evaluated on:
- **Accuracy**: Overall classification accuracy
- **Top-2 Accuracy**: Correct class in top 2 predictions
- **Confusion Matrix**: Per-class performance

## API Usage

You can also use the predictor programmatically:

```python
from src.inference.predictor import WasteClassifier

# Initialize predictor
classifier = WasteClassifier('models/best_waste_classifier.keras')

# Make prediction
results = classifier.predict('path/to/image.jpg')

# Get top prediction
class_name, confidence = classifier.get_top_prediction('path/to/image.jpg')
print(f"Predicted: {class_name} ({confidence:.2%})")
```

## Requirements

- Python 3.8+
- TensorFlow 2.13+
- Flask 2.3+
- See `requirements.txt` for complete list

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

**Model not found error**:
- Make sure to train the model first: `python -m src.training.train`
- Check that `models/best_waste_classifier.keras` exists

**Memory errors during training**:
- Reduce batch size in `src/training/config.py`
- Use a smaller image size

**Web app not starting**:
- Check that port 5000 is not in use
- Verify Flask is installed: `pip install Flask`

## Future Enhancements

- [ ] Mobile app version
- [ ] Real-time video classification
- [ ] Multi-object detection
- [ ] Deployment to cloud (AWS/Azure/GCP)
- [ ] REST API for integration

## License

This project is for educational purposes.

## Acknowledgments

- EfficientNet architecture by Google Research
- Dataset provided for waste classification research
- Flask web framework

## Contact

For questions or issues, please open an issue on the repository.

---

**Happy Recycling! ♻️**