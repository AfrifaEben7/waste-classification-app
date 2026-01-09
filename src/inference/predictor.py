"""
Predictor class for waste classification inference
"""
import os
import numpy as np
import onnxruntime as ort
from PIL import Image
from .utils import preprocess_image_onnx, decode_predictions


class WasteClassifier:
    """Waste Classification Predictor using ONNX Runtime"""
    
    def __init__(self, model_path, model_arch='efficientnet_b4'):
        """
        Initialize the predictor with an ONNX model
        
        Args:
            model_path: Path to the .onnx model file
            model_arch: Model architecture (efficientnet_b0, efficientnet_b1, etc.)
        """
        self.model_path = model_path
        self.model_arch = model_arch
        self.session = None
        self.classes = ['Cardboard', 'General-Waste', 'Glass', 'Metals', 'Paper', 'Plastic']
        self.img_size = 224
        
        # Set image size based on architecture
        if 'b4' in model_arch.lower():
            self.img_size = 380
        elif 'b5' in model_arch.lower():
            self.img_size = 456
        elif 'b6' in model_arch.lower():
            self.img_size = 528
        elif 'b7' in model_arch.lower():
            self.img_size = 600
        elif 'b3' in model_arch.lower():
            self.img_size = 300
        elif 'b2' in model_arch.lower():
            self.img_size = 260
        
        # Load model
        self.load_model()
    
    def load_model(self):
        """Load the ONNX model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        print(f"Loading ONNX model from {self.model_path}...")
        
        # Create ONNX Runtime session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Use CPU execution provider (works reliably on macOS ARM)
        providers = ['CPUExecutionProvider']
        
        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=sess_options,
            providers=providers
        )
        
        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        print(f"Model loaded successfully!")
        print(f"Input name: {self.input_name}")
        print(f"Output name: {self.output_name}")
        print(f"Using image size: {self.img_size}x{self.img_size}")
    
    def predict(self, image_path):
        """
        Predict the waste class for a single image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess image
        img_array = preprocess_image_onnx(image_path, target_size=(self.img_size, self.img_size))
        
        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: img_array})
        predictions = outputs[0][0]  # Get first batch result
        
        # Apply softmax to get probabilities
        exp_preds = np.exp(predictions - np.max(predictions))
        probabilities = exp_preds / np.sum(exp_preds)
        
        # Get top predictions
        top_indices = np.argsort(probabilities)[::-1]
        
        results = {
            'predicted_class': self.classes[top_indices[0]],
            'confidence': float(probabilities[top_indices[0]]),
            'top_predictions': [
                {
                    'class': self.classes[idx],
                    'confidence': float(probabilities[idx])
                }
                for idx in top_indices[:3]
            ]
        }
        
        return results
    
    def predict_batch(self, image_paths):
        """
        Predict waste classes for multiple images
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.predict(image_path)
                results.append(result)
            except Exception as e:
                results.append({
                    'error': str(e),
                    'image_path': image_path
                })
        
        return results
    
    def get_top_prediction(self, image_path):
        """
        Get only the top prediction for an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (class_name, confidence)
        """
        results = self.predict(image_path)
        top_result = results['predictions'][0]
        return top_result['class'], top_result['confidence']