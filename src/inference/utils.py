"""
Utility functions for inference
"""
import numpy as np
from PIL import Image


def preprocess_image_onnx(image_path, target_size=(224, 224)):
    """
    Load and preprocess an image for ONNX model inference
    
    Args:
        image_path: Path to the image file or PIL Image object
        target_size: Target size for the image (height, width)
        
    Returns:
        Preprocessed image numpy array ready for ONNX model input (NCHW format)
    """
    # Load image
    if isinstance(image_path, str):
        img = Image.open(image_path)
    else:
        img = image_path
    
    # Convert to RGB if necessary
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convert to numpy array and normalize
    img_array = np.array(img).astype(np.float32) / 255.0
    
    # Apply ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std
    
    # Convert from HWC to CHW format (ONNX expects channels first)
    img_array = np.transpose(img_array, (2, 0, 1))
    
    # Add batch dimension (NCHW format)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def preprocess_image(image_path, target_size=(224, 224)):
    """
    Load and preprocess an image for model inference (generic version)
    
    Args:
        image_path: Path to the image file or PIL Image object
        target_size: Target size for the image (height, width)
        
    Returns:
        Preprocessed image array ready for model input
    """
    # Load image
    if isinstance(image_path, str):
        img = Image.open(image_path)
    else:
        img = image_path
    
    # Convert to RGB if necessary
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize image
    img = img.resize(target_size, Image.LANCZOS)
    
    # Convert to array
    img_array = np.array(img)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def decode_predictions(predictions, class_names, top_k=3):
    """
    Decode model predictions into human-readable format
    
    Args:
        predictions: Model prediction array
        class_names: List of class names
        top_k: Number of top predictions to return
        
    Returns:
        Dictionary with decoded predictions
    """
    # Get top k predictions
    top_indices = np.argsort(predictions)[::-1][:top_k]
    
    results = {
        'predictions': [],
        'top_class': class_names[top_indices[0]],
        'top_confidence': float(predictions[top_indices[0]])
    }
    
    for idx in top_indices:
        results['predictions'].append({
            'class': class_names[idx],
            'confidence': float(predictions[idx]),
            'percentage': f"{predictions[idx] * 100:.2f}%"
        })
    
    return results


def get_class_color(class_name):
    """
    Get color code for each waste class for visualization
    
    Args:
        class_name: Name of the waste class
        
    Returns:
        Hex color code
    """
    colors = {
        'Cardboard': '#8B4513',  # Brown
        'General-Waste': '#808080',  # Gray
        'Glass': '#00CED1',  # Turquoise
        'Metals': '#C0C0C0',  # Silver
        'Paper': '#4169E1',  # Blue
        'Plastic': '#FFD700'  # Gold
    }
    return colors.get(class_name, '#000000')


def format_confidence(confidence):
    """
    Format confidence score as percentage
    
    Args:
        confidence: Confidence score (0-1)
        
    Returns:
        Formatted percentage string
    """
    return f"{confidence * 100:.2f}%"