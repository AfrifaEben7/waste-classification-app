"""
Simple Gradio interface for waste classification
"""
import gradio as gr
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from inference.predictor import WasteClassifier

# Initialize model
MODEL_PATH = Path(__file__).parent / 'models' / 'efficientnet_b4_final.onnx'
predictor = WasteClassifier(str(MODEL_PATH))

def classify_waste(image):
    """Classify waste image and return results"""
    if image is None:
        return "Please upload an image"
    
    # Predict (returns dict with 'top_predictions' key)
    result = predictor.predict(image)
    
    # Format results as dictionary for Gradio Label component
    predictions_dict = {}
    for pred in result['top_predictions']:
        predictions_dict[pred['class']] = float(pred['confidence'])
    
    return predictions_dict

# Create Gradio interface
demo = gr.Interface(
    fn=classify_waste,
    inputs=gr.Image(type="pil", label="Upload Waste Image"),
    outputs=gr.Label(num_top_classes=3, label="Predictions"),
    title="♻️ Waste Classification System",
    description="Upload an image of waste to classify it into: Cardboard, General-Waste, Glass, Metals, Paper, or Plastic",
    examples=None  # Remove examples for now since data folder is outside app directory
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
