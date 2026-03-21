"""
Real-time waste classification with webcam streaming
"""
import gradio as gr
from pathlib import Path
import sys
import cv2
import numpy as np
from PIL import Image
import threading

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from inference.predictor import WasteClassifier

# Initialize model
MODEL_PATH = Path(__file__).parent / 'models' / 'efficientnet_b4_final.onnx'
predictor = WasteClassifier(str(MODEL_PATH))

# Global variables
webcam_active = False
lock = threading.Lock()

def classify_waste(image):
    """Classify entire image"""
    if image is None:
        return "Please upload an image"
    
    result = predictor.predict(image)
    predictions_dict = {}
    for pred in result['top_predictions']:
        predictions_dict[pred['class']] = float(pred['confidence'])
    
    return predictions_dict

def webcam_stream_realtime():
    """Real-time webcam classification"""
    global webcam_active
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        yield None, "Error: Cannot open webcam"
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_count = 0
    process_every_n_frames = 3  # Process every 3rd frame for speed
    last_prediction = "Initializing..."
    
    while webcam_active:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Convert BGR to RGB for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process every nth frame to save resources
        if frame_count % process_every_n_frames == 0:
            try:
                # Convert to PIL Image for classification
                pil_image = Image.fromarray(frame_rgb)
                
                # Classify
                result = predictor.predict(pil_image)
                
                # Format results
                predictions_dict = {}
                for pred in result['top_predictions']:
                    predictions_dict[pred['class']] = float(pred['confidence'])
                
                with lock:
                    last_prediction = predictions_dict
                    
            except Exception as e:
                print(f"Error during prediction: {e}")
                last_prediction = f"Error: {str(e)}"
        
        frame_count += 1
        
        # Yield frame and prediction
        yield frame_rgb, last_prediction if last_prediction else "Processing..."
    
    cap.release()

def start_webcam():
    """Start webcam stream"""
    global webcam_active
    webcam_active = True
    return "Webcam started - detecting..."

def stop_webcam():
    """Stop webcam stream"""
    global webcam_active
    webcam_active = False
    return "Webcam stopped"

# Create Gradio Blocks interface
with gr.Blocks(title="Waste Classification System") as demo:
    gr.Markdown("# Waste Classification System")
    gr.Markdown("Classify waste in real-time using your webcam or upload an image")
    
    with gr.Tabs():
        # Real-time Webcam Tab
        with gr.Tab("Real-Time Webcam"):
            gr.Markdown("Click 'Start Webcam' to begin live classification")
            
            with gr.Row():
                with gr.Column():
                    webcam_output = gr.Image(label="Webcam Feed", type="numpy")
                with gr.Column():
                    prediction_output = gr.Label(num_top_classes=3, label="Live Predictions")
            
            with gr.Row():
                start_btn = gr.Button("Start Webcam", variant="primary", size="lg")
                stop_btn = gr.Button("Stop Webcam", variant="stop", size="lg")
            
            status = gr.Textbox(label="Status", value="Ready", interactive=False)
            
            start_btn.click(
                fn=start_webcam,
                outputs=status
            ).then(
                fn=webcam_stream_realtime,
                outputs=[webcam_output, prediction_output]
            )
            
            stop_btn.click(
                fn=stop_webcam,
                outputs=status
            )
        
        # Upload Image Tab
        with gr.Tab("Upload Image"):
            gr.Markdown("Upload a single image for classification")
            
            with gr.Row():
                image_input = gr.Image(type="pil", label="Upload Waste Image")
            
            image_pred_output = gr.Label(num_top_classes=3, label="Predictions")
            
            # Auto-classify when image is uploaded
            image_input.change(
                fn=classify_waste,
                inputs=image_input,
                outputs=image_pred_output
            )

if __name__ == "__main__":
    print("Starting Waste Classification System...")
    print("Open http://localhost:7860 in your browser")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)