"""
Real-time waste object detection with webcam streaming using YOLOv8
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

# Try to import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Install with: pip install ultralytics")

# Initialize models
MODEL_PATH = Path(__file__).parent / 'models' / 'efficientnet_b4_final.onnx'
predictor = WasteClassifier(str(MODEL_PATH))

# Load YOLO model if available
if YOLO_AVAILABLE:
    try:
        # Using YOLOv8 nano for speed
        yolo_model = YOLO('yolov8n.pt')
        print("YOLOv8n model loaded successfully")
    except:
        print("Could not load YOLOv8 model. Download will happen on first run.")
        yolo_model = YOLO('yolov8n.pt')
else:
    yolo_model = None

# Waste categories for custom YOLO
WASTE_CLASSES = ['Cardboard', 'General-Waste', 'Glass', 'Metals', 'Paper', 'Plastic']

# Global variables
webcam_active = False
current_prediction = None
lock = threading.Lock()
detection_mode = "yolo"  # or "classify"

def classify_waste(image):
    """Classify entire image"""
    if image is None:
        return "Please upload an image"
    
    result = predictor.predict(image)
    predictions_dict = {}
    for pred in result['top_predictions']:
        predictions_dict[pred['class']] = float(pred['confidence'])
    
    return predictions_dict

def detect_waste_frame(frame):
    """Detect waste objects in frame using YOLO"""
    if not YOLO_AVAILABLE or yolo_model is None:
        return frame, "YOLO not available"
    
    try:
        # Run YOLO detection
        results = yolo_model(frame, conf=0.4, verbose=False)
        
        # Get detections
        detections = results[0]
        
        # Draw boxes and labels
        annotated_frame = detections.plot()
        
        # Extract detection info
        detection_info = ""
        if len(detections.boxes) > 0:
            for i, box in enumerate(detections.boxes):
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                detection_info += f"Object {i+1}: Confidence {conf:.2%}\n"
        else:
            detection_info = "No objects detected"
        
        return annotated_frame, detection_info
        
    except Exception as e:
        print(f"Detection error: {e}")
        return frame, f"Error: {str(e)}"

def classify_detected_regions(frame, boxes):
    """Classify each detected region individually"""
    if boxes is None or len(boxes) == 0:
        return frame, {}
    
    classified_regions = {}
    
    try:
        for idx, box in enumerate(boxes):
            # Extract box coordinates
            x1, y1, x2, y2 = map(int, box[:4])
            
            # Crop region
            region = frame[y1:y2, x1:x2]
            
            if region.size == 0:
                continue
            
            # Convert to PIL and classify
            pil_region = Image.fromarray(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
            result = predictor.predict(pil_region)
            
            # Store classification for this region
            classified_regions[f"Object {idx}"] = result['predicted_class']
            
            # Draw box with classification label
            label = f"{result['predicted_class']} ({result['confidence']:.1%})"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 255, 0), 2)
    
    except Exception as e:
        print(f"Classification error: {e}")
    
    return frame, classified_regions

def webcam_stream_detect():
    """Real-time webcam detection"""
    global webcam_active, current_prediction
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        yield None, "Error: Cannot open webcam"
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_count = 0
    
    while webcam_active:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Detection using YOLO
        if frame_count % 2 == 0:  # Process every 2nd frame for speed
            annotated_frame, detection_info = detect_waste_frame(frame)
            frame_display = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        else:
            frame_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detection_info = "Processing..."
        
        frame_count += 1
        yield frame_display, detection_info
    
    cap.release()

def webcam_stream_hybrid():
    """Real-time webcam with both YOLO detection + classification of regions"""
    global webcam_active
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        yield None, {}
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_count = 0
    
    while webcam_active:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Hybrid detection + classification
        if frame_count % 3 == 0:  # Process every 3rd frame
            try:
                # Detect objects
                results = yolo_model(frame, conf=0.4, verbose=False)
                boxes = results[0].boxes.xyxy.cpu().numpy()
                
                # Classify each region
                frame_labeled, classifications = classify_detected_regions(frame, boxes)
                frame_display = cv2.cvtColor(frame_labeled, cv2.COLOR_BGR2RGB)
                
                # Format output
                class_output = classifications if classifications else {"status": "No objects"}
                
            except Exception as e:
                frame_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                class_output = {"error": str(e)}
        else:
            frame_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            class_output = {"status": "Processing..."}
        
        frame_count += 1
        yield frame_display, class_output
    
    cap.release()

def start_webcam():
    global webcam_active
    webcam_active = True
    return "Webcam started"

def stop_webcam():
    global webcam_active
    webcam_active = False
    return "Webcam stopped"

# Create Gradio interface
with gr.Blocks(title="Waste Object Detection System") as demo:
    gr.Markdown("# Waste Object Detection System")
    gr.Markdown("Detect and classify waste objects in real-time using YOLO + Classification")
    
    with gr.Tabs():
        # YOLO Object Detection Tab
        with gr.Tab("Object Detection (YOLO)"):
            gr.Markdown("Detects waste objects with bounding boxes")
            
            with gr.Row():
                with gr.Column():
                    yolo_output = gr.Image(label="Detection Feed", type="numpy")
                with gr.Column():
                    yolo_info = gr.Textbox(label="Detection Info", lines=5)
            
            with gr.Row():
                yolo_start = gr.Button("Start Detection", variant="primary")
                yolo_stop = gr.Button("Stop", variant="stop")
            
            yolo_status = gr.Textbox(value="Ready", label="Status")
            
            yolo_start.click(
                fn=start_webcam,
                outputs=yolo_status
            ).then(
                fn=webcam_stream_detect,
                outputs=[yolo_output, yolo_info]
            )
            
            yolo_stop.click(
                fn=stop_webcam,
                outputs=yolo_status
            )
        
        # Hybrid: Detection + Classification Tab
        with gr.Tab("Detection + Classification"):
            gr.Markdown("YOLO detects objects, then classifies each region")
            
            with gr.Row():
                with gr.Column():
                    hybrid_output = gr.Image(label="Labeled Feed", type="numpy")
                with gr.Column():
                    hybrid_classes = gr.JSON(label="Classifications")
            
            with gr.Row():
                hybrid_start = gr.Button("Start Hybrid Detection", variant="primary")
                hybrid_stop = gr.Button("Stop", variant="stop")
            
            hybrid_status = gr.Textbox(value="Ready", label="Status")
            
            hybrid_start.click(
                fn=start_webcam,
                outputs=hybrid_status
            ).then(
                fn=webcam_stream_hybrid,
                outputs=[hybrid_output, hybrid_classes]
            )
            
            hybrid_stop.click(
                fn=stop_webcam,
                outputs=hybrid_status
            )
        
        # Upload Image Tab
        with gr.Tab("Upload Image"):
            with gr.Row():
                image_input = gr.Image(type="pil", label="Upload Waste Image")
            
            classify_output = gr.Label(num_top_classes=3, label="Classification")
            
            image_input.change(
                fn=classify_waste,
                inputs=image_input,
                outputs=classify_output
            )

if __name__ == "__main__":
    if not YOLO_AVAILABLE:
        print("\nTo use object detection, install: pip install ultralytics")
    
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
