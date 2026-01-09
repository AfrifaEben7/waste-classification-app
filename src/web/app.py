"""
Flask web application for waste classification
"""
from flask import Flask, render_template, request, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.predictor import WasteClassifier

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-in-production'

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'efficientnet_b4_final.onnx')  # ONNX model
MODEL_ARCH = 'efficientnet_b4'  # Using EfficientNet-B4
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize predictor
predictor = None

def init_predictor():
    """Initialize the waste classifier"""
    global predictor
    if predictor is None:
        if os.path.exists(MODEL_PATH):
            predictor = WasteClassifier(MODEL_PATH, model_arch=MODEL_ARCH)
        else:
            print(f"Warning: Model not found at {MODEL_PATH}")
            print("Please place your .pth model file in the models/ directory")
            print("Or train a new model using: python -m src.training.train")

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    print("=" * 50)
    print("PREDICT ROUTE CALLED!")
    print("Method:", request.method)
    print("Files:", request.files)
    print("=" * 50)
    
    # Check if file was uploaded
    if 'file' not in request.files:
        flash('No file uploaded', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    # Check if file is empty
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    # Check if file is allowed
    if not allowed_file(file.filename):
        flash('Invalid file type. Please upload an image (PNG, JPG, JPEG, GIF)', 'error')
        return redirect(url_for('index'))
    
    try:
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Initialize predictor if needed
        if predictor is None:
            init_predictor()
        
        if predictor is None:
            flash('Model not loaded. Please train the model first.', 'error')
            return redirect(url_for('index'))
        
        # Make prediction
        results = predictor.predict(filepath)
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return render_template('result.html', 
                             results=results,
                             filename=filename)
    
    except Exception as e:
        flash(f'Error processing image: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

if __name__ == '__main__':
    init_predictor()
    app.run(debug=True, host='0.0.0.0', port=8080)