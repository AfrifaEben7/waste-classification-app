from flask import Blueprint, request, render_template
from src.inference.predictor import predict

web_routes = Blueprint('web_routes', __name__)

@web_routes.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@web_routes.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    if file:
        prediction = predict(file)
        return render_template('result.html', prediction=prediction)

@web_routes.route('/result', methods=['GET'])
def result():
    return render_template('result.html')