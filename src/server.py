import os
import io
import time
import logging
from flask import Flask, request, jsonify
from PIL import Image
import torch
from torchvision import transforms

from model_architecture import PetLogisticRegression

# Initialize Flask App
app = Flask(__name__)

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FlaskInferenceAPI")

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PetLogisticRegression()

model_path = os.environ.get("MODEL_PATH", "pet_logreg_model.pth")
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
else:
    logger.warning(f"Model file {model_path} not found! Using uninitialized weights.")
    
model.to(device)
model.eval()

img_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.before_request
def start_timer():
    request.start_time = time.time()

@app.after_request
def log_request(response):
    if hasattr(request, 'start_time'):
        latency = time.time() - request.start_time
        logger.info(f"Path: {request.path} | Method: {request.method} | Status: {response.status_code} | Latency: {latency:.4f}s")
    return response

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"status": "ok", "message": "Service is healthy"})

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        tensor = img_transforms(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(tensor)
            prob = torch.sigmoid(outputs).item()
            
        prediction = "Dog" if prob > 0.5 else "Cat"
        
        return jsonify({
            "predicted_class": prediction,
            "confidence": prob
        })
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
