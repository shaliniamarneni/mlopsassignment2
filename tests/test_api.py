import pytest
import io
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from server import app
from PIL import Image

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_ping_endpoint(client):
    response = client.get('/ping')
    assert response.status_code == 200
    assert response.json == {"status": "ok", "message": "Service is healthy"}

def test_predict_endpoint_no_file(client):
    response = client.post('/predict')
    assert response.status_code == 400
    assert "error" in response.json

def test_predict_endpoint_with_image(client):
    # Create a dummy image
    img = Image.new('RGB', (100, 100), color = 'red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    
    data = {
        'file': (img_byte_arr, 'test.jpg')
    }
    
    response = client.post('/predict', data=data, content_type='multipart/form-data')
    assert response.status_code == 200
    assert "predicted_class" in response.json
    assert "confidence" in response.json
