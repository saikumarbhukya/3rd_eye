from flask import Flask, request, jsonify
import cv2
import numpy as np
from PIL import Image
import io
import time
import torch

app = Flask(__name__)

# Load YOLO model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

@app.route('/process', methods=['POST'])
def process_image():
    start_time = time.time()
    
    # Get image from request
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
        
    file = request.files['file']
    image_bytes = file.read()
    
    # Convert to format suitable for model
    image = Image.open(io.BytesIO(image_bytes))
    
    # Perform detection
    results = model(image)
    
    # Process results
    detections = []
    for pred in results.xyxy[0]:  # xyxy format: x1, y1, x2, y2, confidence, class
        detections.append({
            'label': results.names[int(pred[5])],
            'confidence': float(pred[4]),
            'bbox': {
                'x1': float(pred[0]),
                'y1': float(pred[1]),
                'x2': float(pred[2]),
                'y2': float(pred[3])
            }
        })
    
    processing_time = time.time() - start_time
    
    return jsonify({
        'objects': detections,
        'processing_time': processing_time
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
