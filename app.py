import pickle        
import json
import numpy as np   
import pandas as pd  
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify
import os
from datetime import datetime
from flask_cors import CORS  # Add this import

# Define the DAG Model architecture
class DAGModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=3):
        super(DAGModel, self).__init__()

        # Multiple independent pathways (simulating DAG)
        self.pathway1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )

        self.pathway2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        self.pathway3 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )
        
        # Merge pathways
        self.merge = nn.Sequential(
            nn.Linear((hidden_dim // 2) * 2 + (hidden_dim // 4), hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        p1 = self.pathway1(x)
        p2 = self.pathway2(x)
        p3 = self.pathway3(x)

        merged = torch.cat([p1, p2, p3], dim=1)
        output = self.merge(merged)
        return output

app = Flask(__name__)

# Global variables for model and preprocessing objects
model = None
scaler = None
label_encoder = None
feature_names = None
class_names = None

def load_model():
    """Load the trained DAG model and preprocessing objects"""
    global model, scaler, label_encoder, feature_names, class_names

    try:
        print("Loading DAG Model and preprocessing objects...")

        # Load model configuration
        with open('model_config.json', 'r') as f:
            config = json.load(f)
        
        # Initialize DAG Model
        model = DAGModel(
            input_dim=config['input_dim'],
            num_classes=config['num_classes']
        )

        # Load model weights - UPDATED FILENAME
        model.load_state_dict(torch.load('best_DAG_Model.pth', map_location='cpu'))
        model.eval()
        
        # Load preprocessing objects
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)

        feature_names = config['feature_names']
        class_names = config['class_names']

        print("DAG Model and preprocessing objects loaded successfully!")
        print(f"   Model: DAG Model")
        print(f"   Input dimensions: {config['input_dim']}")
        print(f"   Number of classes: {config['num_classes']}")
        print(f"   Classes: {class_names}")
        print(f"   Features: {len(feature_names)}")

    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise e

@app.route('/')
def home():
    return jsonify({
        "message": "Medication Adherence API is running!",
        "status": "success",
        "model_loaded": model is not None
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for single prediction"""
    try:
        data = request.get_json()

        if not data or 'features' not in data:
            return jsonify({"error": "No features provided"}), 400

        # Convert to numpy array and ensure 2D
        patient_features = np.array(data['features']).reshape(1, -1)
        
        # Validate input shape
        if patient_features.shape[1] != len(feature_names):
            return jsonify({
                "error": f"Expected {len(feature_names)} features, got {patient_features.shape[1]}"
            }), 400

        # Scale features
        features_scaled = scaler.transform(patient_features)

        # Convert to tensor
        features_tensor = torch.FloatTensor(features_scaled)

        # Make prediction
        with torch.no_grad():
            outputs = model(features_tensor)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
        
        # Convert back to original labels
        predictions = label_encoder.inverse_transform(predicted.numpy())

        # Get probability for each class
        class_probabilities = []
        for prob in probabilities.numpy():
            class_prob_dict = {}
            for i, class_name in enumerate(class_names):
                class_prob_dict[class_name] = float(prob[i])
            class_probabilities.append(class_prob_dict)

        # Prepare response
        response = {
            "predictions": predictions.tolist(),
            "probabilities": probabilities.numpy().tolist(),
            "class_probabilities": class_probabilities,
            "class_mapping": dict(zip(range(len(class_names)), class_names)),
            "model_used": "DAG Model",
            "num_predictions": len(predictions)
        }
        
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Endpoint for batch predictions"""
    try:
        data = request.get_json()

        if not data or 'features' not in data:
            return jsonify({"error": "No features provided"}), 400

        patient_features = np.array(data['features'])
        
        # Validate input shape
        if patient_features.shape[1] != len(feature_names):
            return jsonify({
                "error": f"Expected {len(feature_names)} features, got {patient_features.shape[1]}"
            }), 400

        # Scale features
        features_scaled = scaler.transform(patient_features)

        # Convert to tensor
        features_tensor = torch.FloatTensor(features_scaled)
        
        # Make predictions
        with torch.no_grad():
            outputs = model(features_tensor)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

        # Convert back to original labels
        predictions = label_encoder.inverse_transform(predicted.numpy())

        # Prepare detailed response
        detailed_results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities.numpy())):
            detailed_results.append({
                "sample_id": i,
                "prediction": pred,
                "confidence": float(np.max(prob)),
                "all_probabilities": {
                    class_names[j]: float(prob[j]) for j in range(len(class_names))
                }
            })
        
        response = {
            "batch_predictions": predictions.tolist(),
            "detailed_results": detailed_results,
            "total_samples": len(predictions),
            "model_used": "DAG Model"
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Load model when application starts
print("Starting Medication Adherence API...")
load_model()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(debug=False, host='0.0.0.0', port=port)
    app = Flask(__name__)
CORS(app)  # Add this line - enables CORS for all routes
