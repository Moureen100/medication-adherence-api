# 🏥 Medication Adherence Prediction API

**🌍 Live Production URL: https://medication-adherence-api.onrender.com**

A production-ready deep learning API for predicting medication adherence levels using PyTorch DAG Model.

## 🚀 Features
- Real-time adherence predictions
- 13 input features  
- 3 output classes: HIGH NON-ADHERENT, LOW ADHERENT, MEDIUM ADHERENT
- Batch prediction support
- Health monitoring

## 📊 Quick Start

### Using Python
\\\python
import requests

API_URL = \"https://medication-adherence-api.onrender.com\"

# Check API health
health = requests.get(f\"{API_URL}/health\").json()
print(f\"Status: {health['status']}, Model: {health['model_loaded']}\")

# Make prediction (13 features required)
features = [0.5, 0.3, 0.7, 0.2, 0.8, 0.1, 0.9, 0.4, 0.6, 0.2, 0.3, 0.7, 0.5]
response = requests.post(f\"{API_URL}/predict\", json={\"features\": features})
print(f\"Prediction: {response.json()['predictions'][0]}\")
\\\

### Using PowerShell
\\\powershell
\https://medication-adherence-api.onrender.com = \"https://medication-adherence-api.onrender.com\"

# Test health endpoint
Invoke-RestMethod -Uri \"\https://medication-adherence-api.onrender.com/health\" -Method Get

# Make prediction
\{
    "features":  [
                     0.5,
                     0.3,
                     0.7,
                     0.2,
                     0.8,
                     0.1,
                     0.9,
                     0.4,
                     0.6,
                     0.2,
                     0.3,
                     0.7,
                     0.5
                 ]
} = '{\"features\": [0.5, 0.3, 0.7, 0.2, 0.8, 0.1, 0.9, 0.4, 0.6, 0.2, 0.3, 0.7, 0.5]}'
Invoke-RestMethod -Uri \"\https://medication-adherence-api.onrender.com/predict\" -Method Post -Body \{
    "features":  [
                     0.5,
                     0.3,
                     0.7,
                     0.2,
                     0.8,
                     0.1,
                     0.9,
                     0.4,
                     0.6,
                     0.2,
                     0.3,
                     0.7,
                     0.5
                 ]
} -ContentType \"application/json\"
\\\

## 🔌 API Endpoints

| Method | Endpoint | Description | Body |
|--------|----------|-------------|------|
| GET | \/\ | API information | - |
| GET | \/health\ | Health status | - |
| **POST** | \/predict\ | Single prediction | \{\"features\": [array of 13 numbers]}\ |
| **POST** | \/batch_predict\ | Batch predictions | \{\"features\": [[array1], [array2], ...]}\ |

## 🛠️ Technology Stack
- **Backend**: Python, Flask, PyTorch
- **ML Model**: DAG Neural Network
- **Deployment**: Render
- **Features**: 13 clinical parameters

## 📁 Repository
https://github.com/Moureen100/medication-adherence-api
