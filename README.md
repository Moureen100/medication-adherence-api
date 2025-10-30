
# Medication Adherence Prediction API

Deep learning API for predicting medication adherence levels using the DAG Model.

## API URL
https://your-app-name.onrender.com

## Endpoints
- GET / - API information
- GET /health - Health check
- GET /info - Model details
- GET /features - Feature names
- POST /predict - Single prediction
- POST /batch_predict - Batch predictions

## Example Usage
```python
import requests

API_URL = "https://your-app-name.onrender.com"

# Get model info
info = requests.get(f"{API_URL}/info").json()

# Make prediction (13 features expected)
features = [0.5, 0.3, 0.7, 0.2, 0.8, 0.1, 0.9, 0.4, 0.6, 0.2, 0.3, 0.7, 0.5]
response = requests.post(f"{API_URL}/predict", json={"features": features})
print(response.json())

---
