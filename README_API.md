# Demand Prediction Flask API

A Flask web application for predicting product demand using a PyTorch neural network model.

## Features

- **Web Interface**: User-friendly form for making predictions
- **REST API**: JSON endpoints for programmatic access
- **Batch Predictions**: Support for multiple predictions at once
- **Model Information**: View model performance metrics
- **Health Check**: Monitor application status

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have the trained model file (`breakthrough_model.pt`) in the same directory

3. Run the application:
```bash
python app.py
```

The app will be available at `http://localhost:5000`

## API Endpoints

### 1. Web Interface
- **GET /** - Main prediction form interface

### 2. Single Prediction
- **POST /predict**
- Content-Type: application/json
- **Request Body:**
```json
{
    "Week": 40,
    "Month": 10,
    "Product Name": "Mithai Gift Box (Diwali)",
    "Category": "Festive",
    "Store Location": "Delhi",
    "Promotion": 1,
    "Holiday": 1,
    "TrendScore": 50,
    "DemandMultiplier": 2.0,
    "CityMultiplier": 1.3
}
```

- **Response:**
```json
{
    "prediction": 245.67,
    "input_data": {...},
    "success": true,
    "message": "Predicted demand: 245.67 units"
}
```

### 3. Batch Predictions
- **POST /batch_predict**
- Content-Type: application/json
- **Request Body:**
```json
{
    "inputs": [
        {
            "Week": 40,
            "Month": 10,
            "Product Name": "Mithai Gift Box (Diwali)",
            ...
        },
        {
            "Week": 20,
            "Month": 5,
            "Product Name": "Gaming Headset",
            ...
        }
    ]
}
```

- **Response:**
```json
{
    "predictions": [
        {
            "index": 0,
            "prediction": 245.67,
            "input": {...}
        }
    ],
    "errors": [],
    "success": true,
    "total_processed": 2,
    "successful_predictions": 2,
    "failed_predictions": 0
}
```

### 4. Model Information
- **GET /model_info**
- **Response:**
```json
{
    "model_loaded": true,
    "training_r2": 0.8542,
    "test_r2": 0.7834,
    "test_mae": 45.67,
    "num_features": 157,
    "device": "cpu",
    "model_parameters": 25537,
    "success": true
}
```

### 5. Health Check
- **GET /health**
- **Response:**
```json
{
    "status": "healthy",
    "model_loaded": true,
    "timestamp": "2024-01-01T12:00:00"
}
```

## Required Input Fields

| Field | Type | Description | Range/Options |
|-------|------|-------------|---------------|
| Week | Integer | Week of year | 1-52 |
| Month | Integer | Month | 1-12 |
| Product Name | String | Product name | See dataset for options |
| Category | String | Product category | Festive, Electronics, Health, etc. |
| Store Location | String | Store location | Delhi, Mumbai, Bangalore, etc. |
| Promotion | Integer | Promotion active | 0 or 1 |
| Holiday | Integer | Holiday period | 0 or 1 |
| TrendScore | Integer | Trend score | 0-100 |
| DemandMultiplier | Float | Demand multiplier | 0.5-10.0 |
| CityMultiplier | Float | City multiplier | 0.5-3.0 |

## Example Usage with curl

### Single Prediction
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Week": 40,
    "Month": 10,
    "Product Name": "Mithai Gift Box (Diwali)",
    "Category": "Festive",
    "Store Location": "Delhi",
    "Promotion": 1,
    "Holiday": 1,
    "TrendScore": 50,
    "DemandMultiplier": 2.0,
    "CityMultiplier": 1.3
  }'
```

### Model Info
```bash
curl http://localhost:5000/model_info
```

### Health Check
```bash
curl http://localhost:5000/health
```

## Error Handling

All endpoints return appropriate HTTP status codes:
- 200: Success
- 400: Bad Request (missing or invalid input)
- 500: Internal Server Error

Error responses include:
```json
{
    "error": "Error description",
    "success": false
}
```

## Web Interface Features

1. **Interactive Form**: Easy-to-use form with validation
2. **Example Templates**: Pre-filled examples for different product types
3. **Real-time Results**: Instant prediction display
4. **Model Metrics**: View model performance information
5. **Responsive Design**: Works on desktop and mobile devices

## Model Details

The application uses a PyTorch neural network with:
- Input features: 157 (after one-hot encoding)
- Architecture: 157 → 128 → 64 → 32 → 1
- Activation: ReLU with Dropout
- Loss: MSE
- Optimizer: Adam

## Troubleshooting

1. **Model not loaded**: Ensure `breakthrough_model.pt` exists in the application directory
2. **Missing features**: Check that all required input fields are provided
3. **Import errors**: Install all dependencies from `requirements.txt`
4. **Port conflicts**: Change the port in `app.py` if 5000 is already in use
