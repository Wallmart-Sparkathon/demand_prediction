# Demand Prediction API - Date Input Version

## Overview
This Flask API server provides demand prediction capabilities with **date input support**. Instead of manually specifying week and month numbers, you can provide a date and the system will automatically derive the week and month values.

## Features
- ✅ **Date Input**: Use actual dates instead of week/month numbers
- ✅ **Multiple Date Formats**: Supports various date formats
- ✅ **Automatic Derivation**: Week and month are calculated from the date
- ✅ **Weekend Detection**: Identifies weekends for better predictions
- ✅ **Batch Processing**: Process multiple predictions at once
- ✅ **Date Analysis**: Endpoint to analyze date inputs

## Quick Start

### 1. Start the Server
```bash
# Run the startup script
start_api_date_input.bat

# Or run directly
python api_server_date_input.py
```

### 2. Server Access
- **Local**: http://127.0.0.1:5000
- **Network**: http://YOUR_IP:5000
- **External**: http://0.0.0.0:5000

## API Endpoints

### POST /predict
Make a single demand prediction using date input.

**Required Fields:**
- `Date`: Date in various formats (see supported formats below)
- `Product Name`: Name of the product
- `Store Location`: Location of the store
- `Category`: Product category

**Optional Fields:**
- `Promotion`: Promotion status (0 or 1) - defaults to 0
- `Holiday`: Holiday status (0 or 1) - defaults to 0
- `TrendScore`: Trend score (defaults to 45)
- `DemandMultiplier`: Demand multiplier (defaults to 1.5)
- `CityMultiplier`: City multiplier (defaults to 1.0)

**Example Request:**
```json
{
    "Date": "2024-03-15",
    "Product Name": "Laptop",
    "Store Location": "New York",
    "Category": "Electronics",
    "Promotion": 1,
    "Holiday": 0
}
```

**Example Response:**
```json
{
    "prediction": 156.78,
    "input_data": {
        "Date": "2024-03-15",
        "Product Name": "Laptop",
        "Store Location": "New York",
        "Category": "Electronics",
        "Promotion": 1,
        "Holiday": 0
    },
    "processed_data": {
        "Week": 11,
        "Month": 3,
        "Product Name": "Laptop",
        "Store Location": "New York",
        "Category": "Electronics",
        "Promotion": 1,
        "Holiday": 0,
        "TrendScore": 45,
        "DemandMultiplier": 1.5,
        "CityMultiplier": 1.0
    },
    "date_info": {
        "parsed_date": "2024-03-15",
        "month": 3,
        "week": 11,
        "day_of_week": 5,
        "day_of_year": 75,
        "is_weekend": true
    },
    "success": true,
    "message": "Predicted demand: 156.78 units for March 15, 2024"
}
```

### POST /batch_predict
Make multiple predictions at once.

**Example Request:**
```json
{
    "inputs": [
        {
            "Date": "2024-03-15",
            "Product Name": "Laptop",
            "Store Location": "New York",
            "Category": "Electronics"
        },
        {
            "Date": "2024-06-20",
            "Product Name": "Smartphone",
            "Store Location": "Los Angeles",
            "Category": "Electronics"
        }
    ]
}
```

### POST /date_info
Analyze and parse date input for debugging.

**Example Request:**
```json
{
    "Date": "2024-03-15"
}
```

**Example Response:**
```json
{
    "success": true,
    "date_info": {
        "original_input": "2024-03-15",
        "parsed_date": "2024-03-15",
        "formatted_date": "March 15, 2024",
        "month": 3,
        "month_name": "March",
        "week": 11,
        "day_of_week": 5,
        "day_name": "Friday",
        "day_of_year": 75,
        "is_weekend": true
    }
}
```

### GET /model_info
Get information about the loaded model.

### GET /health
Health check endpoint.

## Supported Date Formats

The API supports multiple date formats:

1. **ISO Format**: `2024-03-15` (YYYY-MM-DD)
2. **US Format**: `03/15/2024` (MM/DD/YYYY)
3. **European Format**: `15/03/2024` (DD/MM/YYYY)
4. **Dash Format**: `03-15-2024` (MM-DD-YYYY)
5. **Dash European**: `15-03-2024` (DD-MM-YYYY)
6. **With Time**: `2024-03-15 14:30:00`

## Date Processing Details

### Week Calculation
- Uses ISO week numbering (1-53)
- Automatically converts to model-compatible range (1-52)
- Week 1 is the first week with at least 4 days in the new year

### Month Calculation
- Standard month numbering (1-12)
- January = 1, December = 12

### Additional Date Info
- **Day of Week**: Monday=1, Sunday=7
- **Day of Year**: 1-366
- **Weekend Detection**: Saturday and Sunday are weekends
- **Date Validation**: Ensures valid dates

## Frontend Integration

### HTML Date Picker Integration
```html
<!-- Basic date picker -->
<input type="date" id="predictionDate" name="date" />

<!-- With JavaScript -->
<script>
function makePrediction() {
    const date = document.getElementById('predictionDate').value;
    const data = {
        "Date": date,
        "Product Name": "Laptop",
        "Store Location": "New York",
        "Category": "Electronics"
    };
    
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
        console.log('Prediction:', result.prediction);
        console.log('Date Info:', result.date_info);
    });
}
</script>
```

### JavaScript Date Object
```javascript
// Using JavaScript Date object
const today = new Date();
const dateString = today.toISOString().split('T')[0]; // Format: YYYY-MM-DD

const predictionData = {
    "Date": dateString,
    "Product Name": "Laptop",
    "Store Location": "New York",
    "Category": "Electronics"
};
```

## Error Handling

### Common Errors
1. **Invalid Date Format**: Provide date in supported formats
2. **Missing Required Fields**: Ensure Date, Product Name, Store Location, and Category are provided
3. **Model Not Loaded**: Restart the server if model fails to load

### Error Response Format
```json
{
    "error": "Error message description",
    "success": false
}
```

## Testing Examples

### cURL Examples
```bash
# Single prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Date": "2024-03-15",
    "Product Name": "Laptop",
    "Store Location": "New York",
    "Category": "Electronics"
  }'

# Date info
curl -X POST http://localhost:5000/date_info \
  -H "Content-Type: application/json" \
  -d '{"Date": "2024-03-15"}'

# Health check
curl http://localhost:5000/health
```

### Python Examples
```python
import requests
import json

# Single prediction
url = "http://localhost:5000/predict"
data = {
    "Date": "2024-03-15",
    "Product Name": "Laptop",
    "Store Location": "New York",
    "Category": "Electronics"
}

response = requests.post(url, json=data)
result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Week: {result['date_info']['week']}")
print(f"Month: {result['date_info']['month']}")
```

## Default Values

The following values are set as constants (no need to specify):
- **TrendScore**: 45
- **DemandMultiplier**: 1.5
- **CityMultiplier**: 1.0
- **Promotion**: 0 (if not specified)
- **Holiday**: 0 (if not specified)

## Requirements

- Python 3.8+
- Flask 2.3.3+
- PyTorch 2.0+
- pandas 2.0+
- scikit-learn 1.0+
- numpy 1.20+

## Files Required

1. `api_server_date_input.py` - Main API server file
2. `breakthrough_model.pt` - Trained model file
3. `enhanced_demand_dataset.csv` - Dataset for categorical encoding
4. `start_api_date_input.bat` - Startup script

## Tips for Frontend Development

1. **Use HTML5 Date Picker**: Provides native date selection
2. **Validate Dates**: Check date ranges and formats on frontend
3. **Handle Timezones**: Consider timezone differences if needed
4. **Error Handling**: Implement proper error handling for invalid dates
5. **Date Formatting**: Use consistent date format (YYYY-MM-DD recommended)

## Changelog

**v1.0 - Date Input Version**
- Added date input parsing with multiple format support
- Automatic week and month derivation from dates
- Weekend detection and day-of-year calculation
- New `/date_info` endpoint for date analysis
- Enhanced error handling for date parsing
- Updated documentation for frontend integration

---

**Note**: This API server binds to `0.0.0.0:5000` for external access and port forwarding compatibility.
