# Demand Prediction API v2.0 Documentation

## Overview
Updated API server with simplified input format using month names and week numbers within the month. TrendScore, DemandMultiplier, and CityMultiplier are now constant values.

## Constants
- **TrendScore**: 45 (fixed)
- **DemandMultiplier**: 1.5 (fixed)
- **CityMultiplier**: 1.0 (fixed)

## Input Format Changes

### Previous Format (v1.0)
```json
{
    "Week": 40,              // Week of year (1-52)
    "Month": 10,             // Month number (1-12)
    "TrendScore": 45,        // User input required
    "DemandMultiplier": 1.5, // User input required
    "CityMultiplier": 1.0    // User input required
}
```

### New Format (v2.0)
```json
{
    "Week": 2,               // Week of month (1-5)
    "Month": "October",      // Month name (January, February, etc.)
    // TrendScore, DemandMultiplier, CityMultiplier are now constants
}
```

## API Endpoints

### 1. Single Prediction
**POST /predict**

**Request Body:**
```json
{
    "Week": 2,
    "Month": "October",
    "Product Name": "Mithai Gift Box (Diwali)",
    "Store Location": "Delhi",
    "Category": "Festive",
    "Promotion": 1,
    "Holiday": 1
}
```

**Response:**
```json
{
    "prediction": 245.67,
    "input_data": {
        "Week": 2,
        "Month": "October",
        "Product Name": "Mithai Gift Box (Diwali)",
        "Store Location": "Delhi",
        "Category": "Festive",
        "Promotion": 1,
        "Holiday": 1
    },
    "processed_data": {
        "week_of_year": 42,
        "month_number": 10,
        "trend_score": 45,
        "demand_multiplier": 1.5,
        "city_multiplier": 1.0
    },
    "success": true,
    "message": "Predicted demand: 245.67 units"
}
```

### 2. Batch Prediction
**POST /batch_predict**

**Request Body:**
```json
{
    "inputs": [
        {
            "Week": 2,
            "Month": "October",
            "Product Name": "Mithai Gift Box (Diwali)",
            "Store Location": "Delhi",
            "Category": "Festive",
            "Promotion": 1,
            "Holiday": 1
        },
        {
            "Week": 3,
            "Month": "May",
            "Product Name": "Gaming Headset",
            "Store Location": "Mumbai",
            "Category": "Electronics",
            "Promotion": 0,
            "Holiday": 0
        }
    ]
}
```

### 3. Model Information
**GET /model_info**

**Response:**
```json
{
    "model_loaded": true,
    "training_r2": 0.8542,
    "test_r2": 0.7834,
    "test_mae": 45.67,
    "num_features": 157,
    "device": "cuda",
    "model_parameters": 25537,
    "constants": {
        "trend_score": 45,
        "demand_multiplier": 1.5,
        "city_multiplier": 1.0
    },
    "input_format": {
        "required_fields": ["Week", "Month", "Product Name", "Store Location", "Category", "Promotion", "Holiday"],
        "week_format": "Integer 1-5 (week of the month)",
        "month_format": "Month name (e.g., January, February, etc.)",
        "supported_months": ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    },
    "success": true
}
```

### 4. Health Check
**GET /health**

**Response:**
```json
{
    "status": "healthy",
    "model_loaded": true,
    "timestamp": "2024-01-01T12:00:00",
    "server": "Flask Demand Prediction API",
    "version": "2.0.0",
    "device": "cuda",
    "constants": {
        "trend_score": 45,
        "demand_multiplier": 1.5,
        "city_multiplier": 1.0
    },
    "host_info": {
        "local_ip": "192.168.1.100",
        "hostname": "YOUR-PC"
    }
}
```

### 5. Get Constants
**GET /constants**

**Response:**
```json
{
    "constants": {
        "trend_score": 45,
        "demand_multiplier": 1.5,
        "city_multiplier": 1.0
    },
    "input_format": {
        "week": "Integer 1-5 (week of the month)",
        "month": "Month name (January, February, etc.)",
        "supported_months": ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    },
    "success": true
}
```

## Input Field Specifications

| Field | Type | Description | Valid Values |
|-------|------|-------------|--------------|
| Week | Integer | Week of the month | 1-5 |
| Month | String | Month name | January, February, March, April, May, June, July, August, September, October, November, December |
| Product Name | String | Product name | See dataset for valid values |
| Store Location | String | Store location | See dataset for valid values |
| Category | String | Product category | See dataset for valid values |
| Promotion | Integer | Promotion active | 0 or 1 |
| Holiday | Integer | Holiday period | 0 or 1 |

## Week to Week-of-Year Mapping

The API automatically converts week-of-month to week-of-year using this mapping:

| Month | Week 1 | Week 2 | Week 3 | Week 4 | Week 5 |
|-------|--------|--------|--------|--------|--------|
| January | 1 | 2 | 3 | 4 | 5 |
| February | 5 | 6 | 7 | 8 | 9 |
| March | 10 | 11 | 12 | 13 | 14 |
| April | 14 | 15 | 16 | 17 | 18 |
| May | 19 | 20 | 21 | 22 | 23 |
| June | 23 | 24 | 25 | 26 | 27 |
| July | 28 | 29 | 30 | 31 | 32 |
| August | 32 | 33 | 34 | 35 | 36 |
| September | 37 | 38 | 39 | 40 | 41 |
| October | 41 | 42 | 43 | 44 | 45 |
| November | 46 | 47 | 48 | 49 | 50 |
| December | 50 | 51 | 52 | 52 | 52 |

## Example Usage

### curl Examples

**Single Prediction:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Week": 2,
    "Month": "October",
    "Product Name": "Mithai Gift Box (Diwali)",
    "Store Location": "Delhi",
    "Category": "Festive",
    "Promotion": 1,
    "Holiday": 1
  }'
```

**Get Constants:**
```bash
curl http://localhost:5000/constants
```

**Health Check:**
```bash
curl http://localhost:5000/health
```

### JavaScript Example
```javascript
const prediction = await fetch('http://localhost:5000/predict', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({
        Week: 2,
        Month: "October",
        "Product Name": "Mithai Gift Box (Diwali)",
        "Store Location": "Delhi",
        Category: "Festive",
        Promotion: 1,
        Holiday: 1
    })
});

const result = await prediction.json();
console.log('Predicted demand:', result.prediction);
```

## Error Handling

The API returns appropriate HTTP status codes and error messages:

**400 Bad Request - Missing Field:**
```json
{
    "error": "Missing required field: Month",
    "success": false
}
```

**400 Bad Request - Invalid Month:**
```json
{
    "error": "Invalid month name. Must be one of: ['January', 'February', ...]",
    "success": false
}
```

**400 Bad Request - Invalid Week:**
```json
{
    "error": "Week must be an integer between 1 and 5 (week of the month)",
    "success": false
}
```

**500 Internal Server Error:**
```json
{
    "error": "Model not loaded. Please restart the application.",
    "success": false
}
```

## Migration from v1.0 to v2.0

If you have existing code using v1.0 format, here's how to migrate:

### v1.0 Input:
```json
{
    "Week": 40,
    "Month": 10,
    "TrendScore": 45,
    "DemandMultiplier": 1.5,
    "CityMultiplier": 1.0,
    "Product Name": "Mithai Gift Box (Diwali)",
    "Store Location": "Delhi",
    "Category": "Festive",
    "Promotion": 1,
    "Holiday": 1
}
```

### v2.0 Input:
```json
{
    "Week": 2,
    "Month": "October",
    "Product Name": "Mithai Gift Box (Diwali)",
    "Store Location": "Delhi",
    "Category": "Festive",
    "Promotion": 1,
    "Holiday": 1
}
```

### Conversion Rules:
1. **Week**: Convert from week-of-year to week-of-month
   - Week 40 (October) → Week 2 of October
2. **Month**: Convert from number to name
   - Month 10 → "October"
3. **Constants**: Remove from input (now handled automatically)
   - TrendScore, DemandMultiplier, CityMultiplier

## Running the Server

```bash
# Start v2.0 server
start_api_v2.bat

# Or manually
cd "d:\Code_stuff\demand_prediction"
.spark\Scripts\activate.bat
python api_server_v2.py
```

The server will be available at `http://localhost:5000` with enhanced port forwarding support.
