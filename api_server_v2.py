from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import warnings
import os
from datetime import datetime
import json
import socket
import calendar

warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables to store model and scaler
model = None
scaler = None
feature_cols = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Constants for fixed values
TREND_SCORE = 45
DEMAND_MULTIPLIER = 1.5
CITY_MULTIPLIER = 1.0

# Month name to number mapping
MONTH_MAPPING = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
}

def get_week_of_year(month_name, week_of_month):
    """Convert month name and week of month to week of year (1-52)"""
    try:
        month_num = MONTH_MAPPING[month_name]
        
        # Calculate approximate week of year
        # Each month has roughly 4.33 weeks, but we'll use a more accurate calculation
        weeks_before_month = [0, 0, 4, 9, 13, 18, 22, 27, 31, 36, 40, 45, 49]
        
        base_week = weeks_before_month[month_num - 1]
        week_of_year = base_week + week_of_month
        
        # Ensure week is between 1 and 52
        week_of_year = max(1, min(52, week_of_year))
        
        return week_of_year
    except:
        return 1  # Default to week 1 if conversion fails

# Define model architecture (same as in breakthrough_model.py)
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.network(x).squeeze()

def get_local_ip():
    """Get the local IP address"""
    try:
        # Connect to a remote server to determine local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

def load_model():
    """Load the trained model and preprocessor"""
    global model, scaler, feature_cols
    
    try:
        # Check if model file exists
        model_path = 'breakthrough_model.pt'
        if not os.path.exists(model_path):
            print("❌ Model file not found. Please run breakthrough_model.py first!")
            return False
            
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Extract components
        scaler = checkpoint['scaler']
        feature_cols = checkpoint['feature_cols']
        
        # Initialize model with correct input size
        model = SimpleNN(len(feature_cols)).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print("✅ Model loaded successfully!")
        print(f"   Training R²: {checkpoint['train_r2']:.4f}")
        print(f"   Test R²: {checkpoint['test_r2']:.4f}")
        print(f"   Test MAE: {checkpoint['test_mae']:.2f}")
        print(f"   Features: {len(feature_cols)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def prepare_input_data(input_data):
    """Prepare input data for prediction"""
    try:
        # Convert month name and week of month to week of year
        month_name = input_data['Month']
        week_of_month = input_data['Week']
        
        week_of_year = get_week_of_year(month_name, week_of_month)
        month_num = MONTH_MAPPING[month_name]
        
        # Create the full input data with constants
        full_input_data = {
            'Week': week_of_year,
            'Month': month_num,
            'Product Name': input_data['Product Name'],
            'Store Location': input_data['Store Location'],
            'Category': input_data['Category'],
            'Promotion': input_data['Promotion'],
            'Holiday': input_data['Holiday'],
            'TrendScore': TREND_SCORE,
            'DemandMultiplier': DEMAND_MULTIPLIER,
            'CityMultiplier': CITY_MULTIPLIER
        }
        
        # Create a DataFrame with the input
        df_input = pd.DataFrame([full_input_data])
        
        # Get the reference dataset for encoding
        df_ref = pd.read_csv("enhanced_demand_dataset.csv")
        
        # Encode categorical variables
        df_encoded = pd.get_dummies(df_input, columns=['Product Name', 'Store Location', 'Category'])
        
        # Create a template with all possible feature columns (from training)
        feature_template = pd.DataFrame(columns=feature_cols)
        
        # Fill the template with input data
        for col in feature_template.columns:
            if col in df_encoded.columns:
                feature_template[col] = df_encoded[col].iloc[0]
            else:
                feature_template[col] = 0  # Default value for missing categorical columns
        
        # Convert to numpy array
        X = feature_template.values.astype(np.float32)
        
        # Scale the features
        X_scaled = scaler.transform(X.reshape(1, -1))
        
        return X_scaled
        
    except Exception as e:
        raise Exception(f"Error preparing input data: {e}")

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    """Make a demand prediction"""
    if request.method == 'OPTIONS':
        # Handle preflight request
        response = jsonify({'status': 'OK'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
        
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                'error': 'Model not loaded. Please restart the application.',
                'success': False
            }), 500
        
        # Get input data from request
        data = request.get_json()
        
        # Validate required fields (updated fields)
        required_fields = ['Week', 'Month', 'Product Name', 'Store Location', 
                          'Category', 'Promotion', 'Holiday']
        
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'error': f'Missing required field: {field}',
                    'success': False
                }), 400
        
        # Validate month name
        if data['Month'] not in MONTH_MAPPING:
            return jsonify({
                'error': f'Invalid month name. Must be one of: {list(MONTH_MAPPING.keys())}',
                'success': False
            }), 400
        
        # Validate week of month
        if not isinstance(data['Week'], int) or data['Week'] < 1 or data['Week'] > 5:
            return jsonify({
                'error': 'Week must be an integer between 1 and 5 (week of the month)',
                'success': False
            }), 400
        
        # Prepare input data
        X_scaled = prepare_input_data(data)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        
        # Make prediction
        with torch.no_grad():
            prediction = model(X_tensor).cpu().numpy()
        
        # Ensure prediction is a scalar
        if isinstance(prediction, np.ndarray):
            prediction = float(prediction.item() if prediction.size == 1 else prediction[0])
        else:
            prediction = float(prediction)
        
        # Round to reasonable precision
        prediction = round(max(0, prediction), 2)  # Ensure non-negative
        
        # Calculate week of year for response
        week_of_year = get_week_of_year(data['Month'], data['Week'])
        
        return jsonify({
            'prediction': prediction,
            'input_data': data,
            'processed_data': {
                'week_of_year': week_of_year,
                'month_number': MONTH_MAPPING[data['Month']],
                'trend_score': TREND_SCORE,
                'demand_multiplier': DEMAND_MULTIPLIER,
                'city_multiplier': CITY_MULTIPLIER
            },
            'success': True,
            'message': f'Predicted demand: {prediction} units'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/batch_predict', methods=['POST', 'OPTIONS'])
def batch_predict():
    """Make predictions for multiple inputs"""
    if request.method == 'OPTIONS':
        # Handle preflight request
        response = jsonify({'status': 'OK'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
        
    try:
        if model is None:
            return jsonify({
                'error': 'Model not loaded. Please restart the application.',
                'success': False
            }), 500
        
        data = request.get_json()
        
        if 'inputs' not in data or not isinstance(data['inputs'], list):
            return jsonify({
                'error': 'Expected "inputs" field with a list of input data',
                'success': False
            }), 400
        
        predictions = []
        errors = []
        
        for i, input_data in enumerate(data['inputs']):
            try:
                # Validate required fields
                required_fields = ['Week', 'Month', 'Product Name', 'Store Location', 
                                  'Category', 'Promotion', 'Holiday']
                
                for field in required_fields:
                    if field not in input_data:
                        raise Exception(f'Missing required field: {field}')
                
                # Validate month name
                if input_data['Month'] not in MONTH_MAPPING:
                    raise Exception(f'Invalid month name. Must be one of: {list(MONTH_MAPPING.keys())}')
                
                # Validate week of month
                if not isinstance(input_data['Week'], int) or input_data['Week'] < 1 or input_data['Week'] > 5:
                    raise Exception('Week must be an integer between 1 and 5 (week of the month)')
                
                # Prepare input data
                X_scaled = prepare_input_data(input_data)
                
                # Convert to tensor
                X_tensor = torch.FloatTensor(X_scaled).to(device)
                
                # Make prediction
                with torch.no_grad():
                    prediction = model(X_tensor).cpu().numpy()
                
                # Ensure prediction is a scalar
                if isinstance(prediction, np.ndarray):
                    prediction = float(prediction.item() if prediction.size == 1 else prediction[0])
                else:
                    prediction = float(prediction)
                
                prediction = round(max(0, prediction), 2)
                
                # Calculate week of year for response
                week_of_year = get_week_of_year(input_data['Month'], input_data['Week'])
                
                predictions.append({
                    'index': i,
                    'prediction': prediction,
                    'input': input_data,
                    'processed_data': {
                        'week_of_year': week_of_year,
                        'month_number': MONTH_MAPPING[input_data['Month']],
                        'trend_score': TREND_SCORE,
                        'demand_multiplier': DEMAND_MULTIPLIER,
                        'city_multiplier': CITY_MULTIPLIER
                    }
                })
                
            except Exception as e:
                errors.append({
                    'index': i,
                    'error': str(e),
                    'input': input_data
                })
        
        return jsonify({
            'predictions': predictions,
            'errors': errors,
            'success': True,
            'total_processed': len(data['inputs']),
            'successful_predictions': len(predictions),
            'failed_predictions': len(errors)
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    if model is None:
        return jsonify({
            'error': 'Model not loaded',
            'success': False
        }), 500
    
    try:
        # Load model metadata
        checkpoint = torch.load('breakthrough_model.pt', map_location='cpu', weights_only=False)
        
        return jsonify({
            'model_loaded': True,
            'training_r2': round(checkpoint['train_r2'], 4),
            'test_r2': round(checkpoint['test_r2'], 4),
            'test_mae': round(checkpoint['test_mae'], 2),
            'num_features': len(feature_cols),
            'device': str(device),
            'model_parameters': sum(p.numel() for p in model.parameters()),
            'constants': {
                'trend_score': TREND_SCORE,
                'demand_multiplier': DEMAND_MULTIPLIER,
                'city_multiplier': CITY_MULTIPLIER
            },
            'input_format': {
                'required_fields': ['Week', 'Month', 'Product Name', 'Store Location', 'Category', 'Promotion', 'Holiday'],
                'week_format': 'Integer 1-5 (week of the month)',
                'month_format': 'Month name (e.g., January, February, etc.)',
                'supported_months': list(MONTH_MAPPING.keys())
            },
            'success': True
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat(),
        'server': 'Flask Demand Prediction API',
        'version': '2.0.0',
        'device': str(device),
        'constants': {
            'trend_score': TREND_SCORE,
            'demand_multiplier': DEMAND_MULTIPLIER,
            'city_multiplier': CITY_MULTIPLIER
        },
        'host_info': {
            'local_ip': get_local_ip(),
            'hostname': socket.gethostname()
        }
    })

@app.route('/constants', methods=['GET'])
def get_constants():
    """Get the constant values used in predictions"""
    return jsonify({
        'constants': {
            'trend_score': TREND_SCORE,
            'demand_multiplier': DEMAND_MULTIPLIER,
            'city_multiplier': CITY_MULTIPLIER
        },
        'input_format': {
            'week': 'Integer 1-5 (week of the month)',
            'month': 'Month name (January, February, etc.)',
            'supported_months': list(MONTH_MAPPING.keys())
        },
        'success': True
    })

# Enhanced CORS support for cross-origin requests
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,X-Requested-With')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'false')
    return response

if __name__ == '__main__':
    print("🚀 Starting Demand Prediction API Server v2.0")
    print("=" * 60)
    
    # Get network info
    local_ip = get_local_ip()
    hostname = socket.gethostname()
    
    print(f"🖥️  Host Information:")
    print(f"   Hostname: {hostname}")
    print(f"   Local IP: {local_ip}")
    print(f"   Device: {device}")
    
    print(f"\n🔧 Configuration:")
    print(f"   Trend Score: {TREND_SCORE} (constant)")
    print(f"   Demand Multiplier: {DEMAND_MULTIPLIER} (constant)")
    print(f"   City Multiplier: {CITY_MULTIPLIER} (constant)")
    
    print(f"\n📅 Input Format:")
    print(f"   Week: 1-5 (week of the month)")
    print(f"   Month: Month name (e.g., January, February)")
    print(f"   Supported Months: {list(MONTH_MAPPING.keys())}")
    
    # Load the model
    if load_model():
        print("\n✅ Starting Flask API server...")
        print("📡 API Endpoints:")
        print("   POST /predict - Single prediction")
        print("   POST /batch_predict - Batch predictions") 
        print("   GET /model_info - Model information")
        print("   GET /health - Health check")
        print("   GET /constants - Get constant values")
        
        print(f"\n🌐 Server Access URLs:")
        print(f"   Local:    http://127.0.0.1:5000")
        print(f"   Network:  http://{local_ip}:5000")
        print(f"   External: http://0.0.0.0:5000")
        
        print("\n🔗 For port forwarding, use:")
        print(f"   Target: {local_ip}:5000")
        print("=" * 60)
        
        try:
            # Start the server with enhanced configuration
            app.run(
                debug=False, 
                host='0.0.0.0',  # Bind to all interfaces
                port=5000,
                threaded=True,   # Handle multiple requests
                use_reloader=False  # Disable auto-reload for production
            )
        except Exception as e:
            print(f"❌ Failed to start server: {e}")
            print("Trying alternative port 8000...")
            app.run(
                debug=False, 
                host='0.0.0.0',
                port=8000,
                threaded=True,
                use_reloader=False
            )
    else:
        print("❌ Failed to load model. Please run breakthrough_model.py first!")
        print("Exiting...")
