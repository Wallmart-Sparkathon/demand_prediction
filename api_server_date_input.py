from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import warnings
import os
from datetime import datetime, date
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

def parse_date_input(date_input):
    """Parse date input and extract week and month"""
    try:
        # Try to parse different date formats
        if isinstance(date_input, str):
            # Common date formats
            formats = [
                '%Y-%m-%d',      # 2024-01-15
                '%m/%d/%Y',      # 01/15/2024
                '%d/%m/%Y',      # 15/01/2024
                '%Y-%m-%d %H:%M:%S',  # 2024-01-15 00:00:00
                '%m-%d-%Y',      # 01-15-2024
                '%d-%m-%Y'       # 15-01-2024
            ]
            
            parsed_date = None
            for fmt in formats:
                try:
                    parsed_date = datetime.strptime(date_input, fmt).date()
                    break
                except ValueError:
                    continue
            
            if parsed_date is None:
                raise ValueError(f"Could not parse date: {date_input}")
        
        elif isinstance(date_input, datetime):
            parsed_date = date_input.date()
        elif isinstance(date_input, date):
            parsed_date = date_input
        else:
            raise ValueError(f"Invalid date type: {type(date_input)}")
        
        # Extract month and week number
        month = parsed_date.month
        # ISO week number (1-53)
        week = parsed_date.isocalendar()[1]
        
        # For consistency with training data, we might need to adjust week to 1-52
        if week > 52:
            week = 52
        
        return {
            'parsed_date': parsed_date,
            'month': month,
            'week': week,
            'day_of_week': parsed_date.weekday() + 1,  # Monday=1, Sunday=7
            'day_of_year': parsed_date.timetuple().tm_yday,
            'is_weekend': parsed_date.weekday() >= 5  # Saturday=5, Sunday=6
        }
        
    except Exception as e:
        raise ValueError(f"Error parsing date '{date_input}': {e}")

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
        # Parse the date and extract derived values
        date_info = parse_date_input(input_data['Date'])
        
        # Create the processed input data
        processed_input = {
            'Week': date_info['week'],
            'Month': date_info['month'],
            'Product Name': input_data['Product Name'],
            'Store Location': input_data['Store Location'],
            'Category': input_data['Category'],
            'Promotion': input_data.get('Promotion', 0),
            'Holiday': input_data.get('Holiday', 0),
            'TrendScore': input_data.get('TrendScore', 45),  # Default constant
            'DemandMultiplier': input_data.get('DemandMultiplier', 1.5),  # Default constant
            'CityMultiplier': input_data.get('CityMultiplier', 1.0)  # Default constant
        }
        
        # Create a DataFrame with the processed input
        df_input = pd.DataFrame([processed_input])
        
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
        
        return X_scaled, processed_input, date_info
        
    except Exception as e:
        raise Exception(f"Error preparing input data: {e}")

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    """Make a demand prediction using date input"""
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
        
        # Validate required fields
        required_fields = ['Date', 'Product Name', 'Store Location', 'Category']
        
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'error': f'Missing required field: {field}',
                    'success': False
                }), 400
        
        # Prepare input data
        X_scaled, processed_input, date_info = prepare_input_data(data)
        
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
        
        return jsonify({
            'prediction': prediction,
            'input_data': data,
            'processed_data': processed_input,
            'date_info': {
                'parsed_date': date_info['parsed_date'].isoformat(),
                'month': date_info['month'],
                'week': date_info['week'],
                'day_of_week': date_info['day_of_week'],
                'day_of_year': date_info['day_of_year'],
                'is_weekend': date_info['is_weekend']
            },
            'success': True,
            'message': f'Predicted demand: {prediction} units for {date_info["parsed_date"].strftime("%B %d, %Y")}'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/batch_predict', methods=['POST', 'OPTIONS'])
def batch_predict():
    """Make predictions for multiple inputs with date parsing"""
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
                # Prepare input data
                X_scaled, processed_input, date_info = prepare_input_data(input_data)
                
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
                predictions.append({
                    'index': i,
                    'prediction': prediction,
                    'input': input_data,
                    'processed_data': processed_input,
                    'date_info': {
                        'parsed_date': date_info['parsed_date'].isoformat(),
                        'month': date_info['month'],
                        'week': date_info['week'],
                        'day_of_week': date_info['day_of_week'],
                        'day_of_year': date_info['day_of_year'],
                        'is_weekend': date_info['is_weekend']
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

@app.route('/date_info', methods=['POST'])
def get_date_info():
    """Get date information for debugging/testing"""
    try:
        data = request.get_json()
        
        if 'Date' not in data:
            return jsonify({
                'error': 'Missing required field: Date',
                'success': False
            }), 400
        
        date_info = parse_date_input(data['Date'])
        
        return jsonify({
            'success': True,
            'date_info': {
                'original_input': data['Date'],
                'parsed_date': date_info['parsed_date'].isoformat(),
                'formatted_date': date_info['parsed_date'].strftime('%B %d, %Y'),
                'month': date_info['month'],
                'month_name': calendar.month_name[date_info['month']],
                'week': date_info['week'],
                'day_of_week': date_info['day_of_week'],
                'day_name': calendar.day_name[date_info['parsed_date'].weekday()],
                'day_of_year': date_info['day_of_year'],
                'is_weekend': date_info['is_weekend']
            }
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
            'api_version': 'Date Input v1.0',
            'supported_date_formats': [
                'YYYY-MM-DD (2024-01-15)',
                'MM/DD/YYYY (01/15/2024)',
                'DD/MM/YYYY (15/01/2024)',
                'MM-DD-YYYY (01-15-2024)',
                'DD-MM-YYYY (15-01-2024)'
            ],
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
        'server': 'Flask Demand Prediction API - Date Input Version',
        'version': '1.0.0',
        'device': str(device),
        'api_features': [
            'Date input parsing',
            'Automatic week/month derivation',
            'Multiple date format support',
            'Batch predictions',
            'Date information endpoint'
        ],
        'host_info': {
            'local_ip': get_local_ip(),
            'hostname': socket.gethostname()
        }
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
    print("🚀 Starting Demand Prediction API Server - Date Input Version")
    print("=" * 70)
    
    # Get network info
    local_ip = get_local_ip()
    hostname = socket.gethostname()
    
    print(f"🖥️  Host Information:")
    print(f"   Hostname: {hostname}")
    print(f"   Local IP: {local_ip}")
    print(f"   Device: {device}")
    
    # Load the model
    if load_model():
        print("✅ Starting Flask API server...")
        print("📡 API Endpoints:")
        print("   POST /predict - Single prediction (with date input)")
        print("   POST /batch_predict - Batch predictions (with date input)")
        print("   POST /date_info - Parse and analyze date input")
        print("   GET /model_info - Model information")
        print("   GET /health - Health check")
        
        print(f"\n🌐 Server Access URLs:")
        print(f"   Local:    http://127.0.0.1:5000")
        print(f"   Network:  http://{local_ip}:5000")
        print(f"   External: http://0.0.0.0:5000")
        
        print("\n📅 Date Input Features:")
        print("   • Multiple date formats supported")
        print("   • Automatic week/month derivation")
        print("   • ISO week number calculation")
        print("   • Weekend detection")
        print("   • Day of year calculation")
        
        print("\n🔗 For port forwarding, use:")
        print(f"   Target: {local_ip}:5000")
        print("=" * 70)
        
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
