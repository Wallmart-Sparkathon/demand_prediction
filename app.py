from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import warnings
import os
from datetime import datetime
import json

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

def get_unique_values():
    """Get unique values for categorical fields from the dataset"""
    try:
        df = pd.read_csv("enhanced_demand_dataset.csv")
        return {
            'products': sorted(df['Product Name'].unique().tolist()),
            'stores': sorted(df['Store Location'].unique().tolist()),
            'categories': sorted(df['Category'].unique().tolist())
        }
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return {
            'products': [],
            'stores': [],
            'categories': []
        }

def prepare_input_data(input_data):
    """Prepare input data for prediction"""
    try:
        # Create a DataFrame with the input
        df_input = pd.DataFrame([input_data])
        
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

@app.route('/')
def home():
    """Render the main page"""
    unique_values = get_unique_values()
    return render_template('index.html', **unique_values)

@app.route('/predict', methods=['POST'])
def predict():
    """Make a demand prediction"""
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
        required_fields = ['Week', 'Month', 'Product Name', 'Store Location', 
                          'Category', 'Promotion', 'Holiday', 'TrendScore', 
                          'DemandMultiplier', 'CityMultiplier']
        
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'error': f'Missing required field: {field}',
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
        
        return jsonify({
            'prediction': prediction,
            'input_data': data,
            'success': True,
            'message': f'Predicted demand: {prediction} units'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Make predictions for multiple inputs"""
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
                predictions.append({
                    'index': i,
                    'prediction': prediction,
                    'input': input_data
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

@app.route('/model_info')
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
            'success': True
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("🚀 Starting Demand Prediction Flask App")
    print("=" * 50)
    
    # Load the model
    if load_model():
        print("✅ Starting Flask server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to load model. Please run breakthrough_model.py first!")
        print("Exiting...")
