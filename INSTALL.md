# Installation Guide - Demand Prediction Flask App

## Quick Start Options

### Option 1: Automated Setup (Recommended)
```bash
# Run the setup script to create environment and install everything
setup_env.bat
```

### Option 2: Manual Installation with CUDA Support
```bash
# Create virtual environment
python -m venv .spark

# Activate environment
.spark\Scripts\activate.bat

# Install PyTorch with CUDA support (for better performance)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install other requirements
pip install -r requirements.txt
```

### Option 3: CPU-Only Installation
```bash
# Create virtual environment
python -m venv .spark

# Activate environment
.spark\Scripts\activate.bat

# Install all requirements (will use CPU-only PyTorch)
pip install -r requirements.txt
```

## Running the Application

### Using the Start Script
```bash
start_app.bat
```

### Manual Start
```bash
# Activate environment
.spark\Scripts\activate.bat

# Run the Flask app
python app.py
```

## Requirements

### System Requirements
- Python 3.8 or higher
- Windows (for batch scripts) or modify for Linux/Mac

### For CUDA Support (Optional but Recommended)
- NVIDIA GPU with CUDA capability
- CUDA 12.8 or compatible version
- Sufficient GPU memory (4GB+ recommended)

### Package Versions

| Category | Package | Version | Purpose |
|----------|---------|---------|---------|
| **Web Framework** | Flask | 2.3.3 | Web application framework |
| **ML Core** | torch | 2.7.1+cu128 | Deep learning framework |
| | torchvision | 0.22.1+cu128 | Computer vision utilities |
| | torchaudio | 2.7.1+cu128 | Audio processing |
| | scikit-learn | 1.7.0 | Machine learning algorithms |
| | pandas | 2.3.0 | Data manipulation |
| | numpy | 2.3.1 | Numerical computing |
| **Visualization** | matplotlib | 3.10.3 | Plotting library |
| | seaborn | 0.13.2 | Statistical visualization |
| **Data Sources** | pytrends | 4.9.2 | Google Trends API |
| | requests | 2.32.4 | HTTP requests |

## Troubleshooting

### CUDA Issues
If CUDA installation fails or you don't have an NVIDIA GPU:
1. Use Option 3 (CPU-only installation)
2. The app will still work but model inference will be slower

### Memory Issues
- Ensure at least 4GB RAM available
- For large datasets, consider reducing batch sizes in the model

### Package Conflicts
- Use the provided virtual environment (.spark)
- Don't mix with other Python environments
- If issues persist, delete .spark folder and run setup_env.bat again

### Port Conflicts
If port 5000 is already in use, modify `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Change port number
```

## Verification

After installation, verify everything works:

1. **Check PyTorch Installation:**
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

2. **Test Model Loading:**
```python
python -c "from app import load_model; print('Model loaded:', load_model())"
```

3. **Run API Tests:**
```python
python test_api.py
```

## File Structure

```
demand_prediction/
├── .spark/                 # Virtual environment
├── templates/              # HTML templates
│   └── index.html         # Web interface
├── app.py                 # Flask application
├── breakthrough_model.py  # Model training
├── breakthrough_model.pt  # Trained model
├── requirements.txt       # All dependencies
├── requirements_optimized.txt  # Minimal dependencies
├── setup_env.bat         # Automated setup
├── start_app.bat         # Start application
├── test_api.py           # API testing
└── README_API.md         # API documentation
```
