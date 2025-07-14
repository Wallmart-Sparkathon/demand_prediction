@echo off
echo 🚀 Starting Demand Prediction Flask App
echo =====================================

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

:: Check if .spark virtual environment exists
if not exist ".spark" (
    echo ❌ .spark virtual environment not found!
    echo Please ensure .spark directory exists with required packages
    pause
    exit /b 1
)

:: Activate .spark virtual environment
echo 🔧 Activating .spark virtual environment...
call .spark\Scripts\activate.bat

:: Install requirements (including PyTorch CUDA if not already installed)
echo 📋 Installing/updating requirements...
echo    Note: For better performance, install PyTorch with CUDA:
echo    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

:: Check if model file exists
if not exist "breakthrough_model.pt" (
    echo ❌ Model file not found!
    echo Please run breakthrough_model.py first to train the model
    pause
    exit /b 1
)

:: Start the Flask app
echo 🌟 Starting Flask application...
echo.
echo The app will be available at: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.
python app.py

pause
