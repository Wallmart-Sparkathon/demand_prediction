@echo off
echo 🚀 Setting up Demand Prediction Environment
echo ==========================================

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

:: Check if .spark environment exists, if not create one
if not exist ".spark" (
    echo 📦 Creating .spark virtual environment...
    python -m venv .spark
)

:: Activate .spark virtual environment
echo 🔧 Activating .spark virtual environment...
call .spark\Scripts\activate.bat

:: Upgrade pip
echo 📋 Upgrading pip...
python -m pip install --upgrade pip

:: Install PyTorch with CUDA support
echo 🔥 Installing PyTorch with CUDA 12.8 support...
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

:: Install other requirements
echo 📋 Installing other requirements...
pip install -r requirements_optimized.txt

:: Verify PyTorch installation
echo 🔍 Verifying PyTorch installation...
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

echo ✅ Environment setup complete!
echo.
echo To start the app, run: start_app.bat
pause
