@echo off
echo 🚀 Starting Demand Prediction API Server (Enhanced)
echo ====================================================

cd /d "d:\Code_stuff\demand_prediction"

:: Activate .spark environment
call .spark\Scripts\activate.bat

:: Check if port 5000 is available
echo 🔍 Checking port availability...
netstat -an | find "5000" >nul
if %errorlevel% == 0 (
    echo ⚠️  Port 5000 is already in use
    echo Trying to start on port 8000...
) else (
    echo ✅ Port 5000 is available
)

:: Display network information
echo.
echo 🌐 Network Information:
ipconfig | findstr /R /C:"IPv4.*192\." /C:"IPv4.*10\." /C:"IPv4.*172\."
echo.

:: Start the enhanced API server
echo 🚀 Starting Enhanced Flask API server...
echo.
echo 📡 API Endpoints:
echo   POST /predict - Single prediction
echo   POST /batch_predict - Batch predictions
echo   GET /model_info - Model information  
echo   GET /health - Health check
echo.
echo 🔗 Port Forwarding Tips:
echo   - Use 0.0.0.0:5000 as target
echo   - Ensure Windows Firewall allows port 5000
echo   - Check if antivirus is blocking the port
echo.
echo ⏹️  Press Ctrl+C to stop the server
echo ====================================================

python api_server_enhanced.py

pause
