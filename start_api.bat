@echo off
echo 🚀 Starting Demand Prediction API Server
echo ==========================================

cd /d "d:\Code_stuff\demand_prediction"

:: Activate .spark environment
call .spark\Scripts\activate.bat

:: Start the API server
echo Starting Flask API server on port 5000...
echo.
echo API Endpoints available:
echo   POST /predict - Single prediction
echo   POST /batch_predict - Batch predictions
echo   GET /model_info - Model information  
echo   GET /health - Health check
echo.
echo Server will be available at: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.

python api_server.py

pause
