@echo off
echo ===============================================
echo   Starting Demand Prediction API Server
echo   Date Input Version - v1.0
echo ===============================================
echo.

REM Check if virtual environment exists
if not exist ".spark" (
    echo ❌ Virtual environment '.spark' not found!
    echo Please create the virtual environment first.
    pause
    exit /b 1
)

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call .spark\Scripts\activate.bat

REM Check if required files exist
if not exist "breakthrough_model.pt" (
    echo ❌ Model file 'breakthrough_model.pt' not found!
    echo Please run breakthrough_model.py first to train the model.
    pause
    exit /b 1
)

if not exist "enhanced_demand_dataset.csv" (
    echo ❌ Dataset file 'enhanced_demand_dataset.csv' not found!
    echo Please ensure the dataset file is in the current directory.
    pause
    exit /b 1
)

echo.
echo ✅ All required files found
echo 🚀 Starting API server with date input support...
echo.

REM Start the Flask server
python api_server_date_input.py

REM Keep window open if server exits
echo.
echo Server has stopped. Press any key to exit...
pause > nul
