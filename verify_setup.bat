@echo off
echo =========================================
echo   TEAM-2 PROJECT VERIFICATION
echo =========================================
echo.

REM 1. Check model files
echo 1. Checking model files...
if exist "backend\models\random_forest_model.pkl" (
    if exist "backend\models\scaler.pkl" (
        if exist "backend\models\feature_columns.pkl" (
            echo    [32m OK [0m All 3 model files exist
            dir backend\models\*.pkl
        )
    )
) else (
    echo    [31m ERROR [0m Missing model files!
)
echo.

REM 2. Check API health
echo 2. Checking Flask API...
curl -s http://localhost:5000/api/health >nul 2>&1
if %errorlevel% equ 0 (
    echo    [32m OK [0m API is responding
    curl -s http://localhost:5000/api/health
) else (
    echo    [31m ERROR [0m API is not running! Start with: cd backend ^&^& python app.py
)
echo.

REM 3. Check Python packages
echo 3. Checking Python packages...
python -c "import sys; import sklearn, flask, pandas, numpy; print(f'   [OK] scikit-learn: {sklearn.__version__}'); print(f'   [OK] Flask: {flask.__version__}'); print(f'   [OK] pandas: {pandas.__version__}'); print(f'   [OK] numpy: {numpy.__version__}')" 2>nul
if %errorlevel% neq 0 (
    echo    [31m ERROR [0m Missing packages! Run: pip install -r backend\requirements.txt
)
echo.

REM 4. Check data file
echo 4. Checking training data...
if exist "backend\ml_ready_dataset.csv" (
    echo    [32m OK [0m Training data exists
    python -c "import pandas as pd; df = pd.read_csv('backend/ml_ready_dataset.csv'); print(f'   Records: {len(df):,}, Success rate: {df.success.mean():.1%%}')"
) else (
    echo    [31m ERROR [0m Training data missing!
)
echo.

REM 5. Check notebook
echo 5. Checking Jupyter notebook...
if exist "inspect_models.ipynb" (
    echo    [32m OK [0m Notebook exists
) else (
    echo    [31m ERROR [0m Notebook missing!
)
echo.

REM 6. Check frontend
echo 6. Checking frontend...
if exist "frontend\demo_updated.html" (
    echo    [32m OK [0m Frontend demo exists
    dir frontend\demo_updated.html
) else (
    echo    [31m ERROR [0m Frontend missing!
)
echo.

echo =========================================
echo   VERIFICATION COMPLETE
echo =========================================
echo.
echo Next steps:
echo   1. Run API tests:    cd backend ^&^& python test_api.py
echo   2. Open notebook:    jupyter notebook inspect_models.ipynb
echo   3. Open frontend:    start frontend\demo_updated.html
echo.
pause
