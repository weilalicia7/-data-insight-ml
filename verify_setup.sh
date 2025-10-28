#!/bin/bash

echo "========================================="
echo "  TEAM-2 PROJECT VERIFICATION"
echo "========================================="
echo ""

# 1. Check model files
echo "1. Checking model files..."
if [ -f "backend/models/random_forest_model.pkl" ] && [ -f "backend/models/scaler.pkl" ] && [ -f "backend/models/feature_columns.pkl" ]; then
    echo "   ✓ All 3 model files exist"
    ls -lh backend/models/
else
    echo "   ✗ Missing model files!"
fi
echo ""

# 2. Check API health
echo "2. Checking Flask API..."
if curl -s http://localhost:5000/api/health > /dev/null 2>&1; then
    echo "   ✓ API is responding"
    curl -s http://localhost:5000/api/health | python -m json.tool
else
    echo "   ✗ API is not running! Start with: cd backend && python app.py"
fi
echo ""

# 3. Check Python packages
echo "3. Checking Python packages..."
python -c "
import sys
try:
    import sklearn
    import flask
    import pandas
    import numpy
    print(f'   ✓ scikit-learn: {sklearn.__version__}')
    print(f'   ✓ Flask: {flask.__version__}')
    print(f'   ✓ pandas: {pandas.__version__}')
    print(f'   ✓ numpy: {numpy.__version__}')
except ImportError as e:
    print(f'   ✗ Missing package: {e}')
    sys.exit(1)
"
echo ""

# 4. Check data file
echo "4. Checking training data..."
if [ -f "backend/ml_ready_dataset.csv" ]; then
    echo "   ✓ Training data exists"
    python -c "import pandas as pd; df = pd.read_csv('backend/ml_ready_dataset.csv'); print(f'   Records: {len(df):,}, Success rate: {df.success.mean():.1%}')"
else
    echo "   ✗ Training data missing!"
fi
echo ""

# 5. Check notebook
echo "5. Checking Jupyter notebook..."
if [ -f "inspect_models.ipynb" ]; then
    echo "   ✓ Notebook exists"
else
    echo "   ✗ Notebook missing!"
fi
echo ""

# 6. Check frontend
echo "6. Checking frontend..."
if [ -f "frontend/demo_updated.html" ]; then
    echo "   ✓ Frontend demo exists"
    ls -lh frontend/demo_updated.html
else
    echo "   ✗ Frontend missing!"
fi
echo ""

echo "========================================="
echo "  VERIFICATION COMPLETE"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Run API tests:    cd backend && python test_api.py"
echo "  2. Open notebook:    jupyter notebook inspect_models.ipynb"
echo "  3. Open frontend:    open frontend/demo_updated.html"
echo ""
