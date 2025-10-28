# Quick Start Guide

Get the Mentorship Risk Prediction API running in 3 minutes!

## Step 1: Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

**Required packages:**
- Flask 3.0.0
- flask-cors 4.0.0
- numpy 1.24.3
- pandas 2.0.3
- scikit-learn 1.3.0

## Step 2: Create Model Files (Optional)

The API works in **fallback mode** without model files using heuristics.

To use the actual Random Forest model:

```bash
python create_model_files.py
```

This creates dummy models for testing. To use your real trained model:

1. Edit `create_model_files.py`
2. Update the `load_from_notebook_model()` function
3. Point it to your trained model from the Jupyter notebooks
4. Run the script again

## Step 3: Start the Server

```bash
python app.py
```

You should see:

```
==========================================================
Starting Mentorship Risk Prediction API
==========================================================

✓ Loaded Random Forest model
✓ Loaded StandardScaler
✓ Loaded feature columns (24 features)

Server Configuration:
  - Host: 0.0.0.0
  - Port: 5000
  - Debug: True
  - CORS: Enabled

Available Endpoints:
  - GET  http://localhost:5000/api/health
  - POST http://localhost:5000/api/predict
```

## Step 4: Test the API

### Option 1: Use the Test Script

In a **new terminal**:

```bash
python test_api.py
```

This runs a comprehensive test suite with 5 test cases.

### Option 2: Manual cURL Test

```bash
# Health check
curl http://localhost:5000/api/health

# Prediction
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "workfield": "Computer Science",
    "study_level": "Bac+3",
    "needs": "Professional",
    "registration_month": "July",
    "engagement_score": 0.8,
    "project_confidence_level": 3,
    "mentor_availability": 5,
    "previous_rejection": 0
  }'
```

### Option 3: Python Client

```python
import requests

response = requests.post('http://localhost:5000/api/predict', json={
    'workfield': 'Engineering',
    'study_level': 'Bac+5+',
    'needs': 'Both',
    'registration_month': 'November',
    'engagement_score': 2.5,
    'project_confidence_level': 5,
    'mentor_availability': 10,
    'previous_rejection': 0
})

print(response.json())
```

## Expected Response

```json
{
  "success": true,
  "prediction": {
    "responseRisk": 35,
    "matchQuality": 72,
    "motivationRisk": 18,
    "daysToFailure": 120
  },
  "input": {...},
  "model": "random_forest_calibrated",
  "timestamp": "2025-10-26T..."
}
```

## Understanding the Output

| Metric | Description | Good Range |
|--------|-------------|------------|
| **responseRisk** | Overall failure probability | 0-30% (low risk) |
| **matchQuality** | How well matched this pairing is | 70-100% (good match) |
| **motivationRisk** | Likelihood of ghosting/dropout | 0-30% (engaged) |
| **daysToFailure** | When failure might occur | 180+ days (stable) |

### Risk Levels

- 🔴 **CRITICAL** (80-100%): Intervention needed within 7 days
- 🟠 **HIGH** (60-79%): Close monitoring required
- 🟡 **MEDIUM** (40-59%): Standard check-ins
- 🟢 **LOW** (20-39%): Quarterly reviews
- ✅ **VERY LOW** (0-19%): Minimal oversight

## Common Issues

### Port Already in Use

If port 5000 is taken, edit `app.py`:

```python
app.run(host='0.0.0.0', port=5001, debug=True)  # Change port
```

### Model Files Not Found

The API will automatically use **fallback heuristic mode**. You'll see:

```
⚠ Model file not found. Using fallback prediction.
```

This is normal and the API will still work!

### CORS Errors

CORS is enabled for all origins. If you have issues, check the browser console.

## Next Steps

1. ✅ API is running
2. ✅ Tests pass
3. 📊 Integrate with your frontend (`demo3.html`)
4. 🎯 Replace dummy models with real trained models
5. 🚀 Deploy to production (use Gunicorn)

## Production Deployment

For production environments:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

Options:
- `-w 4`: Use 4 worker processes
- `-b 0.0.0.0:5000`: Bind to all interfaces on port 5000

## Getting Help

- 📖 Full docs: See `README.md`
- 🐛 Issues: Check server logs in the terminal
- 💬 Questions: Review the code comments in `app.py`

---

**Ready to predict mentorship risk!** 🎯
