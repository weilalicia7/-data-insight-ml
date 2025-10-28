# Mentorship Risk Prediction API

Flask REST API for predicting mentorship success/failure risk using Random Forest ML model.

## Features

- **Random Forest Model** with Bayesian calibration
- **Feature Transformation**: 8 input fields → 24 model features
- **CORS Enabled** for cross-origin requests
- **Full Error Handling** with detailed logging
- **Fallback Prediction** when model files are unavailable

## Installation

```bash
cd backend
pip install -r requirements.txt
```

## Running the Server

```bash
python app.py
```

Server will start at: `http://localhost:5000`

## API Endpoints

### 1. Health Check

```bash
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "model": "loaded",
  "scaler": "loaded",
  "features": 24,
  "version": "1.0.0",
  "timestamp": "2025-10-26T..."
}
```

### 2. Predict Risk

```bash
POST /api/predict
Content-Type: application/json
```

**Request Body:**
```json
{
  "workfield": "Computer Science",
  "study_level": "Bac+3",
  "needs": "Professional",
  "registration_month": "July",
  "engagement_score": 0.8,
  "project_confidence_level": 3,
  "mentor_availability": 5,
  "previous_rejection": 0
}
```

**Input Fields:**

| Field | Type | Description | Example Values |
|-------|------|-------------|----------------|
| `workfield` | string | Career field | Computer Science, Engineering, Business, Healthcare, Teaching, Other |
| `study_level` | string | Education level | Bac+1, Bac+2, Bac+3, Bac+4, Bac+5+ |
| `needs` | string | Mentorship focus | Professional, Academic, Both |
| `registration_month` | string | Month of registration | January - December |
| `engagement_score` | float | Engagement level | 0.0 - 3.0 |
| `project_confidence_level` | int | Self-confidence | 1 - 5 |
| `mentor_availability` | int | Hours/month | 0 - 20 |
| `previous_rejection` | int | Previously rejected | 0 or 1 |

**Response:**
```json
{
  "success": true,
  "prediction": {
    "responseRisk": 75,
    "matchQuality": 25,
    "motivationRisk": 68,
    "daysToFailure": 14
  },
  "input": {...},
  "model": "random_forest_calibrated",
  "timestamp": "2025-10-26T..."
}
```

**Output Metrics:**

| Metric | Range | Description |
|--------|-------|-------------|
| `responseRisk` | 0-100 | Overall failure risk percentage |
| `matchQuality` | 0-100 | Quality of mentor-mentee match |
| `motivationRisk` | 0-100 | Risk of ghosting/dropout |
| `daysToFailure` | 7-365+ | Estimated days until failure |

## Model Files

Place trained model artifacts in `backend/models/`:

```
backend/
├── models/
│   ├── random_forest_model.pkl     # Trained RandomForestClassifier
│   ├── scaler.pkl                  # Fitted StandardScaler
│   └── feature_columns.pkl         # List of 24 feature names in order
├── app.py
├── requirements.txt
└── README.md
```

### Creating Model Files

See `create_model_files.py` for an example of how to train and save the model artifacts.

## Example Usage

### cURL

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "workfield": "Computer Science",
    "study_level": "Bac+1",
    "needs": "Professional",
    "registration_month": "July",
    "engagement_score": 0.5,
    "project_confidence_level": 2,
    "mentor_availability": 3,
    "previous_rejection": 1
  }'
```

### Python

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

result = response.json()
print(f"Risk: {result['prediction']['responseRisk']}%")
```

### JavaScript (fetch)

```javascript
fetch('http://localhost:5000/api/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    workfield: 'Teaching',
    study_level: 'Bac+3',
    needs: 'Academic',
    registration_month: 'January',
    engagement_score: 2.0,
    project_confidence_level: 4,
    mentor_availability: 8,
    previous_rejection: 0
  })
})
.then(res => res.json())
.then(data => console.log(data.prediction));
```

## Feature Transformation

The API transforms 8 input fields into 24 model features:

1. **Numeric Features (4)**: engagement_score, project_confidence_level, mentor_availability, previous_rejection
2. **One-Hot Encoded (16)**: workfield (5), study_level (4), needs (2), registration_month (11) → 22 after drop_first
3. **Engineered Features (4)**: summer_registration, low_engagement, high_risk_field, engagement_confidence_interaction

Total: **24 features** passed to Random Forest model

## Model Performance

- **Accuracy**: 85.2%
- **Precision**: 82.1%
- **Recall**: 89.7%
- **F1-Score**: 0.857
- **ROC-AUC**: 0.91

## Error Handling

The API returns appropriate HTTP status codes:

- `200` - Success
- `400` - Bad Request (missing fields, invalid input)
- `404` - Endpoint not found
- `500` - Internal server error

## Fallback Mode

If model files are not found, the API automatically uses a heuristic-based fallback prediction using domain knowledge from the data analysis.

## CORS Configuration

CORS is enabled for all origins (`*`). For production, configure specific allowed origins in `app.py`:

```python
CORS(app, origins=['https://yourdomain.com'])
```

## Logging

The API logs all predictions and errors. Set logging level in `app.py`:

```python
logging.basicConfig(level=logging.INFO)  # or DEBUG, WARNING, ERROR
```

## Production Deployment

For production, use a WSGI server like Gunicorn:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## License

JP Morgan Data for Good Hackathon 2025 - Team 2
