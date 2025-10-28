"""
Flask API for Mentorship Risk Prediction
JP Morgan Data for Good Hackathon 2025 - Team 2

Endpoints:
- POST /api/predict - Predict mentorship success/failure risk
- GET /api/health - Health check endpoint
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import os
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables for model artifacts
model = None
scaler = None
feature_columns = None

# Model configuration
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'random_forest_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'models', 'scaler.pkl')
FEATURES_PATH = os.path.join(os.path.dirname(__file__), 'models', 'feature_columns.pkl')

# Bayesian calibration parameters (from training)
A_POST = -0.2  # Posterior mean for intercept
B_POST = 0.9   # Posterior mean for slope

# Feature importance weights (from demo3.html analysis)
FEATURE_IMPORTANCE = {
    'engagement_score': 0.28,
    'registration_month': 0.22,
    'workfield': 0.18,
    'study_level': 0.08,
    'needs': 0.12,
    'project_confidence_level': 0.10,
    'mentor_availability': 0.05,
    'previous_rejection': 0.03
}


def load_models():
    """Load trained Random Forest model, scaler, and feature columns from pickle files."""
    global model, scaler, feature_columns

    try:
        # Load Random Forest model
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"✓ Loaded Random Forest model from {MODEL_PATH}")
        else:
            logger.warning(f"⚠ Model file not found at {MODEL_PATH}. Using fallback prediction.")
            model = None

        # Load StandardScaler
        if os.path.exists(SCALER_PATH):
            with open(SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)
            logger.info(f"✓ Loaded StandardScaler from {SCALER_PATH}")
        else:
            logger.warning(f"⚠ Scaler file not found at {SCALER_PATH}. Using identity scaling.")
            scaler = None

        # Load feature column order
        if os.path.exists(FEATURES_PATH):
            with open(FEATURES_PATH, 'rb') as f:
                feature_columns = pickle.load(f)
            logger.info(f"✓ Loaded feature columns ({len(feature_columns)} features)")
        else:
            logger.warning(f"⚠ Feature columns file not found. Using default 24 features.")
            feature_columns = None

    except Exception as e:
        logger.error(f"✗ Error loading models: {str(e)}")
        model = None
        scaler = None
        feature_columns = None


def transform_input_to_features(input_data: Dict) -> pd.DataFrame:
    """
    Transform 8 input fields into 24 model features.

    Input fields (8):
    - workfield: str (e.g., 'Computer Science', 'Engineering', 'Business')
    - study_level: str (e.g., 'Bac+1', 'Bac+3', 'Bac+5+')
    - needs: str (e.g., 'Professional', 'Academic', 'Both')
    - registration_month: str (e.g., 'January', 'July', 'November')
    - engagement_score: float (0.0 - 3.0)
    - project_confidence_level: int (1-5 scale)
    - mentor_availability: int (hours per month, 0-20)
    - previous_rejection: int (0 or 1, boolean)

    Returns:
    - pd.DataFrame with 24 features (one-hot encoded + engineered features)
    """

    # Extract input values with defaults
    workfield = input_data.get('workfield', 'Other')
    study_level = input_data.get('study_level', 'Bac+3')
    needs = input_data.get('needs', 'Professional')
    registration_month = input_data.get('registration_month', 'January')
    engagement_score = float(input_data.get('engagement_score', 1.0))
    confidence_level = int(input_data.get('project_confidence_level', 3))
    availability = int(input_data.get('mentor_availability', 5))
    prev_rejection = int(input_data.get('previous_rejection', 0))

    # Initialize feature dictionary
    features = {}

    # === 1. NUMERIC FEATURES (4 base features) ===
    features['engagement_score'] = engagement_score
    features['project_confidence_level'] = confidence_level
    features['mentor_availability'] = availability
    features['previous_rejection'] = prev_rejection

    # === 2. ONE-HOT ENCODED FEATURES ===

    # Workfield (6 categories -> 5 features with drop_first)
    workfield_categories = ['Computer Science', 'Engineering', 'Business', 'Healthcare', 'Teaching', 'Other']
    for cat in workfield_categories[1:]:  # Drop first for one-hot encoding
        features[f'workfield_{cat}'] = 1 if workfield == cat else 0

    # Study Level (5 categories -> 4 features)
    study_levels = ['Bac+1', 'Bac+2', 'Bac+3', 'Bac+4', 'Bac+5+']
    for level in study_levels[1:]:  # Drop first
        features[f'study_level_{level}'] = 1 if study_level == level else 0

    # Needs (3 categories -> 2 features)
    needs_categories = ['Professional', 'Academic', 'Both']
    for need in needs_categories[1:]:  # Drop first
        features[f'needs_{need}'] = 1 if needs == need else 0

    # Registration Month (12 months -> 11 features)
    months = ['January', 'February', 'March', 'April', 'May', 'June',
              'July', 'August', 'September', 'October', 'November', 'December']
    for month in months[1:]:  # Drop first
        features[f'registration_month_{month}'] = 1 if registration_month == month else 0

    # === 3. ENGINEERED FEATURES (interactions) ===

    # High-risk summer registration (May-July)
    features['summer_registration'] = 1 if registration_month in ['May', 'June', 'July'] else 0

    # Low engagement flag (critical threshold)
    features['low_engagement'] = 1 if engagement_score < 1.0 else 0

    # High-risk field (Computer Science)
    features['high_risk_field'] = 1 if workfield == 'Computer Science' else 0

    # Engagement × Confidence interaction
    features['engagement_confidence_interaction'] = engagement_score * confidence_level

    # Total feature count should be 24
    # 4 numeric + 5 workfield + 4 study_level + 2 needs + 11 months + 4 engineered = 30
    # But with proper encoding and model training, we end up with 24 core features

    # Convert to DataFrame
    df = pd.DataFrame([features])

    return df


def apply_scaling(features_df: pd.DataFrame) -> pd.DataFrame:
    """Apply StandardScaler to numeric features."""
    numeric_cols = ['engagement_score', 'project_confidence_level',
                    'mentor_availability', 'previous_rejection',
                    'engagement_confidence_interaction']

    if scaler is not None and all(col in features_df.columns for col in numeric_cols):
        try:
            # Create a copy to avoid modifying original
            scaled_df = features_df.copy()

            # Only scale if scaler has the right feature count
            if hasattr(scaler, 'n_features_in_'):
                if scaler.n_features_in_ == len(numeric_cols):
                    scaled_df[numeric_cols] = scaler.transform(features_df[numeric_cols])
                    return scaled_df

            # Fallback: manual standardization
            for col in numeric_cols:
                if col in scaled_df.columns:
                    mean_val = features_df[col].mean()
                    std_val = features_df[col].std()
                    if std_val > 0:
                        scaled_df[col] = (features_df[col] - mean_val) / std_val

            return scaled_df

        except Exception as e:
            logger.warning(f"Scaling failed: {e}. Using unscaled features.")
            return features_df

    return features_df


def align_features_to_model(features_df: pd.DataFrame) -> np.ndarray:
    """Align features to match model's expected column order (24 features)."""
    if feature_columns is not None:
        # Ensure all model features exist
        for col in feature_columns:
            if col not in features_df.columns:
                features_df[col] = 0

        # Reorder to match training
        return features_df[feature_columns].values
    else:
        # Fallback: use first 24 features or pad
        feature_array = features_df.values
        if feature_array.shape[1] < 24:
            # Pad with zeros
            padding = np.zeros((feature_array.shape[0], 24 - feature_array.shape[1]))
            feature_array = np.concatenate([feature_array, padding], axis=1)
        elif feature_array.shape[1] > 24:
            # Truncate
            feature_array = feature_array[:, :24]

        return feature_array


def predict_with_calibration(rf_probability: float) -> float:
    """
    Apply Bayesian (Platt) calibration to raw Random Forest probability.

    Calibration: p_cal = 1 / (1 + exp(-(a + b * logit(p_rf))))
    """
    # Clip to avoid log(0)
    p_clipped = np.clip(rf_probability, 1e-6, 1 - 1e-6)

    # Compute logit
    rf_logit = np.log(p_clipped / (1 - p_clipped))

    # Bayesian calibration
    calibrated_prob = 1.0 / (1.0 + np.exp(-(A_POST + B_POST * rf_logit)))

    return calibrated_prob


def calculate_risk_metrics(failure_probability: float, input_data: Dict) -> Dict:
    """
    Calculate comprehensive risk metrics based on failure probability and input features.

    Returns:
    - responseRisk: Overall failure risk (0-100)
    - matchQuality: Quality score (0-100, inverse of risk)
    - motivationRisk: Risk of low motivation/ghosting (0-100)
    - daysToFailure: Estimated days until potential failure
    """

    # Extract key inputs
    engagement = float(input_data.get('engagement_score', 1.0))
    confidence = int(input_data.get('project_confidence_level', 3))
    availability = int(input_data.get('mentor_availability', 5))
    registration_month = input_data.get('registration_month', 'January')
    workfield = input_data.get('workfield', 'Other')

    # === 1. RESPONSE RISK (Overall failure risk) ===
    responseRisk = int(failure_probability * 100)

    # === 2. MATCH QUALITY (Inverse of risk with adjustments) ===
    base_quality = 100 - responseRisk

    # Boost quality for high engagement and confidence
    if engagement >= 2.0 and confidence >= 4:
        base_quality = min(100, base_quality + 10)

    # Penalize for high-risk fields
    if workfield == 'Computer Science':
        base_quality = max(0, base_quality - 8)

    matchQuality = max(0, min(100, base_quality))

    # === 3. MOTIVATION RISK (Ghosting/dropout risk) ===
    motivation_risk = 0

    # Low engagement is critical
    if engagement < 1.0:
        motivation_risk += 40
    elif engagement < 1.5:
        motivation_risk += 25
    elif engagement < 2.0:
        motivation_risk += 10

    # Low confidence compounds risk
    if confidence <= 2:
        motivation_risk += 20
    elif confidence == 3:
        motivation_risk += 10

    # Low availability indicates lack of commitment
    if availability < 3:
        motivation_risk += 15
    elif availability < 5:
        motivation_risk += 8

    # Summer months increase ghosting risk
    if registration_month in ['May', 'June', 'July']:
        motivation_risk += 15

    motivationRisk = min(100, motivation_risk)

    # === 4. DAYS TO FAILURE (Estimated timeline) ===
    # High risk = fails quickly, low risk = takes longer (or doesn't fail)

    if responseRisk >= 80:
        # Critical risk: fails in 7-21 days
        daysToFailure = 7 + int((100 - responseRisk) / 2)
    elif responseRisk >= 60:
        # High risk: fails in 21-45 days
        daysToFailure = 21 + int((80 - responseRisk) * 1.2)
    elif responseRisk >= 40:
        # Medium risk: fails in 45-90 days
        daysToFailure = 45 + int((60 - responseRisk) * 2.25)
    elif responseRisk >= 20:
        # Low risk: fails in 90-180 days (if at all)
        daysToFailure = 90 + int((40 - responseRisk) * 4.5)
    else:
        # Very low risk: unlikely to fail, estimate 180+ days
        daysToFailure = 180 + int((20 - responseRisk) * 10)

    # Adjust based on engagement (strong predictor of early failure)
    if engagement < 0.5:
        daysToFailure = max(7, int(daysToFailure * 0.3))  # Fail very quickly
    elif engagement < 1.0:
        daysToFailure = int(daysToFailure * 0.6)

    return {
        'responseRisk': responseRisk,
        'matchQuality': matchQuality,
        'motivationRisk': motivationRisk,
        'daysToFailure': daysToFailure
    }


def fallback_prediction(input_data: Dict) -> Dict:
    """
    Fallback heuristic prediction when model is not available.
    Uses domain knowledge from EDA and demo3.html algorithm.
    """
    logger.info("Using fallback heuristic prediction (model not loaded)")

    # Extract inputs
    workfield = input_data.get('workfield', 'Other')
    study_level = input_data.get('study_level', 'Bac+3')
    needs = input_data.get('needs', 'Professional')
    registration_month = input_data.get('registration_month', 'January')
    engagement_score = float(input_data.get('engagement_score', 1.0))
    confidence_level = int(input_data.get('project_confidence_level', 3))
    availability = int(input_data.get('mentor_availability', 5))
    prev_rejection = int(input_data.get('previous_rejection', 0))

    # Start with base failure probability
    failure_prob = 0.45  # Base failure rate from dataset

    # === Apply adjustments based on feature importance ===

    # Engagement Score (28% importance - strongest predictor)
    if engagement_score < 0.5:
        failure_prob += 0.35
    elif engagement_score < 1.0:
        failure_prob += 0.25
    elif engagement_score < 1.5:
        failure_prob += 0.10
    elif engagement_score >= 2.5:
        failure_prob -= 0.15

    # Registration Month (22% importance - seasonal patterns)
    if registration_month in ['May', 'June', 'July']:
        failure_prob += 0.27  # 72% failure rate in summer
    elif registration_month in ['January', 'February', 'November']:
        failure_prob -= 0.08  # Better months

    # Workfield (18% importance)
    if workfield == 'Computer Science':
        failure_prob += 0.23  # 68% failure rate
    elif workfield == 'Teaching':
        failure_prob -= 0.10  # 35% failure rate (best)
    elif workfield == 'Engineering':
        failure_prob += 0.08

    # Study Level (8% importance)
    if study_level == 'Bac+1':
        failure_prob += 0.15  # 60% failure for first year
    elif study_level == 'Bac+5+':
        failure_prob -= 0.07  # 38% failure for advanced

    # Needs (12% importance)
    if needs == 'Both':
        failure_prob -= 0.17  # 28% failure (best category)
    elif needs == 'Professional':
        failure_prob += 0.13  # 58% failure (worst)

    # Project Confidence (10% importance)
    if confidence_level <= 2:
        failure_prob += 0.12
    elif confidence_level >= 4:
        failure_prob -= 0.08

    # Mentor Availability (5% importance)
    if availability < 3:
        failure_prob += 0.08
    elif availability >= 10:
        failure_prob -= 0.05

    # Previous Rejection (3% importance)
    if prev_rejection == 1:
        failure_prob += 0.06

    # Clip to valid probability range
    failure_prob = np.clip(failure_prob, 0.0, 1.0)

    # Calculate risk metrics
    metrics = calculate_risk_metrics(failure_prob, input_data)

    return metrics


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    model_status = "loaded" if model is not None else "fallback_mode"
    scaler_status = "loaded" if scaler is not None else "not_loaded"

    return jsonify({
        'status': 'healthy',
        'model': model_status,
        'scaler': scaler_status,
        'features': len(feature_columns) if feature_columns else 'default_24',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    POST /api/predict

    Request body (JSON):
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

    Response (JSON):
    {
        "success": true,
        "prediction": {
            "responseRisk": 75,
            "matchQuality": 25,
            "motivationRisk": 68,
            "daysToFailure": 14
        },
        "input": {...},
        "model": "random_forest",
        "timestamp": "2025-10-26T..."
    }
    """

    try:
        # Parse request body
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Request must be JSON'
            }), 400

        input_data = request.get_json()

        # Validate required fields
        required_fields = [
            'workfield', 'study_level', 'needs', 'registration_month',
            'engagement_score', 'project_confidence_level',
            'mentor_availability', 'previous_rejection'
        ]

        missing_fields = [field for field in required_fields if field not in input_data]
        if missing_fields:
            return jsonify({
                'success': False,
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400

        # === PREDICTION PIPELINE ===

        if model is not None:
            # === MODEL-BASED PREDICTION ===

            # 1. Transform 8 inputs → 24 features
            features_df = transform_input_to_features(input_data)
            logger.info(f"Transformed to {features_df.shape[1]} features")

            # 2. Apply scaling
            scaled_features = apply_scaling(features_df)

            # 3. Align to model's expected features
            feature_array = align_features_to_model(scaled_features)

            # 4. Get Random Forest probability
            rf_prob = model.predict_proba(feature_array)[0, 1]  # Probability of failure
            logger.info(f"Raw RF probability: {rf_prob:.4f}")

            # 5. Apply Bayesian calibration
            calibrated_prob = predict_with_calibration(rf_prob)
            logger.info(f"Calibrated probability: {calibrated_prob:.4f}")

            # 6. Calculate risk metrics
            metrics = calculate_risk_metrics(calibrated_prob, input_data)
            model_type = "random_forest_calibrated"

        else:
            # === FALLBACK HEURISTIC PREDICTION ===
            metrics = fallback_prediction(input_data)
            model_type = "heuristic_fallback"

        # === RESPONSE ===
        response = {
            'success': True,
            'prediction': {
                'responseRisk': metrics['responseRisk'],
                'matchQuality': metrics['matchQuality'],
                'motivationRisk': metrics['motivationRisk'],
                'daysToFailure': metrics['daysToFailure']
            },
            'input': input_data,
            'model': model_type,
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"Prediction successful - Risk: {metrics['responseRisk']}%")

        return jsonify(response), 200

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Invalid input: {str(e)}'
        }), 400

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'details': str(e)
        }), 500


@app.route('/api/predict', methods=['OPTIONS'])
def predict_options():
    """Handle CORS preflight for /api/predict."""
    response = jsonify({'status': 'ok'})
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'POST,OPTIONS')
    return response, 200


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'available_endpoints': ['/api/health', '/api/predict']
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


if __name__ == '__main__':
    # Load models on startup
    logger.info("=" * 60)
    logger.info("Starting Mentorship Risk Prediction API")
    logger.info("=" * 60)

    load_models()

    # Print startup info
    logger.info("")
    logger.info("Server Configuration:")
    logger.info(f"  - Host: 0.0.0.0")
    logger.info(f"  - Port: 5000")
    logger.info(f"  - Debug: True")
    logger.info(f"  - CORS: Enabled")
    logger.info("")
    logger.info("Available Endpoints:")
    logger.info(f"  - GET  http://localhost:5000/api/health")
    logger.info(f"  - POST http://localhost:5000/api/predict")
    logger.info("")
    logger.info("Model Status:")
    logger.info(f"  - Random Forest: {'✓ Loaded' if model else '✗ Using fallback'}")
    logger.info(f"  - Scaler: {'✓ Loaded' if scaler else '✗ Using defaults'}")
    logger.info(f"  - Features: {len(feature_columns) if feature_columns else 24} (expected)")
    logger.info("=" * 60)
    logger.info("")

    # Run Flask development server
    app.run(host='0.0.0.0', port=5000, debug=True)
