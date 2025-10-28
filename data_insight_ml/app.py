"""
Flask API for ML Predictions
Provides REST API endpoints for making predictions with trained models
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
import yaml
from datetime import datetime
import logging
import os
import io
from werkzeug.utils import secure_filename
from domain_manager import DomainManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables
model = None
scaler = None
feature_columns = None
model_info = None
domain_manager = None


def load_config():
    """Load configuration"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def load_model_artifacts():
    """Load trained model, scaler, and feature columns"""
    global model, scaler, feature_columns, model_info

    models_dir = 'models'

    try:
        # Load model
        model_path = os.path.join(models_dir, 'best_model.pkl')
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info(f" Loaded model from {model_path}")
        else:
            logger.warning(f" Model file not found: {model_path}")
            return False

        # Load scaler
        scaler_path = os.path.join(models_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            logger.info(f" Loaded scaler from {scaler_path}")

        # Load feature columns
        features_path = os.path.join(models_dir, 'feature_columns.pkl')
        if os.path.exists(features_path):
            with open(features_path, 'rb') as f:
                feature_columns = pickle.load(f)
            logger.info(f" Loaded {len(feature_columns)} feature columns")

        # Load model info
        info_path = os.path.join(models_dir, 'model_info.yaml')
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                model_info = yaml.safe_load(f)
            logger.info(f" Loaded model info")

        return True

    except Exception as e:
        logger.error(f" Error loading model artifacts: {str(e)}")
        return False


def preprocess_input(input_data):
    """
    Preprocess input data to match training format
    """
    try:
        # Convert to DataFrame
        df = pd.DataFrame([input_data])

        # Ensure all expected features are present
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0

        # Reorder columns to match training
        df = df[feature_columns]

        # Scale features
        if scaler is not None:
            df_scaled = scaler.transform(df)
        else:
            df_scaled = df.values

        return df_scaled

    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}")
        raise


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    model_status = "loaded" if model is not None else "not_loaded"
    scaler_status = "loaded" if scaler is not None else "not_loaded"

    response = {
        'status': 'healthy',
        'model': model_status,
        'scaler': scaler_status,
        'features': len(feature_columns) if feature_columns else 0,
        'timestamp': datetime.now().isoformat()
    }

    if model_info:
        response['model_info'] = model_info

    return jsonify(response), 200


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Make predictions

    Request body (JSON):
    {
        "feature1": value1,
        "feature2": value2,
        ...
    }

    Response:
    {
        "success": true,
        "prediction": prediction_value,
        "probability": [prob_class_0, prob_class_1],  # if available
        "confidence": confidence_score,
        "input": input_data,
        "timestamp": "2025-..."
    }
    """

    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please train a model first.'
            }), 500

        # Parse request
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Request must be JSON'
            }), 400

        input_data = request.get_json()

        # Preprocess input
        try:
            X = preprocess_input(input_data)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Input preprocessing failed: {str(e)}'
            }), 400

        # Make prediction
        prediction = model.predict(X)[0]

        # Get probability if available
        probability = None
        confidence = None

        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)[0]
            probability = proba.tolist()
            confidence = float(max(proba))

        # Format response
        response = {
            'success': True,
            'prediction': int(prediction) if isinstance(prediction, (np.integer, int)) else str(prediction),
            'input': input_data,
            'timestamp': datetime.now().isoformat()
        }

        if probability:
            response['probability'] = probability
            response['confidence'] = confidence

        logger.info(f"Prediction made: {prediction} (confidence: {confidence})")

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'details': str(e)
        }), 500


@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """
    Make predictions for multiple inputs

    Request body (JSON):
    {
        "data": [
            {"feature1": value1, "feature2": value2, ...},
            {"feature1": value1, "feature2": value2, ...},
            ...
        ]
    }
    """

    try:
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded'
            }), 500

        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Request must be JSON'
            }), 400

        data = request.get_json()

        if 'data' not in data or not isinstance(data['data'], list):
            return jsonify({
                'success': False,
                'error': 'Request must contain "data" array'
            }), 400

        predictions = []

        for input_data in data['data']:
            try:
                X = preprocess_input(input_data)
                prediction = model.predict(X)[0]

                result = {
                    'prediction': int(prediction) if isinstance(prediction, (np.integer, int)) else str(prediction),
                    'input': input_data
                }

                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)[0]
                    result['probability'] = proba.tolist()
                    result['confidence'] = float(max(proba))

                predictions.append(result)

            except Exception as e:
                predictions.append({
                    'error': str(e),
                    'input': input_data
                })

        response = {
            'success': True,
            'predictions': predictions,
            'count': len(predictions),
            'timestamp': datetime.now().isoformat()
        }

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/features', methods=['GET'])
def get_features():
    """Get list of expected features"""
    if feature_columns is None:
        return jsonify({
            'success': False,
            'error': 'Features not loaded'
        }), 500

    return jsonify({
        'success': True,
        'features': feature_columns,
        'count': len(feature_columns)
    }), 200


# Multi-Domain Endpoints

@app.route('/api/domains', methods=['GET'])
def list_domains():
    """List all available domains"""
    try:
        if domain_manager is None:
            return jsonify({
                'success': False,
                'error': 'Domain manager not initialized'
            }), 500

        domains = domain_manager.list_domains()
        return jsonify({
            'success': True,
            'domains': domains,
            'count': len(domains)
        }), 200

    except Exception as e:
        logger.error(f"Error listing domains: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/domain/<domain_name>/predict', methods=['POST'])
def domain_predict(domain_name):
    """Make prediction using specific domain"""
    try:
        if domain_manager is None:
            return jsonify({
                'success': False,
                'error': 'Domain manager not initialized'
            }), 500

        # Load domain if not already loaded
        if domain_manager.current_domain != domain_name:
            try:
                domain_manager.load_domain(domain_name)
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': f'Failed to load domain: {str(e)}'
                }), 400

        # Get input data
        input_data = request.get_json()

        # Make prediction
        result = domain_manager.predict(input_data)

        return jsonify({
            'success': True,
            **result
        }), 200

    except Exception as e:
        logger.error(f"Domain prediction error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/domain/<domain_name>/demo', methods=['GET'])
def domain_demo(domain_name):
    """Get demo prediction for specific domain"""
    try:
        if domain_manager is None:
            return jsonify({
                'success': False,
                'error': 'Domain manager not initialized'
            }), 500

        # Load domain
        if domain_manager.current_domain != domain_name:
            domain_manager.load_domain(domain_name)

        # Get example data
        examples = domain_manager.get_example_data(domain_name, n_rows=1)

        if not examples:
            return jsonify({
                'success': False,
                'error': 'No demo data available'
            }), 404

        # Remove ID columns and target from demo data
        demo_data = examples[0].copy()
        id_cols = [col for col in demo_data.keys() if 'id' in col.lower()]
        target_col = domain_manager.current_model_data['metadata'].get('target_variable')

        for col in id_cols + [target_col]:
            demo_data.pop(col, None)

        # Make prediction
        result = domain_manager.predict(demo_data)

        return jsonify({
            'success': True,
            'demo_data': demo_data,
            **result
        }), 200

    except Exception as e:
        logger.error(f"Demo prediction error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/upload-predict', methods=['POST'])
def upload_predict():
    """Upload CSV and get batch predictions"""
    try:
        if domain_manager is None:
            return jsonify({
                'success': False,
                'error': 'Domain manager not initialized'
            }), 500

        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400

        file = request.files['file']
        domain_name = request.form.get('domain')

        if not domain_name:
            return jsonify({
                'success': False,
                'error': 'Domain not specified'
            }), 400

        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400

        if not file.filename.endswith('.csv'):
            return jsonify({
                'success': False,
                'error': 'Only CSV files are supported'
            }), 400

        # Load domain
        if domain_manager.current_domain != domain_name:
            domain_manager.load_domain(domain_name)

        # Read CSV
        try:
            df = pd.read_csv(io.BytesIO(file.read()))
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Failed to read CSV: {str(e)}'
            }), 400

        # Preserve ID and name columns before processing
        # ID columns: end with '_id' or are exactly 'id'
        id_cols = [col for col in df.columns if col.lower().endswith('_id') or col.lower() == 'id']
        # Name columns: end with '_name' or are exactly 'name'
        name_cols = [col for col in df.columns if col.lower().endswith('_name') or col.lower() == 'name']
        identifier_cols = id_cols + name_cols

        # Save identifiers for each row
        identifiers = []
        for idx, row in df.head(100).iterrows():
            row_id = {}
            for col in identifier_cols:
                if col in df.columns:
                    # Convert NaN to None for JSON serialization
                    val = row[col]
                    if pd.isna(val):
                        val = None
                    row_id[col] = val
            identifiers.append(row_id)

        # Remove ID, name, and target columns for prediction
        target_col = domain_manager.current_model_data['metadata'].get('target_variable')
        df_features = df.drop(columns=identifier_cols + [target_col], errors='ignore')

        # Make predictions for each row
        results = []
        for i, (_, row) in enumerate(df_features.head(100).iterrows()):  # Limit to 100 rows
            try:
                input_data = row.to_dict()
                result = domain_manager.predict(input_data)

                # Add identifier information to result
                if i < len(identifiers):
                    result['identifiers'] = identifiers[i]

                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting row: {str(e)}")
                error_result = {'error': str(e)}
                if i < len(identifiers):
                    error_result['identifiers'] = identifiers[i]
                results.append(error_result)

        return jsonify({
            'success': True,
            'results': results,
            'count': len(results),
            'domain': domain_name
        }), 200

    except Exception as e:
        logger.error(f"Upload prediction error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/analyze-csv', methods=['POST'])
def analyze_csv():
    """Analyze CSV for data quality issues"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400

        file = request.files['file']
        domain_name = request.form.get('domain')

        if not file.filename.endswith('.csv'):
            return jsonify({'success': False, 'error': 'Only CSV files supported'}), 400

        # Read CSV
        df = pd.read_csv(io.BytesIO(file.read()))

        # Analyze data quality
        analysis = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': {},
            'duplicates': int(df.duplicated().sum()),
            'outliers': {},
            'invalid_values': {},
            'categorical_issues': {},
            'suggestions': []
        }

        # Check missing values
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                missing_pct = (missing_count / len(df)) * 100
                analysis['missing_values'][col] = {
                    'count': int(missing_count),
                    'percentage': round(missing_pct, 1)
                }
                analysis['suggestions'].append({
                    'type': 'missing_values',
                    'column': col,
                    'issue': f'{missing_count} missing values ({missing_pct:.1f}%)',
                    'action': f'Fill with {"median" if df[col].dtype in ["float64", "int64"] else "mode"}',
                    'recommended': True
                })

        # Check duplicates
        if analysis['duplicates'] > 0:
            analysis['suggestions'].append({
                'type': 'duplicates',
                'issue': f'{analysis["duplicates"]} duplicate rows found',
                'action': 'Remove duplicate rows',
                'recommended': True
            })

        # Check for outliers in numeric columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            if 'id' in col.lower():
                continue
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < Q1 - 3*IQR) | (df[col] > Q3 + 3*IQR)).sum()
            if outliers > 0:
                analysis['outliers'][col] = int(outliers)
                analysis['suggestions'].append({
                    'type': 'outliers',
                    'column': col,
                    'issue': f'{outliers} extreme outliers detected',
                    'action': 'Cap outliers using IQR method',
                    'recommended': True
                })

        # Check for invalid ranges
        range_checks = {
            'age': (0, 120),
            'gpa': (0, 4.0),
            'attendance': (0, 1.0),
            'rate': (0, 1.0)
        }
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                for key, (min_val, max_val) in range_checks.items():
                    if key in col.lower():
                        invalid = ((df[col] < min_val) | (df[col] > max_val)).sum()
                        if invalid > 0:
                            analysis['invalid_values'][col] = int(invalid)
                            analysis['suggestions'].append({
                                'type': 'invalid_range',
                                'column': col,
                                'issue': f'{invalid} values outside valid range [{min_val}, {max_val}]',
                                'action': f'Clip values to valid range',
                                'recommended': True
                            })

        # Check categorical inconsistencies
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if 'id' in col.lower() or 'name' in col.lower():
                continue
            unique_count = df[col].nunique()
            if 3 <= unique_count <= 10:  # Likely categorical
                # Check for case inconsistencies
                values = df[col].dropna().unique()
                lower_values = [str(v).lower() for v in values]
                if len(lower_values) != len(set(lower_values)):
                    analysis['categorical_issues'][col] = 'Inconsistent formatting'
                    analysis['suggestions'].append({
                        'type': 'categorical_format',
                        'column': col,
                        'issue': 'Inconsistent case formatting (e.g., Low/low/LOW)',
                        'action': 'Standardize to consistent case',
                        'recommended': True
                    })

        # Add derived features suggestion
        if domain_name:
            analysis['suggestions'].append({
                'type': 'derived_features',
                'issue': 'Raw features only',
                'action': f'Add {domain_name}-specific derived features (risk scores, categories, etc.)',
                'recommended': True
            })

        # Calculate overall quality score
        completeness = 1 - (sum([v['count'] for v in analysis['missing_values'].values()]) / (len(df) * len(df.columns)))
        analysis['quality_score'] = round(completeness * 100, 1)

        return jsonify({
            'success': True,
            'analysis': analysis
        }), 200

    except Exception as e:
        logger.error(f"CSV analysis error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/clean-and-predict', methods=['POST'])
def clean_and_predict():
    """Clean CSV with approved suggestions and generate predictions"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400

        file = request.files['file']
        domain_name = request.form.get('domain')
        approved_suggestions = request.form.get('suggestions', '[]')
        custom_notes = request.form.get('custom_notes', '')

        # Parse approved suggestions
        import json
        approved_suggestions = json.loads(approved_suggestions)

        # Read CSV
        df_original = pd.read_csv(io.BytesIO(file.read()))
        df = df_original.copy()

        # Load domain
        if domain_manager and domain_manager.current_domain != domain_name:
            domain_manager.load_domain(domain_name)

        # Apply cleaning based on approved suggestions
        cleaning_log = []

        # Process custom template commands if provided
        if custom_notes and custom_notes.strip():
            command_lines = custom_notes.strip().split('\n')
            for line in command_lines:
                line = line.strip()
                if not line or line.startswith('#'):  # Skip empty lines and comments
                    continue

                try:
                    # FILTER: condition (e.g., FILTER: age < 100)
                    if line.upper().startswith('FILTER:'):
                        condition = line[7:].strip()
                        before_count = len(df)
                        # Safe evaluation using query
                        df = df.query(condition)
                        after_count = len(df)
                        cleaning_log.append(f'✓ FILTER: Removed {before_count - after_count} rows where NOT ({condition})')

                    # FILL: column WITH value (e.g., FILL: income WITH 0)
                    elif line.upper().startswith('FILL:'):
                        parts = line[5:].strip().split(' WITH ')
                        if len(parts) == 2:
                            col = parts[0].strip()
                            value = parts[1].strip()
                            if col in df.columns:
                                missing_count = df[col].isnull().sum()
                                # Try to convert value to appropriate type
                                if value.lower() == 'median' and df[col].dtype in ['float64', 'int64']:
                                    df[col].fillna(df[col].median(), inplace=True)
                                elif value.lower() == 'mean' and df[col].dtype in ['float64', 'int64']:
                                    df[col].fillna(df[col].mean(), inplace=True)
                                elif value.lower() == 'mode':
                                    mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else None
                                    df[col].fillna(mode_val, inplace=True)
                                else:
                                    # Try numeric conversion, fall back to string
                                    try:
                                        if df[col].dtype in ['float64', 'int64']:
                                            value = float(value)
                                    except:
                                        pass
                                    df[col].fillna(value, inplace=True)
                                cleaning_log.append(f'✓ FILL: Filled {missing_count} missing values in "{col}" with {value}')
                            else:
                                cleaning_log.append(f'⚠ FILL: Column "{col}" not found')

                    # DROP: column (e.g., DROP: temporary_column)
                    elif line.upper().startswith('DROP:'):
                        col = line[5:].strip()
                        if col in df.columns:
                            df.drop(columns=[col], inplace=True)
                            cleaning_log.append(f'✓ DROP: Removed column "{col}"')
                        else:
                            cleaning_log.append(f'⚠ DROP: Column "{col}" not found')

                    # REPLACE: column OLD WITH NEW (e.g., REPLACE: gender M WITH Male)
                    elif line.upper().startswith('REPLACE:'):
                        parts = line[8:].strip().split(' WITH ')
                        if len(parts) == 2:
                            col_and_old = parts[0].strip().split(' ', 1)
                            if len(col_and_old) == 2:
                                col = col_and_old[0].strip()
                                old_value = col_and_old[1].strip()
                                new_value = parts[1].strip()
                                if col in df.columns:
                                    # Try numeric conversion if column is numeric
                                    try:
                                        if df[col].dtype in ['float64', 'int64']:
                                            old_value = float(old_value)
                                            new_value = float(new_value)
                                    except:
                                        pass
                                    count = (df[col] == old_value).sum()
                                    df[col] = df[col].replace(old_value, new_value)
                                    cleaning_log.append(f'✓ REPLACE: Replaced {count} occurrences of "{old_value}" with "{new_value}" in "{col}"')
                                else:
                                    cleaning_log.append(f'⚠ REPLACE: Column "{col}" not found')

                    # RENAME: old_column TO new_column
                    elif line.upper().startswith('RENAME:'):
                        parts = line[7:].strip().split(' TO ')
                        if len(parts) == 2:
                            old_col = parts[0].strip()
                            new_col = parts[1].strip()
                            if old_col in df.columns:
                                df.rename(columns={old_col: new_col}, inplace=True)
                                cleaning_log.append(f'✓ RENAME: Renamed column "{old_col}" to "{new_col}"')
                            else:
                                cleaning_log.append(f'⚠ RENAME: Column "{old_col}" not found')

                    # CLIP: column MIN MAX (e.g., CLIP: age 0 120)
                    elif line.upper().startswith('CLIP:'):
                        parts = line[5:].strip().split()
                        if len(parts) == 3:
                            col = parts[0].strip()
                            min_val = float(parts[1])
                            max_val = float(parts[2])
                            if col in df.columns and df[col].dtype in ['float64', 'int64']:
                                clipped_count = ((df[col] < min_val) | (df[col] > max_val)).sum()
                                df[col] = df[col].clip(min_val, max_val)
                                cleaning_log.append(f'✓ CLIP: Clipped {clipped_count} values in "{col}" to range [{min_val}, {max_val}]')
                            else:
                                cleaning_log.append(f'⚠ CLIP: Column "{col}" not found or not numeric')

                    else:
                        # Unknown command - log as note
                        cleaning_log.append(f'📝 Note: {line}')

                except Exception as e:
                    cleaning_log.append(f'❌ Error in command "{line}": {str(e)}')

        for suggestion in approved_suggestions:
            if not suggestion.get('approved', False):
                continue

            sug_type = suggestion['type']

            if sug_type == 'duplicates':
                before_count = len(df)
                df = df.drop_duplicates()
                after_count = len(df)
                cleaning_log.append(f'Removed {before_count - after_count} duplicate rows')

            elif sug_type == 'missing_values':
                col = suggestion['column']
                if col in df.columns:
                    missing_before = df[col].isnull().sum()
                    if df[col].dtype in ['float64', 'int64']:
                        df[col].fillna(df[col].median(), inplace=True)
                        cleaning_log.append(f'Filled {missing_before} missing values in {col} with median')
                    else:
                        mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
                        df[col].fillna(mode_val, inplace=True)
                        cleaning_log.append(f'Filled {missing_before} missing values in {col} with mode')

            elif sug_type == 'outliers':
                col = suggestion['column']
                if col in df.columns and df[col].dtype in ['float64', 'int64']:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 3 * IQR
                    upper_bound = Q3 + 3 * IQR
                    outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                    df[col] = df[col].clip(lower_bound, upper_bound)
                    cleaning_log.append(f'Capped {outliers_count} outliers in {col}')

            elif sug_type == 'invalid_range':
                col = suggestion['column']
                if col in df.columns and df[col].dtype in ['float64', 'int64']:
                    if 'age' in col.lower():
                        df[col] = df[col].clip(0, 120)
                    elif 'gpa' in col.lower():
                        df[col] = df[col].clip(0, 4.0)
                    elif 'attendance' in col.lower() or 'rate' in col.lower():
                        df[col] = df[col].clip(0, 1.0)
                    cleaning_log.append(f'Fixed invalid range values in {col}')

            elif sug_type == 'categorical_format':
                col = suggestion['column']
                if col in df.columns:
                    df[col] = df[col].str.strip()
                    # Standardize common patterns
                    replacements = {'low': 'Low', 'medium': 'Medium', 'high': 'High'}
                    for old, new in replacements.items():
                        df[col] = df[col].str.replace(old, new, case=False, regex=False)
                    cleaning_log.append(f'Standardized categorical values in {col}')

            elif sug_type == 'derived_features':
                # Add domain-specific features
                if domain_name == 'student' or domain_name == 'student_dropout':
                    if 'age' in df.columns:
                        df['age_group'] = pd.cut(df['age'], bins=[0, 12, 17, 25, 40, 100],
                                                 labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'])
                    if 'gpa' in df.columns and 'attendance_rate' in df.columns:
                        df['academic_risk_score'] = (4 - df['gpa']) / 4 * 0.5 + (1 - df['attendance_rate']) * 0.5
                    cleaning_log.append(f'Added derived features for {domain_name}')

        # Store cleaned data in session (using a simple dict for now)
        cleaned_filename = f'cleaned_{file.filename}'
        # Save to temp file
        temp_dir = os.path.join(os.path.dirname(__file__), 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        cleaned_path = os.path.join(temp_dir, cleaned_filename)
        df.to_csv(cleaned_path, index=False)

        # Generate predictions
        # Preserve ID and name columns
        id_cols = [col for col in df.columns if col.lower().endswith('_id') or col.lower() == 'id']
        name_cols = [col for col in df.columns if col.lower().endswith('_name') or col.lower() == 'name']
        identifier_cols = id_cols + name_cols

        identifiers = []
        for idx, row in df.head(100).iterrows():
            row_id = {}
            for col in identifier_cols:
                if col in df.columns:
                    # Convert NaN to None for JSON serialization
                    val = row[col]
                    if pd.isna(val):
                        val = None
                    row_id[col] = val
            identifiers.append(row_id)

        # Remove ID, name, and target for prediction
        target_col = domain_manager.current_model_data['metadata'].get('target_variable')
        df_features = df.drop(columns=identifier_cols + [target_col], errors='ignore')

        # Make predictions
        results = []
        for i, (_, row) in enumerate(df_features.head(100).iterrows()):
            try:
                input_data = row.to_dict()
                result = domain_manager.predict(input_data)
                if i < len(identifiers):
                    result['identifiers'] = identifiers[i]
                results.append(result)
            except Exception as e:
                logger.error(f"Prediction error: {str(e)}")
                error_result = {'error': str(e)}
                if i < len(identifiers):
                    error_result['identifiers'] = identifiers[i]
                results.append(error_result)

        return jsonify({
            'success': True,
            'cleaning_log': cleaning_log,
            'results': results,
            'count': len(results),
            'domain': domain_name,
            'cleaned_filename': cleaned_filename,
            'original_rows': len(df_original),
            'cleaned_rows': len(df),
            'custom_notes': custom_notes
        }), 200

    except Exception as e:
        logger.error(f"Clean and predict error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/download-cleaned-csv/<filename>')
def download_cleaned_csv(filename):
    """Download cleaned CSV file"""
    try:
        temp_dir = os.path.join(os.path.dirname(__file__), 'temp')
        file_path = os.path.join(temp_dir, filename)

        if not os.path.exists(file_path):
            return jsonify({'success': False, 'error': 'File not found'}), 404

        return send_file(file_path, as_attachment=True, download_name=filename)

    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/download-prediction-report', methods=['POST'])
def download_prediction_report():
    """Generate and download prediction report as CSV"""
    try:
        data = request.get_json()
        results = data.get('results', [])
        domain = data.get('domain', 'unknown')
        cleaning_log = data.get('cleaning_log', [])

        # Create DataFrame from results
        report_data = []
        for i, result in enumerate(results):
            row = {'row_number': i + 1}

            # Add identifiers
            if 'identifiers' in result:
                row.update(result['identifiers'])

            # Add prediction results
            row['prediction'] = 'Positive' if result.get('prediction') == 1 else 'Negative'
            row['confidence'] = f"{result.get('confidence', 0) * 100:.1f}%"
            row['recommendation'] = result.get('recommendation', '')

            report_data.append(row)

        df_report = pd.DataFrame(report_data)

        # Add metadata sheet information as comments
        metadata = f"# Prediction Report\n"
        metadata += f"# Domain: {domain}\n"
        metadata += f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        metadata += f"# Total Predictions: {len(results)}\n"
        if cleaning_log:
            metadata += f"# Cleaning Applied:\n"
            for log_entry in cleaning_log:
                metadata += f"#   - {log_entry}\n"
        metadata += "\n"

        # Save to temp file
        temp_dir = os.path.join(os.path.dirname(__file__), 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        report_filename = f'prediction_report_{domain}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        report_path = os.path.join(temp_dir, report_filename)

        # Write metadata and data
        with open(report_path, 'w') as f:
            f.write(metadata)
            df_report.to_csv(f, index=False)

        return send_file(report_path, as_attachment=True, download_name=report_filename)

    except Exception as e:
        logger.error(f"Report generation error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'available_endpoints': [
            '/api/health',
            '/api/predict',
            '/api/batch-predict',
            '/api/features'
        ]
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


if __name__ == '__main__':
    # Load configuration
    config = load_config()
    api_config = config['api']

    # Load model artifacts
    logger.info("=" * 60)
    logger.info("Starting Data Insight ML API")
    logger.info("=" * 60)

    success = load_model_artifacts()

    if not success:
        logger.warning("\nFailed to load legacy model artifacts!")
        logger.info("Continuing with multi-domain system...")

    # Initialize domain manager
    try:
        domain_manager = DomainManager()
        logger.info(f"Domain manager initialized with {len(domain_manager.available_domains)} domains")
    except Exception as e:
        logger.error(f"Failed to initialize domain manager: {str(e)}")
        domain_manager = None

    # Print startup info
    logger.info("")
    logger.info("Server Configuration:")
    logger.info(f"  - Host: {api_config['host']}")
    logger.info(f"  - Port: {api_config['port']}")
    logger.info(f"  - Debug: {api_config['debug']}")
    logger.info(f"  - CORS: {api_config['cors_enabled']}")
    logger.info("")
    logger.info("Available Endpoints:")
    logger.info(f"  - GET  http://localhost:{api_config['port']}/api/health")
    logger.info(f"  - POST http://localhost:{api_config['port']}/api/predict")
    logger.info(f"  - POST http://localhost:{api_config['port']}/api/batch-predict")
    logger.info(f"  - GET  http://localhost:{api_config['port']}/api/features")
    logger.info(f"  - GET  http://localhost:{api_config['port']}/api/domains")
    logger.info(f"  - POST http://localhost:{api_config['port']}/api/domain/<name>/predict")
    logger.info(f"  - GET  http://localhost:{api_config['port']}/api/domain/<name>/demo")
    logger.info(f"  - POST http://localhost:{api_config['port']}/api/upload-predict")
    logger.info("")

    if model_info:
        logger.info("Model Information:")
        logger.info(f"  - Model: {model_info['model_name']}")
        logger.info(f"  - Test Accuracy: {model_info['test_accuracy']:.2%}")
        logger.info(f"  - Features: {model_info['n_features']}")

    logger.info("=" * 60)
    logger.info("")

    # Run Flask server
    app.run(
        host=api_config['host'],
        port=api_config['port'],
        debug=api_config['debug']
    )
