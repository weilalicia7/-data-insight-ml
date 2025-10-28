"""
Script to create model pickle files from trained Random Forest model.

This script should be run after training your model in Jupyter notebooks.
It saves the model artifacts needed by the Flask API.

Usage:
    python create_model_files.py
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import os

def create_dummy_model_files():
    """
    Creates dummy/example model files for testing the API.

    In production, replace this with your actual trained model from the notebooks.
    """

    print("Creating model artifacts...")

    # === 1. CREATE RANDOM FOREST MODEL ===
    # This is a dummy model - replace with your actual trained model
    rf_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=7,
        random_state=42,
        class_weight='balanced'
    )

    # Create dummy training data (24 features)
    X_dummy = np.random.randn(100, 24)
    y_dummy = np.random.randint(0, 2, 100)

    # Train dummy model
    rf_model.fit(X_dummy, y_dummy)

    print(f"✓ Created Random Forest model (500 trees, max_depth=7)")

    # === 2. CREATE SCALER ===
    # Scaler for numeric features
    scaler = StandardScaler()

    # These are the numeric columns that get scaled
    numeric_features = np.random.randn(100, 5)  # 5 numeric features
    scaler.fit(numeric_features)

    print(f"✓ Created StandardScaler (fitted on 5 numeric features)")

    # === 3. DEFINE FEATURE COLUMNS ===
    # These are the 24 features in the exact order expected by the model
    feature_columns = [
        # Numeric features (4)
        'engagement_score',
        'project_confidence_level',
        'mentor_availability',
        'previous_rejection',

        # Workfield one-hot (5 features, dropped 'Computer Science')
        'workfield_Engineering',
        'workfield_Business',
        'workfield_Healthcare',
        'workfield_Teaching',
        'workfield_Other',

        # Study level one-hot (4 features, dropped 'Bac+1')
        'study_level_Bac+2',
        'study_level_Bac+3',
        'study_level_Bac+4',
        'study_level_Bac+5+',

        # Needs one-hot (2 features, dropped 'Professional')
        'needs_Academic',
        'needs_Both',

        # Month one-hot (selected high-impact months, 3 features)
        'registration_month_May',
        'registration_month_June',
        'registration_month_July',

        # Engineered features (4)
        'summer_registration',
        'low_engagement',
        'high_risk_field',
        'engagement_confidence_interaction'
    ]

    print(f"✓ Defined {len(feature_columns)} feature columns")

    # === 4. SAVE TO PICKLE FILES ===
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)

    # Save Random Forest model
    with open(os.path.join(models_dir, 'random_forest_model.pkl'), 'wb') as f:
        pickle.dump(rf_model, f)
    print(f"✓ Saved random_forest_model.pkl")

    # Save scaler
    with open(os.path.join(models_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Saved scaler.pkl")

    # Save feature columns
    with open(os.path.join(models_dir, 'feature_columns.pkl'), 'wb') as f:
        pickle.dump(feature_columns, f)
    print(f"✓ Saved feature_columns.pkl")

    print("\n" + "=" * 60)
    print("Model files created successfully!")
    print("=" * 60)
    print(f"\nFiles saved in: {os.path.abspath(models_dir)}/")
    print("  - random_forest_model.pkl")
    print("  - scaler.pkl")
    print("  - feature_columns.pkl")
    print("\nNOTE: These are DUMMY models for testing.")
    print("Replace with your actual trained model from notebooks!")
    print("=" * 60)


def load_from_notebook_model():
    """
    Example of how to load and save your actual trained model from notebooks.

    Uncomment and modify this section to use your real trained model.
    """

    # === EXAMPLE: Load from your training notebook ===

    # 1. Load your trained Random Forest model
    # (Assuming you saved it in your notebook as 'rf_model')
    # from sklearn.externals import joblib
    # rf_model = joblib.load('../models/trained_rf.pkl')

    # 2. Load your fitted scaler
    # scaler = joblib.load('../models/fitted_scaler.pkl')

    # 3. Get feature columns from your training data
    # feature_columns = list(X_train.columns)  # From your notebook

    # 4. Save to backend/models/
    # with open('models/random_forest_model.pkl', 'wb') as f:
    #     pickle.dump(rf_model, f)
    # with open('models/scaler.pkl', 'wb') as f:
    #     pickle.dump(scaler, f)
    # with open('models/feature_columns.pkl', 'wb') as f:
    #     pickle.dump(feature_columns, f)

    pass


def verify_model_files():
    """Verify that model files can be loaded correctly."""

    print("\n" + "=" * 60)
    print("Verifying model files...")
    print("=" * 60)

    try:
        # Load Random Forest
        with open('models/random_forest_model.pkl', 'rb') as f:
            model = pickle.load(f)
        print(f"✓ Random Forest: {type(model).__name__}")
        print(f"  - Estimators: {model.n_estimators}")
        print(f"  - Max Depth: {model.max_depth}")

        # Load Scaler
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        print(f"✓ Scaler: {type(scaler).__name__}")
        print(f"  - Features: {scaler.n_features_in_}")

        # Load Feature Columns
        with open('models/feature_columns.pkl', 'rb') as f:
            features = pickle.load(f)
        print(f"✓ Feature Columns: {len(features)} features")
        print(f"  - First 5: {features[:5]}")
        print(f"  - Last 5: {features[-5:]}")

        print("\n✓ All model files loaded successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Error loading model files: {e}")
        print("=" * 60)


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Model File Creation Script")
    print("JP Morgan Data for Good Hackathon 2025 - Team 2")
    print("=" * 60)

    # Create dummy model files for testing
    create_dummy_model_files()

    # Verify they can be loaded
    verify_model_files()

    print("\nTo use your REAL trained model:")
    print("1. Edit load_from_notebook_model() in this file")
    print("2. Point it to your actual trained model from notebooks")
    print("3. Run this script again")
    print("\n" + "=" * 60)
