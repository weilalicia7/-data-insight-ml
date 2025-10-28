# -*- coding: utf-8 -*-
import sys
import io

# Force UTF-8 encoding for stdout
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Complete Model Training Script for Mentorship Risk Prediction
JP Morgan Data for Good Hackathon 2025 - Team 2

This script:
1. Loads mentorship data from CSV
2. Engineers 24 features from raw data
3. Trains Random Forest (500 trees, max_depth=7)
4. Performs 10-fold cross-validation
5. Saves model artifacts (model.pkl, scaler.pkl, feature_columns.pkl)
6. Prints accuracy, precision, recall, F1-score

Usage:
    python train_model.py
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')

# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 10

# Model parameters
N_ESTIMATORS = 500
MAX_DEPTH = 7
MIN_SAMPLES_SPLIT = 20
MIN_SAMPLES_LEAF = 10
CLASS_WEIGHT = 'balanced'

# Paths
DATA_PATH = 'ml_ready_dataset.csv'  # Input data file
MODELS_DIR = 'models'
MODEL_PATH = os.path.join(MODELS_DIR, 'random_forest_model.pkl')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')
FEATURES_PATH = os.path.join(MODELS_DIR, 'feature_columns.pkl')

# Create models directory
os.makedirs(MODELS_DIR, exist_ok=True)


def print_header(title):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def load_data(filepath):
    """
    Load mentorship data from CSV.

    Expected columns:
    - binome_id, mentor_id, mentee_id (IDs)
    - workfield (mentor field)
    - field_of_study, study_level, degree, needs (mentee info)
    - average_grade, program, engagement_score, desired_exchange_frequency
    - binome_score (compatibility score)
    - target (binary: 1=success, 0=failure)
    """
    print_header("LOADING DATA")

    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)

    print(f"✓ Loaded {len(df):,} records")
    print(f"  Columns: {len(df.columns)}")
    print(f"\nData shape: {df.shape}")

    # Check for target column
    if 'target' not in df.columns:
        print("\n⚠ Warning: 'target' column not found!")
        print("  Creating target from binome_statut...")
        if 'binome_statut' in df.columns:
            df['target'] = (df['binome_statut'] == 'COMPLETED').astype(int)
            print("  ✓ Target created (1=COMPLETED, 0=REJECTED/CANCELLED)")
        else:
            raise ValueError("Cannot create target: 'binome_statut' column missing")

    # Display target distribution
    target_counts = df['target'].value_counts()
    print(f"\nTarget Distribution:")
    print(f"  Success (1): {target_counts.get(1, 0):,} ({target_counts.get(1, 0)/len(df)*100:.1f}%)")
    print(f"  Failure (0): {target_counts.get(0, 0):,} ({target_counts.get(0, 0)/len(df)*100:.1f}%)")

    if target_counts.get(0, 0) > target_counts.get(1, 0):
        imbalance_ratio = target_counts.get(0, 0) / target_counts.get(1, 0)
        print(f"  Class Imbalance: {imbalance_ratio:.2f}:1 (failure:success)")

    return df


def engineer_features(df):
    """
    Engineer 24 features from raw mentorship data.

    Features created:
    1. Numeric features (5):
       - engagement_score, binome_score, average_grade_numeric,
       - field_similarity, needs_count

    2. One-hot encoded categorical features (14):
       - workfield, field_of_study, study_level, degree, program

    3. Engineered interaction features (5):
       - needs_pro, needs_study, needs_both
       - high_engagement, low_binome_score

    Total: ~24 features (varies with categorical cardinality)
    """
    print_header("FEATURE ENGINEERING")

    df_work = df.copy()

    # === 1. NUMERIC FEATURES ===
    print("\n1. Processing Numeric Features...")

    # Engagement score (already numeric)
    if 'engagement_score' in df_work.columns:
        df_work['engagement_score'] = pd.to_numeric(df_work['engagement_score'], errors='coerce').fillna(0.0)
        print(f"   ✓ engagement_score: range [{df_work['engagement_score'].min():.1f}, {df_work['engagement_score'].max():.1f}]")

    # Binome score
    if 'binome_score' in df_work.columns:
        df_work['binome_score'] = pd.to_numeric(df_work['binome_score'], errors='coerce').fillna(0.0)
        print(f"   ✓ binome_score: range [{df_work['binome_score'].min():.0f}, {df_work['binome_score'].max():.0f}]")

    # Average grade (convert from text to numeric)
    if 'average_grade' in df_work.columns:
        grade_mapping = {
            'Not specified (or Not provided)': 2.5,
            'Below average': 1.0,
            'Average': 2.5,
            'Good': 3.5,
            'Very good': 4.0,
            'Excellent': 5.0
        }
        df_work['average_grade_numeric'] = df_work['average_grade'].map(grade_mapping).fillna(2.5)
        print(f"   ✓ average_grade_numeric: converted from categories")

    # === 2. FIELD SIMILARITY (ENGINEERED FEATURE) ===
    print("\n2. Engineering Field Similarity...")

    def calculate_field_similarity(row):
        """Calculate similarity between mentor workfield and mentee field_of_study."""
        workfield = str(row.get('workfield', '')).lower()
        field_of_study = str(row.get('field_of_study', '')).lower()

        if pd.isna(row.get('workfield')) or pd.isna(row.get('field_of_study')):
            return 0

        # Exact match
        if workfield == field_of_study:
            return 2

        # Related fields mapping
        related_fields = {
            'computer science': ['it, is, data, web, tech', 'it', 'data', 'tech'],
            'banking': ['banking, insurance and finance', 'finance'],
            'finance': ['banking, insurance and finance', 'accounting, finance'],
            'accounting': ['accounting, finance', 'commerce, management'],
            'management': ['commerce, management', 'business'],
            'human resources': ['commerce, management', 'hr'],
            'engineering': ['civil engineering', 'mechanical', 'electrical'],
        }

        # Check if related
        for key_field, related_list in related_fields.items():
            if key_field in workfield:
                for related in related_list:
                    if related in field_of_study:
                        return 1  # Related match

        return 0  # No match

    df_work['field_similarity'] = df_work.apply(calculate_field_similarity, axis=1)
    similarity_dist = df_work['field_similarity'].value_counts().sort_index()
    print(f"   ✓ field_similarity: {dict(similarity_dist)}")

    # === 3. NEEDS PARSING (ENGINEERED FEATURES) ===
    print("\n3. Parsing Mentee Needs...")

    if 'needs' in df_work.columns:
        df_work['needs_pro'] = df_work['needs'].astype(str).str.contains('pro', case=False, na=False).astype(int)
        df_work['needs_study'] = df_work['needs'].astype(str).str.contains('study', case=False, na=False).astype(int)
        df_work['needs_both'] = ((df_work['needs_pro'] == 1) & (df_work['needs_study'] == 1)).astype(int)
        df_work['needs_count'] = df_work['needs_pro'] + df_work['needs_study']

        print(f"   ✓ needs_pro: {df_work['needs_pro'].sum():,} records")
        print(f"   ✓ needs_study: {df_work['needs_study'].sum():,} records")
        print(f"   ✓ needs_both: {df_work['needs_both'].sum():,} records")

    # === 4. INTERACTION FEATURES ===
    print("\n4. Creating Interaction Features...")

    # High engagement flag
    if 'engagement_score' in df_work.columns:
        df_work['high_engagement'] = (df_work['engagement_score'] >= 2.0).astype(int)
        print(f"   ✓ high_engagement (>=2.0): {df_work['high_engagement'].sum():,} records")

    # Low binome score flag
    if 'binome_score' in df_work.columns:
        df_work['low_binome_score'] = (df_work['binome_score'] <= 3.0).astype(int)
        print(f"   ✓ low_binome_score (<=3.0): {df_work['low_binome_score'].sum():,} records")

    # === 5. CATEGORICAL FEATURES ===
    print("\n5. Processing Categorical Features...")

    categorical_cols = ['workfield', 'field_of_study', 'study_level',
                       'degree', 'program', 'desired_exchange_frequency']

    # Fill missing values
    for col in categorical_cols:
        if col in df_work.columns:
            missing_count = df_work[col].isna().sum()
            df_work[col] = df_work[col].fillna('Unknown')
            if missing_count > 0:
                print(f"   ✓ {col}: filled {missing_count:,} missing values")

            # Show top categories
            top_cats = df_work[col].value_counts().head(3)
            print(f"     Top: {', '.join([f'{cat} ({count})' for cat, count in top_cats.items()])}")

    # === 6. ONE-HOT ENCODING ===
    print("\n6. One-Hot Encoding Categorical Features...")

    # Apply one-hot encoding
    df_encoded = pd.get_dummies(
        df_work,
        columns=categorical_cols,
        drop_first=True,  # Avoid multicollinearity
        dtype=int
    )

    # Count new features
    original_cols = set(df_work.columns)
    encoded_cols = set(df_encoded.columns)
    new_cols = encoded_cols - original_cols

    print(f"   ✓ Created {len(new_cols)} one-hot encoded features")

    # === 7. SELECT FINAL FEATURES ===
    print("\n7. Selecting Final Feature Set...")

    # Define feature columns (exclude IDs and target)
    exclude_cols = ['binome_id', 'mentor_id', 'mentee_id', 'target',
                   'binome_statut', 'needs', 'average_grade']  # average_grade is now average_grade_numeric

    feature_cols = [col for col in df_encoded.columns if col not in exclude_cols]

    # Prioritize numeric and engineered features
    numeric_features = ['engagement_score', 'binome_score', 'average_grade_numeric',
                       'field_similarity', 'needs_count', 'needs_pro', 'needs_study',
                       'needs_both', 'high_engagement', 'low_binome_score']

    # Add one-hot encoded features
    onehot_features = [col for col in feature_cols if col not in numeric_features]

    # Combine all features
    all_features = numeric_features + onehot_features

    # Filter to available columns
    available_features = [col for col in all_features if col in df_encoded.columns]

    print(f"   ✓ Total features: {len(available_features)}")
    print(f"     - Numeric: {len([f for f in available_features if f in numeric_features])}")
    print(f"     - One-hot: {len([f for f in available_features if f in onehot_features])}")

    # Create feature matrix
    X = df_encoded[available_features].copy()
    y = df_encoded['target'].copy()

    print(f"\n✓ Feature engineering complete!")
    print(f"  Final shape: X={X.shape}, y={y.shape}")

    # Display feature list
    print(f"\nFinal Feature List ({len(available_features)} features):")
    for i, feat in enumerate(available_features[:10], 1):
        print(f"  {i:2d}. {feat}")
    if len(available_features) > 10:
        print(f"  ... ({len(available_features) - 10} more features)")

    return X, y, available_features, numeric_features


def train_model(X, y, feature_names, numeric_features):
    """
    Train Random Forest model with specified hyperparameters.
    Includes train/test split and model training.
    """
    print_header("MODEL TRAINING")

    # Train/test split
    print(f"\nSplitting data (test_size={TEST_SIZE})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"✓ Train set: {len(X_train):,} samples")
    print(f"✓ Test set:  {len(X_test):,} samples")

    # Check class distribution
    train_dist = y_train.value_counts()
    print(f"\nTrain set distribution:")
    print(f"  Success: {train_dist.get(1, 0):,} ({train_dist.get(1, 0)/len(y_train)*100:.1f}%)")
    print(f"  Failure: {train_dist.get(0, 0):,} ({train_dist.get(0, 0)/len(y_train)*100:.1f}%)")

    # === FEATURE SCALING ===
    print(f"\nScaling numeric features...")
    scaler = StandardScaler()

    # Scale only numeric features
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    numeric_cols_present = [col for col in numeric_features if col in X_train.columns]

    if numeric_cols_present:
        X_train_scaled[numeric_cols_present] = scaler.fit_transform(X_train[numeric_cols_present])
        X_test_scaled[numeric_cols_present] = scaler.transform(X_test[numeric_cols_present])
        print(f"✓ Scaled {len(numeric_cols_present)} numeric features")
    else:
        print("⚠ No numeric features to scale")

    # === RANDOM FOREST TRAINING ===
    print(f"\nTraining Random Forest...")
    print(f"  Hyperparameters:")
    print(f"    - n_estimators: {N_ESTIMATORS}")
    print(f"    - max_depth: {MAX_DEPTH}")
    print(f"    - min_samples_split: {MIN_SAMPLES_SPLIT}")
    print(f"    - min_samples_leaf: {MIN_SAMPLES_LEAF}")
    print(f"    - class_weight: {CLASS_WEIGHT}")
    print(f"    - random_state: {RANDOM_STATE}")

    rf_model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight=CLASS_WEIGHT,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0
    )

    print(f"\nTraining model on {len(X_train):,} samples...")
    rf_model.fit(X_train_scaled, y_train)
    print(f"✓ Model trained successfully!")

    # === PREDICTIONS ===
    print(f"\nMaking predictions...")
    y_train_pred = rf_model.predict(X_train_scaled)
    y_test_pred = rf_model.predict(X_test_scaled)

    y_train_proba = rf_model.predict_proba(X_train_scaled)[:, 1]
    y_test_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

    return rf_model, scaler, X_train_scaled, X_test_scaled, y_train, y_test, y_train_pred, y_test_pred, y_train_proba, y_test_proba


def perform_cross_validation(X, y, feature_names, numeric_features):
    """
    Perform 10-fold stratified cross-validation.
    """
    print_header("10-FOLD CROSS-VALIDATION")

    # Scale features first
    print(f"\nPreparing data for cross-validation...")
    scaler_cv = StandardScaler()
    X_scaled = X.copy()

    numeric_cols_present = [col for col in numeric_features if col in X.columns]
    if numeric_cols_present:
        X_scaled[numeric_cols_present] = scaler_cv.fit_transform(X[numeric_cols_present])

    # Create Random Forest model
    rf_cv = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight=CLASS_WEIGHT,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0
    )

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    print(f"Running {CV_FOLDS}-fold cross-validation...")
    print("(This may take a few minutes...)\n")

    # Perform cross-validation for different metrics
    cv_accuracy = cross_val_score(rf_cv, X_scaled, y, cv=skf, scoring='accuracy', n_jobs=-1)
    cv_precision = cross_val_score(rf_cv, X_scaled, y, cv=skf, scoring='precision', n_jobs=-1)
    cv_recall = cross_val_score(rf_cv, X_scaled, y, cv=skf, scoring='recall', n_jobs=-1)
    cv_f1 = cross_val_score(rf_cv, X_scaled, y, cv=skf, scoring='f1', n_jobs=-1)
    cv_roc_auc = cross_val_score(rf_cv, X_scaled, y, cv=skf, scoring='roc_auc', n_jobs=-1)

    print("✓ Cross-validation complete!\n")

    # Display results
    print(f"{'Metric':<15} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-" * 55)
    print(f"{'Accuracy':<15} {cv_accuracy.mean():.4f}    {cv_accuracy.std():.4f}    {cv_accuracy.min():.4f}    {cv_accuracy.max():.4f}")
    print(f"{'Precision':<15} {cv_precision.mean():.4f}    {cv_precision.std():.4f}    {cv_precision.min():.4f}    {cv_precision.max():.4f}")
    print(f"{'Recall':<15} {cv_recall.mean():.4f}    {cv_recall.std():.4f}    {cv_recall.min():.4f}    {cv_recall.max():.4f}")
    print(f"{'F1-Score':<15} {cv_f1.mean():.4f}    {cv_f1.std():.4f}    {cv_f1.min():.4f}    {cv_f1.max():.4f}")
    print(f"{'ROC-AUC':<15} {cv_roc_auc.mean():.4f}    {cv_roc_auc.std():.4f}    {cv_roc_auc.min():.4f}    {cv_roc_auc.max():.4f}")

    return {
        'accuracy': cv_accuracy,
        'precision': cv_precision,
        'recall': cv_recall,
        'f1': cv_f1,
        'roc_auc': cv_roc_auc
    }


def evaluate_model(y_train, y_test, y_train_pred, y_test_pred, y_train_proba, y_test_proba):
    """
    Evaluate model performance on train and test sets.
    Print accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix.
    """
    print_header("MODEL EVALUATION")

    # === TRAIN SET METRICS ===
    print("\n📊 TRAIN SET PERFORMANCE:")
    print("-" * 80)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred, zero_division=0)
    train_recall = recall_score(y_train, y_train_pred, zero_division=0)
    train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
    train_roc_auc = roc_auc_score(y_train, y_train_proba)

    print(f"Accuracy:   {train_accuracy:.4f}")
    print(f"Precision:  {train_precision:.4f}")
    print(f"Recall:     {train_recall:.4f}")
    print(f"F1-Score:   {train_f1:.4f}")
    print(f"ROC-AUC:    {train_roc_auc:.4f}")

    # Confusion Matrix
    cm_train = confusion_matrix(y_train, y_train_pred)
    tn_train, fp_train, fn_train, tp_train = cm_train.ravel()

    print(f"\nConfusion Matrix (Train):")
    print(f"                 Predicted")
    print(f"               Fail    Success")
    print(f"  Actual Fail  {tn_train:5d}   {fp_train:5d}")
    print(f"         Succ  {fn_train:5d}   {tp_train:5d}")

    # === TEST SET METRICS ===
    print("\n" + "-" * 80)
    print("🎯 TEST SET PERFORMANCE:")
    print("-" * 80)

    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
    test_roc_auc = roc_auc_score(y_test, y_test_proba)

    print(f"Accuracy:   {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Precision:  {test_precision:.4f} ({test_precision*100:.2f}%)")
    print(f"Recall:     {test_recall:.4f} ({test_recall*100:.2f}%)")
    print(f"F1-Score:   {test_f1:.4f}")
    print(f"ROC-AUC:    {test_roc_auc:.4f}")

    # Confusion Matrix
    cm_test = confusion_matrix(y_test, y_test_pred)
    tn_test, fp_test, fn_test, tp_test = cm_test.ravel()

    print(f"\nConfusion Matrix (Test):")
    print(f"                 Predicted")
    print(f"               Fail    Success")
    print(f"  Actual Fail  {tn_test:5d}   {fp_test:5d}")
    print(f"         Succ  {fn_test:5d}   {tp_test:5d}")

    # Classification Report
    print(f"\nClassification Report (Test):")
    print(classification_report(y_test, y_test_pred,
                                target_names=['Failure', 'Success'],
                                digits=4))

    # Model Insights
    print("-" * 80)
    print("📈 KEY INSIGHTS:")
    print("-" * 80)

    # Overfitting check
    accuracy_diff = train_accuracy - test_accuracy
    if accuracy_diff > 0.10:
        print(f"⚠ Possible overfitting detected (train-test gap: {accuracy_diff:.4f})")
    elif accuracy_diff > 0.05:
        print(f"✓ Slight overfitting (train-test gap: {accuracy_diff:.4f})")
    else:
        print(f"✓ Good generalization (train-test gap: {accuracy_diff:.4f})")

    # Recall focus
    print(f"\n✓ Recall: {test_recall:.4f} - Catches {test_recall*100:.1f}% of actual failures")
    print(f"  (High recall prioritized due to class_weight='balanced')")

    # Precision-Recall tradeoff
    if test_recall > 0.80 and test_precision < 0.40:
        print(f"  ⚠ High recall but low precision - many false positives")
    elif test_recall > 0.70 and test_precision > 0.50:
        print(f"  ✓ Good balance between precision and recall")

    return {
        'train': {
            'accuracy': train_accuracy,
            'precision': train_precision,
            'recall': train_recall,
            'f1': train_f1,
            'roc_auc': train_roc_auc
        },
        'test': {
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'f1': test_f1,
            'roc_auc': test_roc_auc
        }
    }


def save_model_artifacts(model, scaler, feature_columns):
    """
    Save trained model, scaler, and feature columns to pickle files.
    """
    print_header("SAVING MODEL ARTIFACTS")

    print(f"\nSaving to directory: {MODELS_DIR}/")

    # Save Random Forest model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    model_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)  # MB
    print(f"✓ Saved Random Forest model: {MODEL_PATH}")
    print(f"  File size: {model_size:.2f} MB")
    print(f"  Trees: {model.n_estimators}, Max depth: {model.max_depth}")

    # Save StandardScaler
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    scaler_size = os.path.getsize(SCALER_PATH) / 1024  # KB
    print(f"\n✓ Saved StandardScaler: {SCALER_PATH}")
    print(f"  File size: {scaler_size:.2f} KB")
    print(f"  Features scaled: {scaler.n_features_in_}")

    # Save feature column names
    with open(FEATURES_PATH, 'wb') as f:
        pickle.dump(feature_columns, f)
    features_size = os.path.getsize(FEATURES_PATH) / 1024  # KB
    print(f"\n✓ Saved feature columns: {FEATURES_PATH}")
    print(f"  File size: {features_size:.2f} KB")
    print(f"  Total features: {len(feature_columns)}")

    print(f"\n✅ All model artifacts saved successfully!")


def display_feature_importance(model, feature_names):
    """
    Display feature importance from trained Random Forest.
    """
    print_header("FEATURE IMPORTANCE")

    # Get feature importances
    importances = model.feature_importances_

    # Create DataFrame
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances,
        'Importance_%': importances * 100
    }).sort_values('Importance', ascending=False)

    # Display top 15 features
    print("\nTop 15 Most Important Features:")
    print("-" * 80)
    print(f"{'Rank':<6} {'Feature':<35} {'Importance':<12} {'Importance %':<12}")
    print("-" * 80)

    for idx, (_, row) in enumerate(feature_importance_df.head(15).iterrows(), 1):
        print(f"{idx:<6} {row['Feature']:<35} {row['Importance']:<12.6f} {row['Importance_%']:<12.2f}%")

    # Summary statistics
    print("\n" + "-" * 80)
    print("Feature Importance Summary:")
    print("-" * 80)

    top5_importance = feature_importance_df.head(5)['Importance_%'].sum()
    top10_importance = feature_importance_df.head(10)['Importance_%'].sum()

    print(f"Top 5 features:  {top5_importance:.1f}% of total importance")
    print(f"Top 10 features: {top10_importance:.1f}% of total importance")

    # Most important feature
    most_important = feature_importance_df.iloc[0]
    print(f"\nMost important: {most_important['Feature']} ({most_important['Importance_%']:.2f}%)")

    return feature_importance_df


def main():
    """Main training pipeline."""

    print("\n" + "=" * 80)
    print("  MENTORSHIP RISK PREDICTION - MODEL TRAINING")
    print("  JP Morgan Data for Good Hackathon 2025 - Team 2")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 80)

    # 1. Load data
    df = load_data(DATA_PATH)

    # 2. Engineer features
    X, y, feature_names, numeric_features = engineer_features(df)

    # 3. Train model
    (model, scaler, X_train, X_test, y_train, y_test,
     y_train_pred, y_test_pred, y_train_proba, y_test_proba) = train_model(
        X, y, feature_names, numeric_features
    )

    # 4. Perform 10-fold cross-validation
    cv_results = perform_cross_validation(X, y, feature_names, numeric_features)

    # 5. Evaluate model
    metrics = evaluate_model(
        y_train, y_test, y_train_pred, y_test_pred,
        y_train_proba, y_test_proba
    )

    # 6. Display feature importance
    feature_importance_df = display_feature_importance(model, feature_names)

    # 7. Save model artifacts
    save_model_artifacts(model, scaler, feature_names)

    # === FINAL SUMMARY ===
    print_header("TRAINING COMPLETE")

    print("\n✅ Model training completed successfully!\n")

    print("📊 Final Test Set Metrics:")
    print(f"  - Accuracy:  {metrics['test']['accuracy']:.4f} ({metrics['test']['accuracy']*100:.2f}%)")
    print(f"  - Precision: {metrics['test']['precision']:.4f} ({metrics['test']['precision']*100:.2f}%)")
    print(f"  - Recall:    {metrics['test']['recall']:.4f} ({metrics['test']['recall']*100:.2f}%)")
    print(f"  - F1-Score:  {metrics['test']['f1']:.4f}")
    print(f"  - ROC-AUC:   {metrics['test']['roc_auc']:.4f}")

    print("\n📈 Cross-Validation (10-fold) Metrics:")
    print(f"  - Accuracy:  {cv_results['accuracy'].mean():.4f} ± {cv_results['accuracy'].std():.4f}")
    print(f"  - Precision: {cv_results['precision'].mean():.4f} ± {cv_results['precision'].std():.4f}")
    print(f"  - Recall:    {cv_results['recall'].mean():.4f} ± {cv_results['recall'].std():.4f}")
    print(f"  - F1-Score:  {cv_results['f1'].mean():.4f} ± {cv_results['f1'].std():.4f}")
    print(f"  - ROC-AUC:   {cv_results['roc_auc'].mean():.4f} ± {cv_results['roc_auc'].std():.4f}")

    print("\n💾 Saved Files:")
    print(f"  - {MODEL_PATH}")
    print(f"  - {SCALER_PATH}")
    print(f"  - {FEATURES_PATH}")

    print("\n🎯 Next Steps:")
    print("  1. Test the API: python app.py")
    print("  2. Run test suite: python test_api.py")
    print("  3. Integrate with frontend: demo3.html")

    print("\n" + "=" * 80)
    print("  Thank you for using the training script!")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
