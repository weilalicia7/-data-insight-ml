"""
Model Training Script with Multiple Algorithms
Automatically trains and compares different ML models
"""

import pandas as pd
import numpy as np
import pickle
import yaml
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')

# Try to import xgboost (optional)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Note: XGBoost not available. Install with: pip install xgboost")


def load_config():
    """Load configuration"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def load_metadata():
    """Load data metadata"""
    with open('data_metadata.yaml', 'r') as f:
        return yaml.safe_load(f)


def print_header(title):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def load_prepared_data():
    """Load the prepared dataset"""
    print_header("LOADING PREPARED DATA")

    if not os.path.exists('ml_ready_dataset.csv'):
        print("\nERROR: ml_ready_dataset.csv not found!")
        print("Please run 'python prepare_data.py' first.")
        exit(1)

    df = pd.read_csv('ml_ready_dataset.csv')
    print(f" Loaded {len(df):,} samples with {len(df.columns)} columns")

    metadata = load_metadata()
    target_col = metadata['target_column']

    if target_col not in df.columns:
        print(f"\nERROR: Target column '{target_col}' not found!")
        exit(1)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    print(f"\nDataset shape:")
    print(f"  X: {X.shape}")
    print(f"  y: {y.shape}")

    # Check target distribution
    print(f"\nTarget distribution:")
    for val, count in y.value_counts().items():
        print(f"  {val}: {count:,} ({count/len(y)*100:.1f}%)")

    return X, y, metadata


def create_models(config):
    """Create model instances based on configuration"""
    print_header("INITIALIZING MODELS")

    models = {}

    # Random Forest
    if config['models']['random_forest']:
        rf_params = config['models']['rf_params']
        models['Random Forest'] = RandomForestClassifier(
            n_estimators=rf_params['n_estimators'],
            max_depth=rf_params['max_depth'],
            min_samples_split=rf_params['min_samples_split'],
            min_samples_leaf=rf_params['min_samples_leaf'],
            class_weight=rf_params['class_weight'],
            random_state=config['data']['random_state'],
            n_jobs=-1
        )
        print(" Random Forest initialized")

    # Logistic Regression
    if config['models']['logistic_regression']:
        lr_params = config['models']['lr_params']
        models['Logistic Regression'] = LogisticRegression(
            max_iter=lr_params['max_iter'],
            class_weight=lr_params['class_weight'],
            random_state=config['data']['random_state'],
            n_jobs=-1
        )
        print(" Logistic Regression initialized")

    # XGBoost
    if config['models']['xgboost'] and XGBOOST_AVAILABLE:
        xgb_params = config['models']['xgb_params']
        models['XGBoost'] = xgb.XGBClassifier(
            n_estimators=xgb_params['n_estimators'],
            max_depth=xgb_params['max_depth'],
            learning_rate=xgb_params['learning_rate'],
            scale_pos_weight=xgb_params['scale_pos_weight'],
            random_state=config['data']['random_state'],
            n_jobs=-1,
            verbosity=0
        )
        print(" XGBoost initialized")

    if not models:
        print("\nERROR: No models enabled in config.yaml")
        exit(1)

    print(f"\nTotal models to train: {len(models)}")

    return models


def train_and_evaluate(models, X, y, config):
    """Train all models and compare performance"""
    print_header("TRAINING & EVALUATION")

    # Split data
    test_size = config['data']['test_size']
    random_state = config['data']['random_state']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"\nTrain set: {len(X_train):,} samples")
    print(f"Test set:  {len(X_test):,} samples")

    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # Train each model
    for name, model in models.items():
        print(f"\n{'=' * 80}")
        print(f"Training: {name}")
        print(f"{'=' * 80}")

        # Train
        print("Fitting model...")
        model.fit(X_train_scaled, y_train)
        print(" Training complete!")

        # Predictions
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)

        # Probabilities (if available)
        if hasattr(model, 'predict_proba'):
            y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
            y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_train_proba = y_train_pred
            y_test_proba = y_test_pred

        # Evaluate
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
        test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
        test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)

        # ROC-AUC (for binary classification)
        try:
            test_roc_auc = roc_auc_score(y_test, y_test_proba)
        except:
            test_roc_auc = None

        # Cross-validation
        print("\nPerforming 5-fold cross-validation...")
        cv_folds = min(5, config['training']['cross_validation_folds'])
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=skf, scoring='accuracy')

        # Print results
        print(f"\nResults for {name}:")
        print(f"{'=' * 50}")
        print(f"Train Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"Test Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"Test Precision: {test_precision:.4f} ({test_precision*100:.2f}%)")
        print(f"Test Recall:    {test_recall:.4f} ({test_recall*100:.2f}%)")
        print(f"Test F1-Score:  {test_f1:.4f}")
        if test_roc_auc:
            print(f"Test ROC-AUC:   {test_roc_auc:.4f}")
        print(f"\nCross-Val Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_test_pred)
        print(f"\nConfusion Matrix:")
        print(cm)

        # Store results
        results[name] = {
            'model': model,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'test_roc_auc': test_roc_auc,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }

    return results, scaler, X_train, X_test, y_train, y_test


def select_best_model(results):
    """Select the best performing model"""
    print_header("MODEL COMPARISON")

    print(f"\n{'Model':<25} {'Test Acc':<12} {'F1-Score':<12} {'CV Acc':<12}")
    print("=" * 80)

    best_model_name = None
    best_score = 0

    for name, metrics in results.items():
        print(f"{name:<25} {metrics['test_accuracy']:<12.4f} {metrics['test_f1']:<12.4f} {metrics['cv_mean']:<12.4f}")

        # Use F1-score as primary metric for best model
        if metrics['test_f1'] > best_score:
            best_score = metrics['test_f1']
            best_model_name = name

    print("\n" + "=" * 80)
    print(f" BEST MODEL: {best_model_name} (F1-Score: {best_score:.4f})")
    print("=" * 80)

    return best_model_name, results[best_model_name]


def save_model_artifacts(best_model_name, best_result, scaler, feature_names):
    """Save trained model and artifacts"""
    print_header("SAVING MODEL ARTIFACTS")

    os.makedirs('models', exist_ok=True)

    # Save model
    model_path = os.path.join('models', 'best_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(best_result['model'], f)
    print(f" Saved model: {model_path}")

    # Save scaler
    scaler_path = os.path.join('models', 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f" Saved scaler: {scaler_path}")

    # Save feature names
    features_path = os.path.join('models', 'feature_columns.pkl')
    with open(features_path, 'wb') as f:
        pickle.dump(feature_names, f)
    print(f" Saved feature columns: {features_path}")

    # Save model info
    model_info = {
        'model_name': best_model_name,
        'train_accuracy': float(best_result['train_accuracy']),
        'test_accuracy': float(best_result['test_accuracy']),
        'test_f1': float(best_result['test_f1']),
        'cv_mean': float(best_result['cv_mean']),
        'n_features': len(feature_names),
        'training_date': datetime.now().isoformat()
    }

    info_path = os.path.join('models', 'model_info.yaml')
    with open(info_path, 'w') as f:
        yaml.dump(model_info, f)
    print(f" Saved model info: {info_path}")

    # Feature importance (if available)
    if hasattr(best_result['model'], 'feature_importances_'):
        print("\nTop 10 Feature Importances:")
        importances = best_result['model'].feature_importances_
        indices = np.argsort(importances)[::-1][:10]

        for i, idx in enumerate(indices, 1):
            print(f"  {i}. {feature_names[idx]}: {importances[idx]:.4f}")

        # Save feature importance
        feat_imp_path = os.path.join('models', 'feature_importance.csv')
        feat_imp_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        feat_imp_df.to_csv(feat_imp_path, index=False)
        print(f"\n Saved feature importance: {feat_imp_path}")


def main():
    """Main training pipeline"""

    print("\n" + "=" * 80)
    print("  DATA INSIGHT ML - MODEL TRAINING")
    print("  Auto-Training & Model Selection")
    print("=" * 80)

    # Load configuration
    config = load_config()

    # Load prepared data
    X, y, metadata = load_prepared_data()

    # Create models
    models = create_models(config)

    # Train and evaluate
    results, scaler, X_train, X_test, y_train, y_test = train_and_evaluate(models, X, y, config)

    # Select best model
    best_model_name, best_result = select_best_model(results)

    # Save artifacts
    save_model_artifacts(best_model_name, best_result, scaler, X.columns.tolist())

    # Final summary
    print_header("TRAINING COMPLETE")

    print("\n Model training completed successfully!")
    print(f"\n  Best Model: {best_model_name}")
    print(f"  Test Accuracy: {best_result['test_accuracy']:.2%}")
    print(f"  F1-Score: {best_result['test_f1']:.4f}")

    print("\n Next step:")
    print("  python app.py")

    print("\n" + "=" * 80 + "\n")


if __name__ == '__main__':
    main()
