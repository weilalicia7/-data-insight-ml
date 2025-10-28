"""
Domain Setup Script
Creates domain structure with synthetic data and pre-trained models
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import example data generator
from example_data_generator import (
    generate_donor_dataset,
    generate_program_completion_dataset,
    generate_grant_application_dataset,
    generate_customer_churn_dataset,
    generate_student_dropout_dataset,
    generate_child_wellbeing_dataset
)


def create_domain_structure(domain_name):
    """Create directory structure for a domain"""
    domain_path = os.path.join('domains', domain_name)
    os.makedirs(domain_path, exist_ok=True)
    return domain_path


def train_domain_model(df, target_column, domain_name):
    """
    Train a model for a specific domain.

    Args:
        df: DataFrame with features and target
        target_column: Name of target column
        domain_name: Name of the domain

    Returns:
        Dictionary with model, scaler, features, and metrics
    """
    print(f"\n{'=' * 60}")
    print(f"Training model for: {domain_name}")
    print(f"{'=' * 60}")

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Drop ID columns
    id_cols = [col for col in X.columns if 'id' in col.lower()]
    X = X.drop(columns=id_cols, errors='ignore')

    # One-hot encode categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # Get feature names
    feature_columns = X.columns.tolist()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest
    print(f"Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=7,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f"[OK] Training complete!")
    print(f"  Accuracy:  {accuracy:.2%}")
    print(f"  Precision: {precision:.2%}")
    print(f"  Recall:    {recall:.2%}")
    print(f"  F1-Score:  {f1:.4f}")

    return {
        'model': model,
        'scaler': scaler,
        'features': feature_columns,
        'metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        }
    }


def create_domain(domain_name, display_name, description, icon, use_case,
                  data_generator, target_column, target_description,
                  sample_inputs, recommendations):
    """
    Create a complete domain with data, model, and metadata.

    Args:
        domain_name: Internal name (e.g., 'donor_retention')
        display_name: Display name (e.g., 'Donor Retention')
        description: Short description
        icon: Emoji icon
        use_case: Use case description
        data_generator: Function to generate synthetic data
        target_column: Name of target variable
        target_description: Description of target variable
        sample_inputs: List of sample prediction examples
        recommendations: Dictionary of recommendation actions
    """
    print(f"\n{'#' * 60}")
    print(f"Creating domain: {display_name}")
    print(f"{'#' * 60}")

    # Create directory
    domain_path = create_domain_structure(domain_name)

    # Generate synthetic data
    print(f"\nGenerating synthetic data...")
    df = data_generator()
    print(f"[OK] Generated {len(df)} rows")

    # Save data
    data_path = os.path.join(domain_path, 'example_data.csv')
    df.to_csv(data_path, index=False)
    print(f"[OK] Saved: {data_path}")

    # Train model
    trained = train_domain_model(df, target_column, domain_name)

    # Save model artifacts
    model_path = os.path.join(domain_path, 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(trained['model'], f)
    print(f"[OK] Saved model: {model_path}")

    scaler_path = os.path.join(domain_path, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(trained['scaler'], f)
    print(f"[OK] Saved scaler: {scaler_path}")

    features_path = os.path.join(domain_path, 'features.pkl')
    with open(features_path, 'wb') as f:
        pickle.dump(trained['features'], f)
    print(f"[OK] Saved features: {features_path}")

    # Create metadata
    metadata = {
        'domain': domain_name,
        'display_name': display_name,
        'description': description,
        'icon': icon,
        'use_case': use_case,
        'target_variable': target_column,
        'target_description': target_description,
        'features_count': len(trained['features']),
        'example_rows': len(df),
        'model_type': 'Random Forest',
        'accuracy': trained['metrics']['accuracy'],
        'precision': trained['metrics']['precision'],
        'recall': trained['metrics']['recall'],
        'f1_score': trained['metrics']['f1_score'],
        'training_date': datetime.now().strftime('%Y-%m-%d'),
        'recommendations': recommendations,
        'sample_inputs': sample_inputs
    }

    metadata_path = os.path.join(domain_path, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"[OK] Saved metadata: {metadata_path}")

    # Create README
    readme_content = f"""# {display_name}

## Description
{description}

## Use Case
{use_case}

## Model Information
- **Model Type:** Random Forest
- **Accuracy:** {trained['metrics']['accuracy']:.2%}
- **Precision:** {trained['metrics']['precision']:.2%}
- **Recall:** {trained['metrics']['recall']:.2%}
- **F1-Score:** {trained['metrics']['f1_score']:.4f}

## Target Variable
- **Name:** {target_column}
- **Description:** {target_description}

## Features
Total features: {len(trained['features'])}

## Example Data
Rows available: {len(df)}

## Getting Started

```python
from domain_manager import DomainManager

# Load domain
dm = DomainManager()
dm.load_domain('{domain_name}')

# Make prediction
result = dm.predict({{
    # ... your feature values
}})

print(result)
```

## Sample Predictions

{json.dumps(sample_inputs, indent=2)}

## Generated
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    readme_path = os.path.join(domain_path, 'README.md')
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    print(f"[OK] Saved README: {readme_path}")

    print(f"\n[SUCCESS] Domain '{domain_name}' created successfully!")

    return domain_path


def setup_all_domains():
    """Setup all available domains"""

    print("\n" + "=" * 60)
    print("  DATA INSIGHT ML - DOMAIN SETUP")
    print("  Setting up multi-domain system")
    print("=" * 60)

    # Create domains directory
    os.makedirs('domains', exist_ok=True)

    # Domain 1: Donor Retention
    create_domain(
        domain_name='donor_retention',
        display_name='Donor Retention Prediction',
        description='Predict which donors are likely to donate again',
        icon='💰',
        use_case='Optimize fundraising by focusing on high-probability donors',
        data_generator=lambda: generate_donor_dataset(1000),
        target_column='donated_again',
        target_description='1 = Will donate again, 0 = Will not donate',
        sample_inputs=[
            {
                'name': 'High-value engaged donor',
                'description': 'Regular donor with high engagement',
                'expected': 'Will donate (high confidence)'
            },
            {
                'name': 'At-risk donor',
                'description': 'Infrequent donor with low engagement',
                'expected': 'Will not donate (high confidence)'
            }
        ],
        recommendations={
            'high_confidence_threshold': 0.8,
            'action_positive': 'Prioritize for next fundraising campaign',
            'action_negative': 'Initiate re-engagement campaign with personalized outreach'
        }
    )

    # Domain 2: Program Completion
    create_domain(
        domain_name='program_completion',
        display_name='Program Completion Prediction',
        description='Predict which program participants will complete successfully',
        icon='🎓',
        use_case='Identify at-risk participants early and provide targeted support',
        data_generator=lambda: generate_program_completion_dataset(800),
        target_column='completed',
        target_description='1 = Completed program, 0 = Did not complete',
        sample_inputs=[
            {
                'name': 'High-attendance participant',
                'description': 'Strong engagement and mentor support',
                'expected': 'Will complete (high confidence)'
            },
            {
                'name': 'At-risk participant',
                'description': 'Low attendance and engagement',
                'expected': 'At risk of dropout (high confidence)'
            }
        ],
        recommendations={
            'high_confidence_threshold': 0.75,
            'action_positive': 'Continue current support level',
            'action_negative': 'Assign additional mentor and schedule check-in'
        }
    )

    # Domain 3: Grant Scoring
    create_domain(
        domain_name='grant_scoring',
        display_name='Grant Application Scoring',
        description='Score and prioritize grant applications automatically',
        icon='📋',
        use_case='Efficiently evaluate applications and identify top candidates',
        data_generator=lambda: generate_grant_application_dataset(500),
        target_column='approved',
        target_description='1 = Approved, 0 = Rejected',
        sample_inputs=[
            {
                'name': 'Strong application',
                'description': 'Established org with strong proposal',
                'expected': 'Approve (high confidence)'
            },
            {
                'name': 'Weak application',
                'description': 'New org with unclear objectives',
                'expected': 'Reject (medium confidence)'
            }
        ],
        recommendations={
            'high_confidence_threshold': 0.8,
            'action_positive': 'Fast-track for review board',
            'action_negative': 'Request additional information or decline'
        }
    )

    # Domain 4: Customer/Member Churn
    create_domain(
        domain_name='customer_churn',
        display_name='Member Churn Prediction',
        description='Predict which members are at risk of churning',
        icon='👥',
        use_case='Retain members through proactive engagement',
        data_generator=lambda: generate_customer_churn_dataset(1000),
        target_column='churned',
        target_description='1 = Churned, 0 = Retained',
        sample_inputs=[
            {
                'name': 'Engaged member',
                'description': 'Active usage and long tenure',
                'expected': 'Low churn risk'
            },
            {
                'name': 'At-risk member',
                'description': 'Declining activity and support tickets',
                'expected': 'High churn risk'
            }
        ],
        recommendations={
            'high_confidence_threshold': 0.75,
            'action_positive': 'Standard retention efforts',
            'action_negative': 'Urgent: Offer retention incentive and personal outreach'
        }
    )

    # Domain 5: Student Dropout Risk
    create_domain(
        domain_name='student_dropout',
        display_name='Student Dropout Risk Prediction',
        description='Identify students at risk of dropping out early',
        icon='📚',
        use_case='Provide early intervention to keep students in school',
        data_generator=lambda: generate_student_dropout_dataset(600),
        target_column='at_risk',
        target_description='1 = At risk of dropout, 0 = On track',
        sample_inputs=[
            {
                'name': 'High-risk student',
                'description': 'Low attendance, poor grades, limited parent involvement',
                'expected': 'At risk (high confidence)'
            },
            {
                'name': 'Thriving student',
                'description': 'Good attendance, strong GPA, active in extracurriculars',
                'expected': 'On track (high confidence)'
            }
        ],
        recommendations={
            'high_confidence_threshold': 0.75,
            'action_positive': 'Continue monitoring and support',
            'action_negative': 'Urgent: Assign counselor, contact parents, provide tutoring'
        }
    )

    # Domain 6: Child Wellbeing Risk
    create_domain(
        domain_name='child_wellbeing',
        display_name='Child Wellbeing Risk Assessment',
        description='Identify children who need additional support',
        icon='🧒',
        use_case='Ensure all children receive necessary care and support',
        data_generator=lambda: generate_child_wellbeing_dataset(500),
        target_column='needs_support',
        target_description='1 = Needs support, 0 = Doing well',
        sample_inputs=[
            {
                'name': 'Thriving child',
                'description': 'Good nutrition, health checkups, supportive home',
                'expected': 'Doing well (high confidence)'
            },
            {
                'name': 'At-risk child',
                'description': 'Poor nutrition, missed checkups, challenging home environment',
                'expected': 'Needs support (high confidence)'
            }
        ],
        recommendations={
            'high_confidence_threshold': 0.75,
            'action_positive': 'Continue routine monitoring',
            'action_negative': 'Priority: Provide nutrition support, health services, family counseling'
        }
    )

    # Final summary
    print("\n" + "=" * 60)
    print("  SETUP COMPLETE!")
    print("=" * 60)

    print("\n[SUCCESS] All domains created successfully!")
    print(f"\nDomains available:")
    print("  [*] Donor Retention")
    print("  [*] Program Completion")
    print("  [*] Grant Scoring")
    print("  [*] Member Churn")
    print("  [*] Student Dropout Risk")
    print("  [*] Child Wellbeing")

    print("\n[INFO] Domain files created in: ./domains/")

    print("\n[NEXT STEPS]")
    print("  1. Test domains: python domain_manager.py")
    print("  2. Start API: python app.py")
    print("  3. Open demo: demo.html")

    print("\n" + "=" * 60 + "\n")


if __name__ == '__main__':
    setup_all_domains()
