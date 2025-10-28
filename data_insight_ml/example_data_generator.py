"""
Example Data Generator
Creates sample datasets for testing the toolkit
"""

import pandas as pd
import numpy as np


def generate_donor_dataset(n_samples=1000):
    """Generate sample donor retention dataset"""

    np.random.seed(42)

    data = {
        'donor_id': range(1, n_samples + 1),
        'last_donation_amount': np.random.exponential(100, n_samples),
        'donation_frequency': np.random.poisson(3, n_samples),
        'years_since_first': np.random.exponential(2, n_samples),
        'email_opens': np.random.poisson(8, n_samples),
        'age': np.random.normal(45, 15, n_samples).clip(18, 90),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples)
    }

    df = pd.DataFrame(data)

    # Generate target based on features (with some noise)
    retention_score = (
        df['donation_frequency'] * 0.3 +
        df['email_opens'] * 0.2 +
        df['last_donation_amount'] / 50 * 0.2 +
        df['years_since_first'] * 0.15 +
        np.random.normal(0, 1, n_samples)
    )

    df['donated_again'] = (retention_score > retention_score.median()).astype(int)

    return df


def generate_program_completion_dataset(n_samples=500):
    """Generate sample program completion dataset"""

    np.random.seed(42)

    data = {
        'participant_id': range(1, n_samples + 1),
        'age': np.random.normal(28, 8, n_samples).clip(18, 65),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'income_level': np.random.choice(['Low', 'Medium', 'High'], n_samples),
        'attendance_rate': np.random.beta(8, 2, n_samples),
        'engagement_score': np.random.normal(7, 2, n_samples).clip(1, 10),
        'mentor_assigned': np.random.choice([0, 1], n_samples),
        'hours_per_week': np.random.normal(15, 5, n_samples).clip(5, 40)
    }

    df = pd.DataFrame(data)

    # Generate completion based on features
    completion_score = (
        df['attendance_rate'] * 5 +
        df['engagement_score'] * 0.5 +
        df['mentor_assigned'] * 2 +
        df['hours_per_week'] * 0.1 +
        np.random.normal(0, 1, n_samples)
    )

    df['completed'] = (completion_score > completion_score.median()).astype(int)

    return df


def generate_grant_application_dataset(n_samples=300):
    """Generate sample grant application dataset"""

    np.random.seed(42)

    data = {
        'application_id': range(1, n_samples + 1),
        'org_size': np.random.choice(['Small', 'Medium', 'Large'], n_samples),
        'years_operating': np.random.exponential(5, n_samples).clip(0, 50),
        'budget': np.random.exponential(100000, n_samples),
        'previous_grants': np.random.poisson(2, n_samples),
        'mission_alignment': np.random.choice(['Low', 'Medium', 'High'], n_samples),
        'proposal_quality': np.random.normal(6, 2, n_samples).clip(1, 10),
        'staff_size': np.random.exponential(10, n_samples).clip(1, 100)
    }

    df = pd.DataFrame(data)

    # Generate approval based on features
    approval_score = (
        df['proposal_quality'] * 0.8 +
        df['previous_grants'] * 0.5 +
        (df['mission_alignment'] == 'High').astype(int) * 2 +
        np.log1p(df['years_operating']) * 0.3 +
        np.random.normal(0, 1.5, n_samples)
    )

    df['approved'] = (approval_score > approval_score.quantile(0.6)).astype(int)

    return df


def generate_customer_churn_dataset(n_samples=800):
    """Generate sample customer/member churn dataset"""

    np.random.seed(42)

    data = {
        'customer_id': range(1, n_samples + 1),
        'months_member': np.random.exponential(12, n_samples),
        'monthly_spending': np.random.exponential(50, n_samples),
        'support_tickets': np.random.poisson(1, n_samples),
        'feature_usage_count': np.random.poisson(10, n_samples),
        'referrals_made': np.random.poisson(1, n_samples),
        'account_type': np.random.choice(['Basic', 'Premium', 'Enterprise'], n_samples),
        'payment_method': np.random.choice(['Card', 'Bank', 'PayPal'], n_samples)
    }

    df = pd.DataFrame(data)

    # Generate churn based on features
    churn_risk = (
        -df['months_member'] * 0.1 +
        -df['monthly_spending'] / 20 +
        df['support_tickets'] * 0.5 +
        -df['feature_usage_count'] * 0.2 +
        -df['referrals_made'] * 0.5 +
        np.random.normal(0, 1, n_samples)
    )

    df['churned'] = (churn_risk > churn_risk.median()).astype(int)

    return df


def generate_student_dropout_dataset(n_samples=600):
    """Generate sample student dropout risk dataset"""

    np.random.seed(43)

    data = {
        'student_id': range(1, n_samples + 1),
        'age': np.random.normal(16, 2, n_samples).clip(12, 19),
        'grade_level': np.random.choice([6, 7, 8, 9, 10, 11, 12], n_samples),
        'attendance_rate': np.random.beta(5, 2, n_samples),
        'gpa': np.random.normal(2.5, 0.8, n_samples).clip(0, 4.0),
        'absences_per_month': np.random.poisson(3, n_samples),
        'parent_involvement': np.random.choice(['Low', 'Medium', 'High'], n_samples),
        'economic_status': np.random.choice(['Low Income', 'Middle Income', 'High Income'], n_samples),
        'behavioral_incidents': np.random.poisson(1, n_samples),
        'extracurricular_activities': np.random.poisson(1, n_samples).clip(0, 5),
        'tutoring_hours': np.random.exponential(2, n_samples).clip(0, 10),
        'family_size': np.random.poisson(4, n_samples).clip(2, 8)
    }

    df = pd.DataFrame(data)

    # Generate dropout risk based on features
    dropout_score = (
        -df['attendance_rate'] * 5 +
        -df['gpa'] * 1.5 +
        df['absences_per_month'] * 0.3 +
        df['behavioral_incidents'] * 0.5 +
        -df['extracurricular_activities'] * 0.4 +
        -df['tutoring_hours'] * 0.2 +
        (df['parent_involvement'] == 'Low').astype(int) * 2 +
        (df['economic_status'] == 'Low Income').astype(int) * 1 +
        np.random.normal(0, 1, n_samples)
    )

    df['at_risk'] = (dropout_score > dropout_score.median()).astype(int)

    return df


def generate_child_wellbeing_dataset(n_samples=500):
    """Generate sample child wellbeing risk assessment dataset"""

    np.random.seed(44)

    data = {
        'child_id': range(1, n_samples + 1),
        'age': np.random.normal(8, 3, n_samples).clip(3, 14),
        'nutrition_score': np.random.normal(7, 2, n_samples).clip(1, 10),
        'health_checkups_per_year': np.random.poisson(2, n_samples).clip(0, 6),
        'school_attendance_rate': np.random.beta(6, 2, n_samples),
        'family_income_level': np.random.choice(['Very Low', 'Low', 'Medium', 'High'], n_samples),
        'caregiver_education': np.random.choice(['None', 'Primary', 'Secondary', 'Higher'], n_samples),
        'siblings_count': np.random.poisson(2, n_samples).clip(0, 6),
        'home_environment_score': np.random.normal(6, 2, n_samples).clip(1, 10),
        'access_to_clean_water': np.random.choice(['No', 'Yes'], n_samples, p=[0.3, 0.7]),
        'vaccination_status': np.random.choice(['Incomplete', 'Complete'], n_samples, p=[0.35, 0.65]),
        'behavioral_issues': np.random.choice(['None', 'Mild', 'Moderate', 'Severe'], n_samples, p=[0.5, 0.3, 0.15, 0.05])
    }

    df = pd.DataFrame(data)

    # Generate wellbeing risk based on features
    wellbeing_score = (
        df['nutrition_score'] * 0.5 +
        df['health_checkups_per_year'] * 0.3 +
        df['school_attendance_rate'] * 3 +
        df['home_environment_score'] * 0.4 +
        (df['access_to_clean_water'] == 'Yes').astype(int) * 2 +
        (df['vaccination_status'] == 'Complete').astype(int) * 1.5 +
        (df['caregiver_education'] == 'Higher').astype(int) * 1 +
        (df['family_income_level'] == 'High').astype(int) * 1 +
        -(df['behavioral_issues'] == 'Severe').astype(int) * 3 +
        np.random.normal(0, 1.5, n_samples)
    )

    # 1 = needs support, 0 = doing well
    df['needs_support'] = (wellbeing_score < wellbeing_score.median()).astype(int)

    return df


def main():
    """Generate all example datasets"""

    print("Generating example datasets...")
    print("=" * 60)

    datasets = {
        'donor_retention': generate_donor_dataset(1000),
        'program_completion': generate_program_completion_dataset(500),
        'grant_applications': generate_grant_application_dataset(300),
        'customer_churn': generate_customer_churn_dataset(800)
    }

    for name, df in datasets.items():
        filename = f'example_{name}.csv'
        df.to_csv(filename, index=False)

        print(f"\n{name.replace('_', ' ').title()}:")
        print(f"  Saved: {filename}")
        print(f"  Samples: {len(df)}")
        print(f"  Features: {len(df.columns) - 1}")  # Exclude target

        # Show target distribution
        target_col = df.columns[-1]
        counts = df[target_col].value_counts()
        print(f"  Target distribution:")
        for val, count in counts.items():
            print(f"    {val}: {count} ({count/len(df)*100:.1f}%)")

    print("\n" + "=" * 60)
    print(" Example datasets created successfully!")
    print("\nTo use an example dataset:")
    print("  python prepare_data.py example_donor_retention.csv")
    print("\n")


if __name__ == '__main__':
    main()
