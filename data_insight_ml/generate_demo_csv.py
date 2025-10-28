"""
Generate Demo CSV Files for Upload Testing
Creates small CSV files for each domain to test the upload functionality
"""

import pandas as pd
import numpy as np

np.random.seed(100)

# 1. Donor Retention Demo CSV
print("Generating Donor Retention demo CSV...")
donor_data = {
    'last_donation_amount': np.random.exponential(100, 15).round(2),
    'donation_frequency': np.random.poisson(3, 15),
    'years_since_first': np.random.exponential(2, 15).round(1),
    'email_opens': np.random.poisson(8, 15),
    'age': np.random.normal(45, 15, 15).clip(18, 90).round(0),
    'region': np.random.choice(['North', 'South', 'East', 'West'], 15)
}
donor_df = pd.DataFrame(donor_data)
donor_df.to_csv('demo_upload_donor_retention.csv', index=False)
print(f"  Created: demo_upload_donor_retention.csv ({len(donor_df)} rows)")

# 2. Program Completion Demo CSV
print("\nGenerating Program Completion demo CSV...")
program_data = {
    'age': np.random.normal(28, 8, 15).clip(18, 65).round(0),
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 15),
    'income_level': np.random.choice(['Low', 'Medium', 'High'], 15),
    'attendance_rate': np.random.beta(8, 2, 15).round(2),
    'engagement_score': np.random.normal(7, 2, 15).clip(1, 10).round(1),
    'mentor_assigned': np.random.choice([0, 1], 15),
    'hours_per_week': np.random.normal(15, 5, 15).clip(5, 40).round(1)
}
program_df = pd.DataFrame(program_data)
program_df.to_csv('demo_upload_program_completion.csv', index=False)
print(f"  Created: demo_upload_program_completion.csv ({len(program_df)} rows)")

# 3. Grant Scoring Demo CSV
print("\nGenerating Grant Scoring demo CSV...")
grant_data = {
    'org_size': np.random.choice(['Small', 'Medium', 'Large'], 15),
    'years_operating': np.random.exponential(5, 15).clip(0, 50).round(1),
    'budget': np.random.exponential(100000, 15).round(0),
    'previous_grants': np.random.poisson(2, 15),
    'mission_alignment': np.random.choice(['Low', 'Medium', 'High'], 15),
    'proposal_quality': np.random.normal(6, 2, 15).clip(1, 10).round(1),
    'staff_size': np.random.exponential(10, 15).clip(1, 100).round(0)
}
grant_df = pd.DataFrame(grant_data)
grant_df.to_csv('demo_upload_grant_scoring.csv', index=False)
print(f"  Created: demo_upload_grant_scoring.csv ({len(grant_df)} rows)")

# 4. Member Churn Demo CSV
print("\nGenerating Member Churn demo CSV...")
churn_data = {
    'months_member': np.random.exponential(12, 15).round(1),
    'monthly_spending': np.random.exponential(50, 15).round(2),
    'support_tickets': np.random.poisson(1, 15),
    'feature_usage_count': np.random.poisson(10, 15),
    'referrals_made': np.random.poisson(1, 15),
    'account_type': np.random.choice(['Basic', 'Premium', 'Enterprise'], 15),
    'payment_method': np.random.choice(['Card', 'Bank', 'PayPal'], 15)
}
churn_df = pd.DataFrame(churn_data)
churn_df.to_csv('demo_upload_customer_churn.csv', index=False)
print(f"  Created: demo_upload_customer_churn.csv ({len(churn_df)} rows)")

# 5. Student Dropout Demo CSV
print("\nGenerating Student Dropout demo CSV...")
student_data = {
    'age': np.random.normal(16, 2, 15).clip(12, 19).round(0),
    'grade_level': np.random.choice([6, 7, 8, 9, 10, 11, 12], 15),
    'attendance_rate': np.random.beta(5, 2, 15).round(2),
    'gpa': np.random.normal(2.5, 0.8, 15).clip(0, 4.0).round(2),
    'absences_per_month': np.random.poisson(3, 15),
    'parent_involvement': np.random.choice(['Low', 'Medium', 'High'], 15),
    'economic_status': np.random.choice(['Low Income', 'Middle Income', 'High Income'], 15),
    'behavioral_incidents': np.random.poisson(1, 15),
    'extracurricular_activities': np.random.poisson(1, 15).clip(0, 5),
    'tutoring_hours': np.random.exponential(2, 15).clip(0, 10).round(1),
    'family_size': np.random.poisson(4, 15).clip(2, 8)
}
student_df = pd.DataFrame(student_data)
student_df.to_csv('demo_upload_student_dropout.csv', index=False)
print(f"  Created: demo_upload_student_dropout.csv ({len(student_df)} rows)")

# 6. Child Wellbeing Demo CSV
print("\nGenerating Child Wellbeing demo CSV...")
child_data = {
    'age': np.random.normal(8, 3, 15).clip(3, 14).round(0),
    'nutrition_score': np.random.normal(7, 2, 15).clip(1, 10).round(1),
    'health_checkups_per_year': np.random.poisson(2, 15).clip(0, 6),
    'school_attendance_rate': np.random.beta(6, 2, 15).round(2),
    'family_income_level': np.random.choice(['Very Low', 'Low', 'Medium', 'High'], 15),
    'caregiver_education': np.random.choice(['None', 'Primary', 'Secondary', 'Higher'], 15),
    'siblings_count': np.random.poisson(2, 15).clip(0, 6),
    'home_environment_score': np.random.normal(6, 2, 15).clip(1, 10).round(1),
    'access_to_clean_water': np.random.choice(['No', 'Yes'], 15, p=[0.3, 0.7]),
    'vaccination_status': np.random.choice(['Incomplete', 'Complete'], 15, p=[0.35, 0.65]),
    'behavioral_issues': np.random.choice(['None', 'Mild', 'Moderate', 'Severe'], 15, p=[0.5, 0.3, 0.15, 0.05])
}
child_df = pd.DataFrame(child_data)
child_df.to_csv('demo_upload_child_wellbeing.csv', index=False)
print(f"  Created: demo_upload_child_wellbeing.csv ({len(child_df)} rows)")

print("\n" + "="*60)
print("All demo CSV files created successfully!")
print("="*60)
print("\nFiles created:")
print("  1. demo_upload_donor_retention.csv")
print("  2. demo_upload_program_completion.csv")
print("  3. demo_upload_grant_scoring.csv")
print("  4. demo_upload_customer_churn.csv")
print("  5. demo_upload_student_dropout.csv")
print("  6. demo_upload_child_wellbeing.csv")
print("\nThese files are ready to upload through the demo.html interface!")
print("Each file contains 15 sample records with realistic data.")
