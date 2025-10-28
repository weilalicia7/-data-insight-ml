"""
Create Messy Demo Dataset to Demonstrate Data Cleaning
Simulates common data quality issues in NGO datasets
"""

import pandas as pd
import numpy as np

np.random.seed(42)

print("\n" + "=" * 60)
print("CREATING MESSY DEMO DATASET")
print("Simulating common NGO data quality issues")
print("=" * 60)

# Create a messy student dataset with common problems
n_samples = 50

data = {
    'student_id': [f'STU{str(i).zfill(4)}' for i in range(1001, 1051)],
    'student_name': [
        'John Smith', 'mary johnson', '  Sarah Davis  ', 'MIKE WILSON',
        'Emma Brown', 'James Lee', 'sophia garcia', None, 'Liam Martinez',
        'Olivia Rodriguez', 'Noah Hernandez', 'Ava Lopez', 'William Gonzalez',
        'Isabella Moore', 'James Anderson', None, 'Charlotte Thomas',
        'Benjamin Taylor', 'Amelia White', 'Lucas Harris'
    ] * 3,  # Repeat to get 60, then trim

    'age': [
        15, 16, None, 17, 14, 18, 16, 15, None, 17,
        16, 15, 14, 16, 17, 15, None, 16, 17, 14,
        15, 16, 18, 17, 16, 15, 14, None, 16, 17,
        15, 16, 17, 14, 15, 16, None, 18, 17, 16,
        15, 14, 16, 17, 15, None, 16, 17, 14, 15
    ],

    'grade_level': [
        9, 10, 11, 12, 9, 12, 10, 9, 11, 12,
        10, 9, 9, 10, 11, 9, 10, 10, 11, 9,
        9, 10, 12, 11, 10, 9, 9, 10, 10, 11,
        9, 10, 11, 9, 9, 10, 11, 12, 11, 10,
        9, 9, 10, 11, 9, 10, 10, 11, 9, 9
    ],

    # Attendance rate with outliers and missing values
    'attendance_rate': [
        0.85, 0.92, None, 0.78, 1.5, 0.88, 0.91, None, 0.76, 0.89,
        0.95, 0.82, 0.87, None, 0.79, 0.93, 0.84, 0.88, -0.1, 0.90,
        0.86, None, 0.81, 0.94, 0.83, 0.87, 0.91, 0.85, None, 0.92,
        0.88, 0.86, None, 0.89, 0.93, 0.84, 0.87, 0.90, 0.85, None,
        0.91, 0.86, 0.88, None, 0.84, 0.92, 0.87, 0.89, 0.85, 0.90
    ],

    # GPA with outliers and missing values
    'gpa': [
        3.5, 3.8, 2.9, None, 3.2, 3.9, 3.6, 2.7, 3.1, None,
        3.7, 3.3, 2.8, 3.5, None, 3.4, 3.2, 3.8, 5.0, 3.6,  # 5.0 is invalid
        3.1, 2.9, None, 3.7, 3.4, 3.5, 2.8, 3.2, 3.9, None,
        3.6, 3.3, 3.1, None, 3.8, 3.5, 2.9, 3.7, 3.4, 3.2,
        None, 3.6, 3.3, 3.8, 3.5, None, 3.4, 3.7, 3.1, 3.9
    ],

    'absences_per_month': [
        2, 1, 3, 4, None, 2, 1, 3, 5, 2,
        1, 2, None, 3, 4, 1, 2, 2, 15, 1,  # 15 is outlier
        3, 2, None, 1, 2, 3, 2, 4, 1, 2,
        None, 3, 2, 1, 2, 3, None, 2, 3, 4,
        2, 1, None, 2, 3, 2, 1, None, 3, 2
    ],

    # Inconsistent categorical formatting
    'parent_involvement': [
        'Low', 'High', 'medium', 'LOW', 'Medium', None, 'high', 'Low', 'HIGH', 'Medium',
        'low', 'High', 'Medium', None, 'low', 'high', 'Medium', 'Low', 'MEDIUM', 'High',
        'low', None, 'Medium', 'high', 'Low', 'medium', 'High', 'LOW', 'Medium', None,
        'High', 'low', 'Medium', 'HIGH', None, 'Low', 'medium', 'High', 'low', 'Medium',
        'High', None, 'low', 'Medium', 'HIGH', 'Low', 'medium', None, 'High', 'Low'
    ],

    'economic_status': [
        'Low Income', 'Middle Income', None, 'low income', 'HIGH INCOME', 'Middle Income',
        'Low Income', None, 'middle income', 'High Income', 'Low Income', 'Middle Income',
        None, 'low income', 'High Income', 'Middle Income', 'LOW INCOME', None, 'High Income',
        'Middle Income', 'Low Income', None, 'MIDDLE INCOME', 'High Income', 'Low Income',
        'middle income', None, 'High Income', 'Low Income', 'Middle Income', 'low income',
        None, 'High Income', 'Middle Income', 'Low Income', 'HIGH INCOME', None, 'Middle Income',
        'Low Income', 'high income', None, 'Middle Income', 'Low Income', 'HIGH INCOME', None,
        'Middle Income', 'low income', 'High Income', 'Low Income', 'MIDDLE INCOME'
    ],

    'behavioral_incidents': [
        0, 1, 2, None, 1, 0, 2, 1, None, 0,
        1, 2, 0, None, 3, 1, 0, 2, 20, 1,  # 20 is outlier
        None, 2, 0, 1, 2, None, 1, 0, 2, 1,
        0, None, 2, 1, 0, 1, None, 2, 1, 0,
        1, 2, None, 0, 1, 2, None, 1, 0, 2
    ],

    'extracurricular_activities': [
        2, 3, 1, None, 2, 4, 1, 3, None, 2,
        3, 1, 2, None, 1, 3, 2, 4, None, 1,
        2, 3, None, 2, 1, 3, None, 2, 1, 3,
        2, None, 3, 1, 2, 4, None, 1, 2, 3,
        None, 2, 1, 3, 2, None, 3, 1, 2, 4
    ],

    'tutoring_hours': [
        2.5, None, 3.0, 1.5, 4.0, None, 2.0, 3.5, 1.0, None,
        2.5, 3.0, None, 2.0, 1.5, 3.5, None, 2.5, 3.0, 2.0,
        None, 1.5, 3.0, 2.5, None, 4.0, 2.0, 3.0, None, 1.5,
        2.5, 3.0, None, 2.0, 3.5, None, 1.5, 3.0, 2.5, 4.0,
        None, 2.0, 3.0, 2.5, None, 1.5, 3.5, 2.0, None, 3.0
    ],

    'family_size': [
        4, 5, 3, None, 4, 5, None, 3, 4, 5,
        3, 4, None, 5, 4, 3, None, 5, 4, 3,
        None, 4, 5, 3, 4, None, 5, 3, 4, 5,
        3, None, 4, 5, 3, 4, None, 5, 4, 3,
        4, None, 5, 3, 4, 5, None, 3, 4, 5
    ]
}

# Trim to exact size
for key in data.keys():
    data[key] = data[key][:n_samples]

df = pd.DataFrame(data)

# Add some duplicate rows
df = pd.concat([df, df.iloc[[0, 1, 2]]], ignore_index=True)

print(f"\n[INFO] Created messy dataset with {len(df)} rows")
print(f"[INFO] Data quality issues included:")
print("  - Missing values (represented as None/NaN)")
print("  - Inconsistent categorical formatting (Low/low/LOW)")
print("  - Inconsistent spacing in names")
print("  - Invalid values (GPA > 4.0, attendance_rate > 1.0 or < 0)")
print("  - Outliers (very high absences, behavioral incidents)")
print("  - Duplicate rows")

# Save messy data
output_file = 'messy_student_data.csv'
df.to_csv(output_file, index=False)

print(f"\n[SUCCESS] Saved messy dataset to: {output_file}")
print("\nTo clean this data, run:")
print(f"  python data_cleaner.py {output_file} student")

print("\n" + "=" * 60)
print("\nPreview of messy data:")
print("=" * 60)
print(df.head(10).to_string(index=False))

print("\n" + "=" * 60)
print("Data Quality Issues Summary:")
print("=" * 60)
print(f"Total missing values: {df.isnull().sum().sum()}")
print(f"Rows with missing data: {df.isnull().any(axis=1).sum()}")
print(f"Duplicate rows: {df.duplicated().sum()}")
print("=" * 60)
