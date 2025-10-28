# -*- coding: utf-8 -*-
import sys
import io

# Force UTF-8 encoding for stdout
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Synthetic Data Generator for Mentorship Risk Prediction
Generates realistic synthetic data matching the schema from the original Excel files.

Based on statistics from the original dataset:
- 13,513 records total
- Success rate: 23.9% (3,227 COMPLETED)
- Failure rate: 76.1% (10,286 REJECTED/CANCELLED)
- Class imbalance: 3.19:1
"""

import pandas as pd
import numpy as np
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
N_RECORDS = 13513  # Match original dataset size
SUCCESS_RATE = 0.239  # 23.9% success rate

print("\n" + "=" * 80)
print("  SYNTHETIC DATA GENERATION")
print("  JP Morgan Data for Good Hackathon 2025 - Team 2")
print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("=" * 80)

print(f"\nGenerating {N_RECORDS:,} synthetic mentorship records...")
print(f"Target success rate: {SUCCESS_RATE*100:.1f}%")

# === CATEGORICAL DATA DISTRIBUTIONS ===
# Based on analysis from EDA notebooks

# Workfield categories (mentor field)
workfield_categories = [
    'Computer science',
    'Engineering',
    'Banking-Finance',
    'Accounting, management',
    'Human Resources',
    'Marketing',
    'Healthcare',
    'Teaching',
    'Law',
    'Other'
]
workfield_weights = [0.20, 0.15, 0.12, 0.10, 0.08, 0.07, 0.08, 0.06, 0.04, 0.10]

# Field of study (mentee field)
field_of_study_categories = [
    'IT, IS, Data, Web, Tech',
    'Commerce, Management, Economics, Management',
    'Banking, Insurance and Finance',
    'Accounting, Finance',
    'Engineering',
    'Healthcare, Medicine',
    'Law, Legal Studies',
    'Education, Teaching',
    'Marketing, Communication',
    'Other'
]
field_of_study_weights = [0.25, 0.15, 0.12, 0.10, 0.12, 0.08, 0.05, 0.05, 0.04, 0.04]

# Study level
study_levels = ['Bac+1', 'Bac+2', 'Bac+3', 'Bac+4', 'Bac+5+']
study_level_weights = [0.15, 0.20, 0.30, 0.20, 0.15]  # Most common: Bac+3

# Degree type
degrees = ['Licence', 'BTS', 'Master', 'Autre']
degree_weights = [0.35, 0.25, 0.30, 0.10]

# Needs (mentorship focus)
needs_categories = ['[pro]', '[study]', '[pro, study]']
needs_weights = [0.60, 0.15, 0.25]  # Most want professional mentorship

# Average grade
average_grades = [
    'Not specified (or Not provided)',
    'Below average',
    'Average',
    'Good',
    'Very good',
    'Excellent'
]
grade_weights = [0.50, 0.05, 0.15, 0.15, 0.10, 0.05]

# Program type
programs = ['PP', 'PNP']
program_weights = [0.30, 0.70]  # Mostly PNP

# Desired exchange frequency
frequencies = [
    'Once a week',
    'Once every two weeks (or Bi-weekly)',
    'More than once per week'
]
frequency_weights = [0.45, 0.40, 0.15]

# Binome status (final outcomes only)
statuses = ['COMPLETED', 'REJECTED', 'CANCELLED']
# Adjust weights to achieve 23.9% success rate
status_weights = [SUCCESS_RATE, (1-SUCCESS_RATE)*0.67, (1-SUCCESS_RATE)*0.33]

print("\n✓ Defined categorical distributions")

# === GENERATE BASE DATA ===
print("\nGenerating features...")

data = {
    'binome_id': range(100000, 100000 + N_RECORDS),
    'mentor_id': np.random.randint(50000, 70000, N_RECORDS),
    'mentee_id': np.random.randint(180000, 210000, N_RECORDS),
}

# Categorical features
data['workfield'] = np.random.choice(workfield_categories, N_RECORDS, p=workfield_weights)
data['field_of_study'] = np.random.choice(field_of_study_categories, N_RECORDS, p=field_of_study_weights)
data['study_level'] = np.random.choice(study_levels, N_RECORDS, p=study_level_weights)
data['degree'] = np.random.choice(degrees, N_RECORDS, p=degree_weights)
data['needs'] = np.random.choice(needs_categories, N_RECORDS, p=needs_weights)
data['average_grade'] = np.random.choice(average_grades, N_RECORDS, p=grade_weights)
data['program'] = np.random.choice(programs, N_RECORDS, p=program_weights)
data['desired_exchange_frequency'] = np.random.choice(frequencies, N_RECORDS, p=frequency_weights)

print("✓ Generated categorical features")

# === GENERATE NUMERIC FEATURES WITH REALISTIC CORRELATIONS ===

# Engagement score (0-4, most important predictor)
# Lower engagement → higher failure rate
engagement_base = np.random.beta(2, 3, N_RECORDS) * 4  # Skewed toward lower values

# Binome score (0-12, compatibility score)
binome_base = np.random.normal(3.7, 2.5, N_RECORDS)  # Mean 3.7 from original
binome_base = np.clip(binome_base, 0, 12)

print("✓ Generated numeric features")

# === GENERATE TARGET WITH REALISTIC CORRELATIONS ===
print("\nGenerating target variable with feature correlations...")

# Calculate failure probability for each record based on features
failure_probs = np.full(N_RECORDS, 0.50)  # Base 50% failure

# Adjust based on engagement score (strongest predictor: 24% importance)
engagement_factor = (4 - engagement_base) / 4 * 0.40  # Low engagement → high failure
failure_probs += engagement_factor

# Adjust based on workfield (16% importance)
workfield_failure_rates = {
    'Computer science': 0.68,
    'Engineering': 0.55,
    'Banking-Finance': 0.50,
    'Accounting, management': 0.52,
    'Human Resources': 0.48,
    'Marketing': 0.47,
    'Healthcare': 0.42,
    'Teaching': 0.35,
    'Law': 0.45,
    'Other': 0.50
}
for idx, wf in enumerate(data['workfield']):
    field_rate = workfield_failure_rates.get(wf, 0.50)
    failure_probs[idx] += (field_rate - 0.50) * 0.30

# Adjust based on study level (8% importance)
study_level_failure_rates = {
    'Bac+1': 0.60,
    'Bac+2': 0.52,
    'Bac+3': 0.48,
    'Bac+4': 0.45,
    'Bac+5+': 0.38
}
for idx, sl in enumerate(data['study_level']):
    level_rate = study_level_failure_rates.get(sl, 0.50)
    failure_probs[idx] += (level_rate - 0.50) * 0.20

# Adjust based on needs (12% importance)
needs_failure_rates = {
    '[pro]': 0.58,
    '[study]': 0.48,
    '[pro, study]': 0.28
}
for idx, need in enumerate(data['needs']):
    need_rate = needs_failure_rates.get(need, 0.50)
    failure_probs[idx] += (need_rate - 0.50) * 0.25

# Adjust based on binome score (11% importance)
binome_factor = (binome_base - 6) / 12 * 0.20  # Low score → high failure
failure_probs += binome_factor

# Clip probabilities to valid range
failure_probs = np.clip(failure_probs, 0.05, 0.95)

# Generate target based on probabilities
target = (np.random.random(N_RECORDS) > failure_probs).astype(int)

# Generate binome_statut based on target
binome_statut = []
for t in target:
    if t == 1:
        binome_statut.append('COMPLETED')
    else:
        # Random choice between REJECTED and CANCELLED
        binome_statut.append(np.random.choice(['REJECTED', 'CANCELLED'], p=[0.67, 0.33]))

data['binome_statut'] = binome_statut
data['target'] = target

print("✓ Generated target with realistic correlations")

# === ADJUST ENGAGEMENT AND BINOME SCORES BASED ON TARGET ===
# Add noise and correlation with outcome

# Successful mentorships → higher engagement
engagement_adjusted = engagement_base.copy()
for idx, t in enumerate(target):
    if t == 1:  # Success
        # Boost engagement for successful cases
        engagement_adjusted[idx] += np.random.uniform(0.3, 1.2)
    else:  # Failure
        # Reduce engagement for failures
        engagement_adjusted[idx] -= np.random.uniform(0.0, 0.8)

engagement_adjusted = np.clip(engagement_adjusted, 0, 4)
data['engagement_score'] = engagement_adjusted

# Successful mentorships → higher binome score
binome_adjusted = binome_base.copy()
for idx, t in enumerate(target):
    if t == 1:  # Success
        binome_adjusted[idx] += np.random.uniform(1, 3)
    else:  # Failure
        binome_adjusted[idx] -= np.random.uniform(0, 2)

binome_adjusted = np.clip(binome_adjusted, 0, 12)
data['binome_score'] = binome_adjusted

print("✓ Adjusted scores based on outcomes")

# === ADD MISSING VALUES (REALISTIC) ===
print("\nAdding realistic missing values...")

# 10.2% missing workfield (from original)
missing_workfield_idx = np.random.choice(N_RECORDS, int(N_RECORDS * 0.102), replace=False)
data['workfield'] = np.array(data['workfield'])
data['workfield'][missing_workfield_idx] = np.nan

# 0.4% missing binome_score (from original)
missing_binome_idx = np.random.choice(N_RECORDS, int(N_RECORDS * 0.004), replace=False)
data['binome_score'][missing_binome_idx] = np.nan

print(f"✓ Added missing values: {len(missing_workfield_idx)} workfield, {len(missing_binome_idx)} binome_score")

# === CREATE DATAFRAME ===
df = pd.DataFrame(data)

# Reorder columns to match original
column_order = [
    'binome_id', 'mentor_id', 'workfield', 'mentee_id',
    'field_of_study', 'study_level', 'degree', 'needs',
    'average_grade', 'program', 'engagement_score',
    'desired_exchange_frequency', 'binome_score',
    'target', 'binome_statut'
]
df = df[column_order]

# === VERIFY DATA QUALITY ===
print("\n" + "=" * 80)
print("  DATA QUALITY CHECKS")
print("=" * 80)

# Check target distribution
success_count = (df['target'] == 1).sum()
failure_count = (df['target'] == 0).sum()
actual_success_rate = success_count / len(df)

print(f"\nTarget Distribution:")
print(f"  Success (1): {success_count:,} ({actual_success_rate*100:.1f}%)")
print(f"  Failure (0): {failure_count:,} ({(1-actual_success_rate)*100:.1f}%)")
print(f"  Imbalance ratio: {failure_count/success_count:.2f}:1")

if abs(actual_success_rate - SUCCESS_RATE) < 0.05:
    print(f"  ✓ Target distribution matches expected ({SUCCESS_RATE*100:.1f}%)")
else:
    print(f"  ⚠ Target distribution differs from expected ({SUCCESS_RATE*100:.1f}%)")

# Check missing values
missing = df.isnull().sum()
print(f"\nMissing Values:")
for col in missing[missing > 0].index:
    count = missing[col]
    pct = count / len(df) * 100
    print(f"  {col}: {count:,} ({pct:.1f}%)")

# Feature statistics
print(f"\nNumeric Feature Statistics:")
print(f"  engagement_score: mean={df['engagement_score'].mean():.2f}, "
      f"std={df['engagement_score'].std():.2f}, "
      f"range=[{df['engagement_score'].min():.1f}, {df['engagement_score'].max():.1f}]")
print(f"  binome_score: mean={df['binome_score'].mean():.2f}, "
      f"std={df['binome_score'].std():.2f}, "
      f"range=[{df['binome_score'].min():.1f}, {df['binome_score'].max():.1f}]")

# Check correlations with target
print(f"\nFeature-Target Correlations:")
print(f"  engagement_score: {df[['engagement_score', 'target']].corr().iloc[0,1]:.3f}")
print(f"  binome_score: {df[['binome_score', 'target']].corr().iloc[0,1]:.3f}")

# === SAVE TO CSV ===
print("\n" + "=" * 80)
print("  SAVING DATASET")
print("=" * 80)

output_file = 'ml_ready_dataset.csv'
df.to_csv(output_file, index=False)

import os
file_size = os.path.getsize(output_file) / (1024 * 1024)

print(f"\n✓ Saved: {output_file}")
print(f"  File size: {file_size:.2f} MB")
print(f"  Records: {len(df):,}")
print(f"  Columns: {len(df.columns)}")

# Display sample records
print(f"\nSample Records (first 5):")
print(df.head().to_string())

print("\n" + "=" * 80)
print("  GENERATION COMPLETE")
print("=" * 80)

print(f"\n✅ Synthetic data generation complete!")
print(f"\n📊 Dataset Summary:")
print(f"  - File: {output_file}")
print(f"  - Records: {len(df):,}")
print(f"  - Features: 10 (excluding IDs and target)")
print(f"  - Target: binary (1=success, 0=failure)")
print(f"  - Success rate: {actual_success_rate*100:.1f}%")

print(f"\n🎯 Next Step:")
print(f"  Run model training: python train_model.py")

print("\n" + "=" * 80 + "\n")
