# -*- coding: utf-8 -*-
import sys
import io

# Force UTF-8 encoding for stdout
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Data Preparation Script
Prepares mentorship data from Excel files to CSV format for model training.

This script:
1. Loads data from Excel files (mentors, mentees, binomes)
2. Merges datasets
3. Cleans and filters data
4. Creates target variable (1=success, 0=failure)
5. Saves to CSV: ml_ready_dataset.csv

Usage:
    python prepare_data.py
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

# File paths (update these to match your data location)
MENTOR_FILE = 'Hackaton_Benevoles_JPMORGAN.xlsx'
MENTEE_FILE = 'Hackaton_Jeunes_JPMORGAN.xlsx'
BINOME_FILE = 'Hackaton_Binomes_JPMORGAN.xlsx'

OUTPUT_FILE = 'ml_ready_dataset.csv'


def print_header(title):
    """Print formatted header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def load_excel_data():
    """Load data from Excel files."""
    print_header("LOADING DATA")

    # Check if files exist
    files = [MENTOR_FILE, MENTEE_FILE, BINOME_FILE]
    for file in files:
        if not os.path.exists(file):
            print(f"⚠ Warning: {file} not found!")
            print(f"  Please update the file path in the script or place the file in the current directory.")

    # Load mentor data
    print(f"\nLoading mentors from: {MENTOR_FILE}")
    try:
        mentor_df = pd.read_excel(MENTOR_FILE)
        print(f"✓ Loaded {len(mentor_df):,} mentors")
        print(f"  Columns: {list(mentor_df.columns[:5])} ...")
    except FileNotFoundError:
        print("✗ File not found. Creating sample data...")
        mentor_df = create_sample_mentor_data()

    # Load mentee data
    print(f"\nLoading mentees from: {MENTEE_FILE}")
    try:
        mentee_df = pd.read_excel(MENTEE_FILE)
        print(f"✓ Loaded {len(mentee_df):,} mentees")
        print(f"  Columns: {list(mentee_df.columns[:5])} ...")
    except FileNotFoundError:
        print("✗ File not found. Creating sample data...")
        mentee_df = create_sample_mentee_data()

    # Load binome (pair) data
    print(f"\nLoading binomes from: {BINOME_FILE}")
    try:
        binome_df = pd.read_excel(BINOME_FILE)
        print(f"✓ Loaded {len(binome_df):,} binomes")
        print(f"  Columns: {list(binome_df.columns[:5])} ...")
    except FileNotFoundError:
        print("✗ File not found. Creating sample data...")
        binome_df = create_sample_binome_data()

    return mentor_df, mentee_df, binome_df


def create_sample_mentor_data():
    """Create sample mentor data for testing."""
    print("  Creating 100 sample mentors...")
    np.random.seed(42)

    workfields = ['Computer science', 'Engineering', 'Banking-Finance',
                 'Accounting, management', 'Human Resources', 'Marketing']

    data = {
        'mentor_id': range(1, 101),
        'workfield': np.random.choice(workfields, 100)
    }

    return pd.DataFrame(data)


def create_sample_mentee_data():
    """Create sample mentee data for testing."""
    print("  Creating 200 sample mentees...")
    np.random.seed(42)

    fields_of_study = ['IT, IS, Data, Web, Tech', 'Banking, Insurance and Finance',
                      'Commerce, Management, Economics, Management', 'Accounting, Finance',
                      'Engineering', 'Other']

    study_levels = ['Bac+1', 'Bac+2', 'Bac+3', 'Bac+4', 'Bac+5+']
    degrees = ['Licence', 'BTS', 'Master', 'Autre']
    needs = ['[pro]', '[study]', '[pro, study]']
    programs = ['PP', 'PNP']
    grades = ['Not specified (or Not provided)', 'Average', 'Good', 'Very good', 'Excellent']
    frequencies = ['Once a week', 'Once every two weeks (or Bi-weekly)', 'More than once per week']

    data = {
        'mentee_id': range(1001, 1201),
        'field_of_study': np.random.choice(fields_of_study, 200),
        'study_level': np.random.choice(study_levels, 200),
        'degree': np.random.choice(degrees, 200),
        'needs': np.random.choice(needs, 200),
        'average_grade': np.random.choice(grades, 200),
        'program': np.random.choice(programs, 200),
        'engagement_score': np.random.uniform(0, 4, 200).round(1),
        'desired_exchange_frequency': np.random.choice(frequencies, 200)
    }

    return pd.DataFrame(data)


def create_sample_binome_data():
    """Create sample binome (pair) data for testing."""
    print("  Creating 150 sample binomes...")
    np.random.seed(42)

    statuses = ['COMPLETED', 'REJECTED', 'CANCELLED', 'ACTIVE']
    # Weighted to match real distribution (more failures)
    status_weights = [0.24, 0.46, 0.25, 0.05]

    data = {
        'binome_id': range(10001, 10151),
        'mentor_id': np.random.randint(1, 101, 150),
        'mentee_id': np.random.randint(1001, 1201, 150),
        'binome_statut': np.random.choice(statuses, 150, p=status_weights),
        'binome_score': np.random.uniform(0, 12, 150).round(1)
    }

    return pd.DataFrame(data)


def clean_data(mentor_df, mentee_df, binome_df):
    """Clean and deduplicate data."""
    print_header("CLEANING DATA")

    # Remove duplicates from binome data
    print(f"\nOriginal binome records: {len(binome_df):,}")
    print(f"Exact duplicates: {binome_df.duplicated().sum():,}")

    binome_clean = binome_df.drop_duplicates()
    print(f"After removing duplicates: {len(binome_clean):,}")

    # Deduplicate mentee data (keep first occurrence)
    if 'mentee_id' in mentee_df.columns:
        mentee_original = len(mentee_df)
        mentee_df_clean = mentee_df.drop_duplicates(subset=['mentee_id'], keep='first')
        print(f"\nMentee deduplication: {mentee_original:,} → {len(mentee_df_clean):,}")
    else:
        mentee_df_clean = mentee_df

    # Filter to shared IDs
    valid_mentor_ids = set(mentor_df['mentor_id']) if 'mentor_id' in mentor_df.columns else set()
    valid_mentee_ids = set(mentee_df_clean['mentee_id']) if 'mentee_id' in mentee_df_clean.columns else set()

    binome_shared = binome_clean[
        binome_clean['mentor_id'].isin(valid_mentor_ids) &
        binome_clean['mentee_id'].isin(valid_mentee_ids)
    ].copy()

    print(f"\nAfter filtering to valid IDs: {len(binome_shared):,}")

    return mentor_df, mentee_df_clean, binome_shared


def merge_datasets(mentor_df, mentee_df, binome_df):
    """Merge mentor, mentee, and binome datasets."""
    print_header("MERGING DATASETS")

    # Select binome columns
    binome_cols = ['binome_id', 'binome_statut', 'mentor_id', 'mentee_id']
    if 'binome_score' in binome_df.columns:
        binome_cols.append('binome_score')

    binome_selected = binome_df[binome_cols].copy()

    # Merge with mentor data
    print(f"\nMerging {len(binome_selected):,} binomes with mentors...")
    mentor_cols = ['mentor_id']
    if 'workfield' in mentor_df.columns:
        mentor_cols.append('workfield')

    final_df = binome_selected.merge(
        mentor_df[mentor_cols],
        on='mentor_id',
        how='inner'
    )
    print(f"✓ After mentor merge: {len(final_df):,} records")

    # Merge with mentee data
    print(f"\nMerging with mentees...")
    mentee_cols = ['mentee_id']
    for col in ['field_of_study', 'study_level', 'degree', 'needs',
                'average_grade', 'program', 'engagement_score',
                'desired_exchange_frequency']:
        if col in mentee_df.columns:
            mentee_cols.append(col)

    final_df = final_df.merge(
        mentee_df[mentee_cols],
        on='mentee_id',
        how='inner'
    )
    print(f"✓ After mentee merge: {len(final_df):,} records")

    print(f"\n✓ Final merged dataset:")
    print(f"  - Total records: {len(final_df):,}")
    print(f"  - Unique mentors: {final_df['mentor_id'].nunique():,}")
    print(f"  - Unique mentees: {final_df['mentee_id'].nunique():,}")

    return final_df


def create_target_variable(df):
    """Create binary target variable from binome status."""
    print_header("CREATING TARGET VARIABLE")

    print(f"\nOriginal status distribution:")
    print(df['binome_statut'].value_counts())

    # Filter to final outcomes only
    print(f"\nFiltering to final outcomes (COMPLETED, REJECTED, CANCELLED)...")
    df_ml = df[df['binome_statut'].isin(['COMPLETED', 'REJECTED', 'CANCELLED'])].copy()
    print(f"✓ Filtered to {len(df_ml):,} records")

    # Create binary target: 1 = COMPLETED (success), 0 = REJECTED/CANCELLED (failure)
    df_ml['target'] = (df_ml['binome_statut'] == 'COMPLETED').astype(int)

    # Display distribution
    success_count = (df_ml['target'] == 1).sum()
    failure_count = (df_ml['target'] == 0).sum()

    print(f"\nTarget distribution:")
    print(f"  Success (1): {success_count:,} ({success_count/len(df_ml)*100:.1f}%)")
    print(f"  Failure (0): {failure_count:,} ({failure_count/len(df_ml)*100:.1f}%)")

    if failure_count > success_count:
        imbalance_ratio = failure_count / success_count
        print(f"  Imbalance ratio: {imbalance_ratio:.2f}:1 (failure:success)")

    return df_ml


def save_dataset(df):
    """Save prepared dataset to CSV."""
    print_header("SAVING DATASET")

    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"\nMissing values found:")
        for col, count in missing[missing > 0].items():
            print(f"  {col}: {count:,} ({count/len(df)*100:.1f}%)")
    else:
        print(f"\n✓ No missing values")

    # Save to CSV
    df.to_csv(OUTPUT_FILE, index=False)

    file_size = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)  # MB

    print(f"\n✓ Saved dataset to: {OUTPUT_FILE}")
    print(f"  File size: {file_size:.2f} MB")
    print(f"  Total records: {len(df):,}")
    print(f"  Total columns: {len(df.columns)}")

    # Display column list
    print(f"\nColumns saved ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")

    return OUTPUT_FILE


def main():
    """Main data preparation pipeline."""

    print("\n" + "=" * 80)
    print("  MENTORSHIP DATA PREPARATION")
    print("  JP Morgan Data for Good Hackathon 2025 - Team 2")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 80)

    # 1. Load data
    mentor_df, mentee_df, binome_df = load_excel_data()

    # 2. Clean data
    mentor_df, mentee_df, binome_df = clean_data(mentor_df, mentee_df, binome_df)

    # 3. Merge datasets
    merged_df = merge_datasets(mentor_df, mentee_df, binome_df)

    # 4. Create target variable
    final_df = create_target_variable(merged_df)

    # 5. Save dataset
    output_file = save_dataset(final_df)

    # Final summary
    print_header("PREPARATION COMPLETE")

    print(f"\n✅ Data preparation completed successfully!")
    print(f"\n📊 Dataset Summary:")
    print(f"  - File: {output_file}")
    print(f"  - Records: {len(final_df):,}")
    print(f"  - Features: {len(final_df.columns) - 4}")  # Exclude IDs and target
    print(f"  - Target: binary (1=success, 0=failure)")

    print(f"\n🎯 Next Step:")
    print(f"  Run model training: python train_model.py")

    print("\n" + "=" * 80 + "\n")


if __name__ == '__main__':
    main()
