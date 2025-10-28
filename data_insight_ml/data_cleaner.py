"""
Data Cleaning and Preprocessing Toolkit for NGO Datasets
Handles common data quality issues and adds valuable features
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class DataCleaner:
    """
    Comprehensive data cleaning and preprocessing for NGO datasets
    """

    def __init__(self):
        self.cleaning_report = {
            'missing_values': {},
            'outliers_removed': {},
            'duplicates_removed': 0,
            'invalid_values': {},
            'features_added': [],
            'transformations': []
        }

    def clean_dataset(self, df, domain_type=None):
        """
        Main cleaning pipeline

        Args:
            df: Input DataFrame
            domain_type: Type of domain ('donor', 'program', 'student', etc.)

        Returns:
            Cleaned DataFrame and cleaning report
        """
        print("\n" + "=" * 60)
        print("DATA CLEANING PIPELINE")
        print("=" * 60)

        original_shape = df.shape
        print(f"\n[INPUT] Original data: {original_shape[0]} rows, {original_shape[1]} columns")

        # Step 1: Remove duplicates
        df = self._remove_duplicates(df)

        # Step 2: Handle missing values
        df = self._handle_missing_values(df, domain_type)

        # Step 3: Fix data types
        df = self._fix_data_types(df)

        # Step 4: Standardize categorical values
        df = self._standardize_categories(df)

        # Step 5: Handle outliers
        df = self._handle_outliers(df)

        # Step 6: Add derived features
        df = self._add_derived_features(df, domain_type)

        # Step 7: Validate data ranges
        df = self._validate_ranges(df)

        final_shape = df.shape
        print(f"\n[OUTPUT] Cleaned data: {final_shape[0]} rows, {final_shape[1]} columns")
        print(f"[INFO] Rows removed: {original_shape[0] - final_shape[0]}")
        print(f"[INFO] Columns added: {final_shape[1] - original_shape[1]}")

        print("\n" + "=" * 60)
        print("CLEANING COMPLETE")
        print("=" * 60)

        return df, self.cleaning_report

    def _remove_duplicates(self, df):
        """Remove duplicate rows"""
        print("\n[STEP 1] Removing duplicates...")

        initial_rows = len(df)
        df = df.drop_duplicates()
        duplicates_removed = initial_rows - len(df)

        self.cleaning_report['duplicates_removed'] = duplicates_removed

        if duplicates_removed > 0:
            print(f"  [OK] Removed {duplicates_removed} duplicate rows")
        else:
            print(f"  [OK] No duplicates found")

        return df

    def _handle_missing_values(self, df, domain_type):
        """Handle missing values intelligently"""
        print("\n[STEP 2] Handling missing values...")

        missing_before = df.isnull().sum().sum()

        if missing_before == 0:
            print("  [OK] No missing values found")
            return df

        print(f"  [INFO] Found {missing_before} missing values")

        for col in df.columns:
            missing_count = df[col].isnull().sum()

            if missing_count == 0:
                continue

            missing_pct = (missing_count / len(df)) * 100
            self.cleaning_report['missing_values'][col] = {
                'count': int(missing_count),
                'percentage': round(missing_pct, 2)
            }

            # If more than 50% missing, consider dropping the column
            if missing_pct > 50:
                print(f"  [WARNING] Column '{col}' has {missing_pct:.1f}% missing - consider removing")
                continue

            # Handle based on data type
            if df[col].dtype in ['float64', 'int64']:
                # Numeric: use median
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"  [OK] Filled {missing_count} missing values in '{col}' with median: {median_val:.2f}")

            elif df[col].dtype == 'object':
                # Categorical: use mode or 'Unknown'
                if df[col].mode().shape[0] > 0:
                    mode_val = df[col].mode()[0]
                    df[col].fillna(mode_val, inplace=True)
                    print(f"  [OK] Filled {missing_count} missing values in '{col}' with mode: '{mode_val}'")
                else:
                    df[col].fillna('Unknown', inplace=True)
                    print(f"  [OK] Filled {missing_count} missing values in '{col}' with 'Unknown'")

        missing_after = df.isnull().sum().sum()
        print(f"  [SUMMARY] Missing values: {missing_before} -> {missing_after}")

        return df

    def _fix_data_types(self, df):
        """Fix and standardize data types"""
        print("\n[STEP 3] Fixing data types...")

        type_fixes = 0

        for col in df.columns:
            # Try to convert strings that look like numbers
            if df[col].dtype == 'object':
                try:
                    # Remove common formatting characters
                    if df[col].str.contains(r'[$,]', na=False).any():
                        df[col] = df[col].str.replace('$', '').str.replace(',', '')
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        type_fixes += 1
                        print(f"  [OK] Converted '{col}' from string to numeric (removed $ and ,)")
                except:
                    pass

        if type_fixes == 0:
            print("  [OK] All data types are correct")
        else:
            print(f"  [OK] Fixed {type_fixes} column data types")

        return df

    def _standardize_categories(self, df):
        """Standardize categorical values"""
        print("\n[STEP 4] Standardizing categorical values...")

        standardizations = 0

        for col in df.columns:
            if df[col].dtype == 'object':
                # Remove leading/trailing spaces
                df[col] = df[col].str.strip()

                # Standardize common variations
                replacements = {
                    'yes': 'Yes',
                    'YES': 'Yes',
                    'no': 'No',
                    'NO': 'No',
                    'male': 'Male',
                    'MALE': 'Male',
                    'female': 'Female',
                    'FEMALE': 'Female',
                    'low': 'Low',
                    'LOW': 'Low',
                    'medium': 'Medium',
                    'MEDIUM': 'Medium',
                    'high': 'High',
                    'HIGH': 'High'
                }

                for old_val, new_val in replacements.items():
                    if (df[col] == old_val).any():
                        df[col] = df[col].replace(old_val, new_val)
                        standardizations += 1

        if standardizations > 0:
            print(f"  [OK] Standardized {standardizations} categorical values")
        else:
            print("  [OK] Categories already standardized")

        return df

    def _handle_outliers(self, df):
        """Detect and handle outliers"""
        print("\n[STEP 5] Handling outliers...")

        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        outliers_found = 0

        for col in numeric_cols:
            # Skip ID columns
            if 'id' in col.lower():
                continue

            # Use IQR method
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR

            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_count = outlier_mask.sum()

            if outlier_count > 0:
                outliers_found += outlier_count
                self.cleaning_report['outliers_removed'][col] = int(outlier_count)

                # Cap outliers instead of removing
                df.loc[df[col] < lower_bound, col] = lower_bound
                df.loc[df[col] > upper_bound, col] = upper_bound

                print(f"  [OK] Capped {outlier_count} outliers in '{col}'")

        if outliers_found == 0:
            print("  [OK] No extreme outliers detected")
        else:
            print(f"  [SUMMARY] Handled {outliers_found} total outliers")

        return df

    def _add_derived_features(self, df, domain_type):
        """Add valuable derived features"""
        print("\n[STEP 6] Adding derived features...")

        features_added = []

        # Add features based on existing columns

        # 1. Age groups
        if 'age' in df.columns:
            df['age_group'] = pd.cut(
                df['age'],
                bins=[0, 12, 17, 25, 40, 60, 100],
                labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Middle Age', 'Senior']
            )
            features_added.append('age_group')

        # 2. Engagement score (if attendance/participation metrics exist)
        engagement_cols = [col for col in df.columns if 'attendance' in col.lower() or 'participation' in col.lower()]
        if engagement_cols:
            df['engagement_level'] = df[engagement_cols].mean(axis=1)
            df['engagement_category'] = pd.cut(
                df['engagement_level'],
                bins=[0, 0.3, 0.6, 1.0],
                labels=['Low', 'Medium', 'High']
            )
            features_added.extend(['engagement_level', 'engagement_category'])

        # 3. Risk indicators for educational domains
        if domain_type in ['student', 'education', 'student_dropout']:
            if 'gpa' in df.columns and 'attendance_rate' in df.columns:
                df['academic_risk_score'] = (
                    (4 - df['gpa']) / 4 * 0.5 +  # GPA risk (inverted)
                    (1 - df['attendance_rate']) * 0.5  # Attendance risk
                )
                df['academic_risk_category'] = pd.cut(
                    df['academic_risk_score'],
                    bins=[0, 0.3, 0.6, 1.0],
                    labels=['Low Risk', 'Medium Risk', 'High Risk']
                )
                features_added.extend(['academic_risk_score', 'academic_risk_category'])

        # 4. Financial indicators for donor domains
        if domain_type in ['donor', 'fundraising', 'donor_retention']:
            if 'last_donation_amount' in df.columns:
                df['donation_size_category'] = pd.cut(
                    df['last_donation_amount'],
                    bins=[0, 50, 200, 1000, float('inf')],
                    labels=['Small', 'Medium', 'Large', 'Major']
                )
                features_added.append('donation_size_category')

            if 'donation_frequency' in df.columns:
                df['donor_type'] = pd.cut(
                    df['donation_frequency'],
                    bins=[0, 1, 3, 10, float('inf')],
                    labels=['One-time', 'Occasional', 'Regular', 'Major Donor']
                )
                features_added.append('donor_type')

        # 5. Health/wellbeing scores
        if domain_type in ['child', 'health', 'child_wellbeing']:
            health_cols = [col for col in df.columns if any(term in col.lower() for term in ['nutrition', 'health', 'checkup'])]
            if health_cols:
                df['health_score'] = df[health_cols].mean(axis=1)
                features_added.append('health_score')

        self.cleaning_report['features_added'] = features_added

        if features_added:
            print(f"  [OK] Added {len(features_added)} derived features:")
            for feature in features_added:
                print(f"      - {feature}")
        else:
            print("  [INFO] No derived features added")

        return df

    def _validate_ranges(self, df):
        """Validate and fix data ranges"""
        print("\n[STEP 7] Validating data ranges...")

        validations = 0

        # Common range validations
        validations_to_check = {
            'age': (0, 120),
            'gpa': (0, 4.0),
            'attendance_rate': (0, 1.0),
            'confidence': (0, 1.0),
            'percentage': (0, 100)
        }

        for col in df.columns:
            # Skip non-numeric columns
            if df[col].dtype not in ['float64', 'int64']:
                continue

            for key, (min_val, max_val) in validations_to_check.items():
                if key in col.lower():
                    # Check and fix out of range values
                    out_of_range = ((df[col] < min_val) | (df[col] > max_val)).sum()

                    if out_of_range > 0:
                        df[col] = df[col].clip(min_val, max_val)
                        validations += out_of_range
                        print(f"  [OK] Fixed {out_of_range} out-of-range values in '{col}'")

        if validations == 0:
            print("  [OK] All values within expected ranges")

        return df

    def generate_quality_report(self, df):
        """Generate comprehensive data quality report"""
        print("\n" + "=" * 60)
        print("DATA QUALITY REPORT")
        print("=" * 60)

        # Basic statistics
        print(f"\n[DATASET INFO]")
        print(f"  Total rows: {len(df)}")
        print(f"  Total columns: {len(df.columns)}")
        print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

        # Missing values
        print(f"\n[MISSING VALUES]")
        missing = df.isnull().sum()
        if missing.sum() == 0:
            print("  None - dataset is complete!")
        else:
            for col, count in missing[missing > 0].items():
                pct = (count / len(df)) * 100
                print(f"  {col}: {count} ({pct:.1f}%)")

        # Data types
        print(f"\n[DATA TYPES]")
        type_counts = df.dtypes.value_counts()
        for dtype, count in type_counts.items():
            print(f"  {dtype}: {count} columns")

        # Numeric columns statistics
        print(f"\n[NUMERIC STATISTICS]")
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) > 0:
            print(f"  Numeric columns: {len(numeric_cols)}")
            print(f"\n{df[numeric_cols].describe().round(2)}")

        # Categorical columns
        print(f"\n[CATEGORICAL COLUMNS]")
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            print(f"  Categorical columns: {len(categorical_cols)}")
            for col in categorical_cols[:5]:  # Show first 5
                unique_count = df[col].nunique()
                print(f"    {col}: {unique_count} unique values")

        print("\n" + "=" * 60)

        return self.cleaning_report


def clean_csv_file(input_file, output_file=None, domain_type=None):
    """
    Clean a CSV file and save the result

    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file (optional)
        domain_type: Domain type for specialized cleaning

    Returns:
        Cleaned DataFrame
    """
    print(f"\n[INFO] Loading data from: {input_file}")

    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"[ERROR] Failed to load CSV: {str(e)}")
        return None

    # Clean the data
    cleaner = DataCleaner()
    df_clean, report = cleaner.clean_dataset(df, domain_type)

    # Generate quality report
    cleaner.generate_quality_report(df_clean)

    # Save cleaned data
    if output_file:
        df_clean.to_csv(output_file, index=False)
        print(f"\n[SUCCESS] Cleaned data saved to: {output_file}")
    else:
        # Auto-generate output filename
        output_file = input_file.replace('.csv', '_cleaned.csv')
        df_clean.to_csv(output_file, index=False)
        print(f"\n[SUCCESS] Cleaned data saved to: {output_file}")

    return df_clean


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("\nUsage: python data_cleaner.py <input_file.csv> [domain_type]")
        print("\nDomain types:")
        print("  - donor (donor retention)")
        print("  - student (student dropout)")
        print("  - child (child wellbeing)")
        print("  - program (program completion)")
        print("  - grant (grant scoring)")
        print("  - customer (customer churn)")
        print("\nExample:")
        print("  python data_cleaner.py raw_student_data.csv student")
        sys.exit(1)

    input_file = sys.argv[1]
    domain_type = sys.argv[2] if len(sys.argv) > 2 else None

    clean_csv_file(input_file, domain_type=domain_type)
