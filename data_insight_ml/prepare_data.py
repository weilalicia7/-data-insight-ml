"""
Data Preparation Script with Auto-Detection
Automatically detects data types and prepares data for ML training
"""

import pandas as pd
import numpy as np
import yaml
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def load_config():
    """Load configuration from config.yaml"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def print_header(title):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def load_data(filepath):
    """Load CSV data and perform initial inspection"""
    print_header("LOADING DATA")

    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        sys.exit(1)

    print(f"Loading: {filepath}")

    # Try different encodings
    for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
        try:
            df = pd.read_csv(filepath, encoding=encoding)
            print(f"Successfully loaded with encoding: {encoding}")
            break
        except UnicodeDecodeError:
            continue
    else:
        print("ERROR: Could not read file with any encoding")
        sys.exit(1)

    print(f"\nDataset shape: {df.shape}")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")

    # Check minimum data requirements
    if len(df) < 50:
        print("\n WARNING: Dataset has fewer than 50 rows.")
        print("  Recommendation: Add more data for better model performance.")

    return df


def auto_detect_target(df, config):
    """Auto-detect or verify target column"""
    print_header("TARGET COLUMN DETECTION")

    target_col = config['data']['target_column']

    # Check if specified target exists
    if target_col in df.columns:
        print(f"Target column found: '{target_col}'")
        target_values = df[target_col].unique()
        print(f"Unique values: {len(target_values)}")

        if len(target_values) <= 10:
            print(f"Values: {target_values[:10]}")

        # Determine if classification or regression
        if len(target_values) <= 20:
            print("\nTask type: CLASSIFICATION")
            value_counts = df[target_col].value_counts()
            print("\nDistribution:")
            for val, count in value_counts.items():
                print(f"  {val}: {count:,} ({count/len(df)*100:.1f}%)")
        else:
            print("\nTask type: REGRESSION")
            print(f"Range: [{df[target_col].min()}, {df[target_col].max()}]")
            print(f"Mean: {df[target_col].mean():.2f}")

        return target_col

    # Try to find target column
    print(f"Target column '{target_col}' not found.")
    print("\nSearching for common target column names...")

    common_targets = [
        'target', 'label', 'class', 'outcome', 'result', 'success',
        'failure', 'approved', 'status', 'category', 'y', 'prediction'
    ]

    for col in df.columns:
        if col.lower() in common_targets:
            print(f"\nFound potential target: '{col}'")
            response = input(f"Use '{col}' as target? (y/n): ")
            if response.lower() == 'y':
                return col

    # Let user choose
    print("\nAvailable columns:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")

    while True:
        try:
            choice = int(input("\nEnter column number for target: "))
            if 1 <= choice <= len(df.columns):
                return df.columns[choice - 1]
        except ValueError:
            pass
        print("Invalid choice. Try again.")


def detect_column_types(df, target_col, config):
    """Automatically detect column types"""
    print_header("FEATURE TYPE DETECTION")

    id_cols = config['data']['id_columns']
    max_categories = config['features']['max_categories']

    numeric_cols = []
    categorical_cols = []
    date_cols = []
    id_columns = []

    for col in df.columns:
        if col == target_col:
            continue

        # Check if ID column
        if col.lower() in [x.lower() for x in id_cols]:
            id_columns.append(col)
            continue

        # Check if date column
        if df[col].dtype == 'object':
            try:
                pd.to_datetime(df[col].head(100))
                date_cols.append(col)
                continue
            except:
                pass

        # Numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):
            unique_count = df[col].nunique()

            # If few unique values, might be categorical
            if unique_count <= max_categories and unique_count < len(df) * 0.05:
                categorical_cols.append(col)
            else:
                numeric_cols.append(col)

        # Categorical columns
        elif df[col].dtype == 'object' or df[col].dtype.name == 'category':
            unique_count = df[col].nunique()

            if unique_count > max_categories:
                print(f"\n  WARNING: '{col}' has {unique_count} unique values (high cardinality)")
                print(f"  Recommendation: Review if this should be included")
                response = input(f"  Include '{col}'? (y/n): ")
                if response.lower() == 'y':
                    categorical_cols.append(col)
                else:
                    id_columns.append(col)
            else:
                categorical_cols.append(col)

    # Print summary
    print(f"\nNumeric features: {len(numeric_cols)}")
    for col in numeric_cols[:5]:
        print(f"  - {col}")
    if len(numeric_cols) > 5:
        print(f"  ... and {len(numeric_cols) - 5} more")

    print(f"\nCategorical features: {len(categorical_cols)}")
    for col in categorical_cols[:5]:
        unique_count = df[col].nunique()
        print(f"  - {col} ({unique_count} categories)")
    if len(categorical_cols) > 5:
        print(f"  ... and {len(categorical_cols) - 5} more")

    if date_cols:
        print(f"\nDate features: {len(date_cols)}")
        for col in date_cols:
            print(f"  - {col}")

    if id_columns:
        print(f"\nID columns (will be excluded): {len(id_columns)}")
        for col in id_columns[:3]:
            print(f"  - {col}")

    return {
        'numeric': numeric_cols,
        'categorical': categorical_cols,
        'date': date_cols,
        'id': id_columns
    }


def handle_missing_values(df, column_types, config):
    """Handle missing values based on configuration"""
    print_header("HANDLING MISSING VALUES")

    strategy = config['features']['handle_missing']

    total_missing = df.isnull().sum().sum()

    if total_missing == 0:
        print("No missing values found!")
        return df

    print(f"Total missing values: {total_missing:,}")

    # Show columns with missing values
    missing_cols = df.columns[df.isnull().any()].tolist()
    print(f"\nColumns with missing values: {len(missing_cols)}")
    for col in missing_cols[:10]:
        missing_count = df[col].isnull().sum()
        missing_pct = (missing_count / len(df)) * 100
        print(f"  - {col}: {missing_count:,} ({missing_pct:.1f}%)")

    df_clean = df.copy()

    if strategy == "auto":
        # Numeric: fill with median
        for col in column_types['numeric']:
            if col in missing_cols:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)

        # Categorical: fill with mode or 'Unknown'
        for col in column_types['categorical']:
            if col in missing_cols:
                mode_val = df_clean[col].mode()
                if len(mode_val) > 0:
                    df_clean[col].fillna(mode_val[0], inplace=True)
                else:
                    df_clean[col].fillna('Unknown', inplace=True)

    elif strategy == "drop":
        df_clean = df_clean.dropna()
        print(f"\nRows after dropping missing: {len(df_clean):,}")

    print(f"\n Remaining missing values: {df_clean.isnull().sum().sum()}")

    return df_clean


def engineer_features(df, column_types, config):
    """Engineer features from raw data"""
    print_header("FEATURE ENGINEERING")

    df_eng = df.copy()

    # Date features
    if column_types['date'] and config['features']['extract_date_features']:
        print("\nExtracting date features...")
        for col in column_types['date']:
            df_eng[col] = pd.to_datetime(df_eng[col])

            if config['features']['extract_date_features']['day_of_week']:
                df_eng[f'{col}_day_of_week'] = df_eng[col].dt.dayofweek
                column_types['numeric'].append(f'{col}_day_of_week')

            if config['features']['extract_date_features']['month']:
                df_eng[f'{col}_month'] = df_eng[col].dt.month
                column_types['numeric'].append(f'{col}_month')

            if config['features']['extract_date_features']['is_weekend']:
                df_eng[f'{col}_is_weekend'] = (df_eng[col].dt.dayofweek >= 5).astype(int)
                column_types['numeric'].append(f'{col}_is_weekend')

            print(f"  - Extracted features from '{col}'")

    # Interaction features (example)
    if len(column_types['numeric']) >= 2:
        print("\nCreating interaction features...")
        # Example: multiply first two numeric features
        col1, col2 = column_types['numeric'][:2]
        df_eng[f'{col1}_x_{col2}'] = df_eng[col1] * df_eng[col2]
        column_types['numeric'].append(f'{col1}_x_{col2}')
        print(f"  - Created: {col1}_x_{col2}")

    print(f"\nTotal features after engineering: {len(column_types['numeric']) + len(column_types['categorical'])}")

    return df_eng, column_types


def prepare_for_training(df, target_col, column_types, config):
    """Prepare final dataset for training"""
    print_header("PREPARING FOR TRAINING")

    # Remove ID columns
    df_model = df.drop(columns=column_types['id'], errors='ignore')

    # Remove original date columns (keep engineered features)
    df_model = df_model.drop(columns=column_types['date'], errors='ignore')

    # One-hot encode categorical variables
    if config['features']['encode_categorical'] and column_types['categorical']:
        print("\nOne-hot encoding categorical features...")
        df_model = pd.get_dummies(df_model, columns=column_types['categorical'], drop_first=True)
        print(f"  Encoded {len(column_types['categorical'])} categorical features")

    # Separate features and target
    X = df_model.drop(columns=[target_col])
    y = df_model[target_col]

    print(f"\nFinal dataset shape:")
    print(f"  Features (X): {X.shape}")
    print(f"  Target (y): {y.shape}")
    print(f"  Total features: {X.shape[1]}")

    # Save prepared data
    output_file = 'ml_ready_dataset.csv'
    df_model.to_csv(output_file, index=False)
    print(f"\n Saved prepared data: {output_file}")

    # Save metadata
    metadata = {
        'target_column': target_col,
        'numeric_features': column_types['numeric'],
        'categorical_features': column_types['categorical'],
        'total_features': X.shape[1],
        'n_samples': len(df_model),
        'preparation_date': datetime.now().isoformat()
    }

    with open('data_metadata.yaml', 'w') as f:
        yaml.dump(metadata, f)
    print(f" Saved metadata: data_metadata.yaml")

    return X, y


def main():
    """Main data preparation pipeline"""

    print("\n" + "=" * 80)
    print("  DATA INSIGHT ML - DATA PREPARATION")
    print("  Auto-Detection & Feature Engineering")
    print("=" * 80)

    # Load configuration
    config = load_config()

    # Get data file from command line or prompt
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    else:
        data_file = input("\nEnter path to your CSV file: ")

    # 1. Load data
    df = load_data(data_file)

    # 2. Detect target column
    target_col = auto_detect_target(df, config)

    # 3. Detect column types
    column_types = detect_column_types(df, target_col, config)

    # 4. Handle missing values
    df_clean = handle_missing_values(df, column_types, config)

    # 5. Engineer features
    df_eng, column_types = engineer_features(df_clean, column_types, config)

    # 6. Prepare for training
    X, y = prepare_for_training(df_eng, target_col, column_types, config)

    # Final summary
    print_header("PREPARATION COMPLETE")
    print("\n Data is ready for model training!")
    print(f"  - Samples: {len(X):,}")
    print(f"  - Features: {X.shape[1]}")
    print(f"  - Target: '{target_col}'")

    print("\n Next step:")
    print("  python train_model.py")

    print("\n" + "=" * 80 + "\n")


if __name__ == '__main__':
    main()
