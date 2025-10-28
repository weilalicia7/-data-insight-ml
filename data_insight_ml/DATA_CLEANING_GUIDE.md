# NGO Data Cleaning and Preprocessing Guide

## Overview

NGO datasets often have data quality issues that can negatively impact machine learning model performance. This toolkit provides automated data cleaning and preprocessing specifically designed for NGO use cases.

## Common NGO Data Problems

### 1. **Missing Values**
- Incomplete surveys/forms
- Optional fields left blank
- Lost or unavailable data
- Example: `age: NaN`, `parent_involvement: None`

### 2. **Inconsistent Formatting**
- Mixed case in categories (`Low`, `low`, `LOW`)
- Inconsistent spacing (`"John Smith"`, `"  John Smith  "`)
- Different representations of same value
- Example: `economic_status: ['Low Income', 'low income', 'LOW INCOME']`

### 3. **Invalid Values**
- Data entry errors
- Values outside valid ranges
- Example: `gpa: 5.0` (should be 0-4.0), `attendance_rate: 1.5` (should be 0-1.0)

### 4. **Outliers**
- Extreme values that skew analysis
- Data entry mistakes
- Example: `absences_per_month: 20` (when normal is 0-5)

### 5. **Duplicate Records**
- Same person entered multiple times
- Accidental re-submissions
- Example: Same student ID appearing twice

## What the Data Cleaner Does

### Automated Cleaning Steps

1. **Remove Duplicates** - Eliminates exact duplicate rows
2. **Handle Missing Values** - Fills gaps intelligently:
   - Numeric fields → Median value
   - Categorical fields → Most common value (mode)
3. **Fix Data Types** - Converts strings to numbers where appropriate
4. **Standardize Categories** - Makes formatting consistent
5. **Handle Outliers** - Caps extreme values (doesn't delete rows)
6. **Add Derived Features** - Creates useful new columns:
   - Age groups (Child, Teen, Adult, etc.)
   - Risk scores
   - Engagement levels
7. **Validate Ranges** - Ensures values are within valid bounds

## How to Use

### Basic Usage

```bash
# Clean a CSV file
python data_cleaner.py <your_file.csv> <domain_type>
```

### Domain Types

Specify the domain for specialized cleaning:

- `donor` - Donor retention datasets
- `student` - Student/education datasets
- `child` - Child wellbeing datasets
- `program` - Program completion datasets
- `grant` - Grant application datasets
- `customer` - Customer/member churn datasets

### Examples

#### Example 1: Clean Student Data
```bash
python data_cleaner.py raw_student_data.csv student
```

Output: `raw_student_data_cleaned.csv`

#### Example 2: Clean Donor Data
```bash
python data_cleaner.py donor_list.csv donor
```

Output: `donor_list_cleaned.csv`

#### Example 3: No Domain Specified
```bash
python data_cleaner.py generic_data.csv
```

Still works! But won't add domain-specific features.

## What Gets Fixed - Real Example

### Before Cleaning (53 rows, 13 columns):

```
student_id    student_name  age  grade_level  attendance_rate  gpa  parent_involvement
STU1001      John Smith     15.0      9           0.85         3.5      Low
STU1002      mary johnson   16.0     10           0.92         3.8      High
STU1003      Sarah Davis     NaN     11            NaN         2.9      medium
STU1004      MIKE WILSON    17.0     12           0.78         NaN      LOW
STU1005      Emma Brown     14.0      9           1.50         3.2      Medium
```

**Problems:**
- Missing age (STU1003)
- Missing attendance (STU1003)
- Missing GPA (STU1004)
- Invalid attendance 1.50 (STU1005, should be ≤ 1.0)
- Inconsistent formatting (low/Low/LOW, mixed case names)

### After Cleaning (50 rows, 18 columns):

```
student_id  student_name  age  grade_level  attendance_rate  gpa  parent_involvement  age_group  academic_risk_score
STU1001     John Smith    15.0      9           0.85         3.5      Low              Teen           0.138
STU1002     mary johnson  16.0     10           0.92         3.8      High             Teen           0.065
STU1003     Sarah Davis   16.0     11           0.87         2.9      Medium           Teen           0.200
STU1004     MIKE WILSON   17.0     12           0.78         3.45     Low              Teen           0.179
STU1005     Emma Brown    14.0      9           1.00         3.2      Medium           Teen           0.075
```

**Improvements:**
- ✓ Missing values filled (age: 16.0, attendance: 0.87, GPA: 3.45)
- ✓ Invalid attendance capped to 1.0
- ✓ Categories standardized (Low/Medium/High)
- ✓ 3 duplicate rows removed (53 → 50 rows)
- ✓ 5 new useful columns added (age_group, academic_risk_score, etc.)

## Cleaning Report

After cleaning, you get a detailed report:

```
============================================================
DATA CLEANING PIPELINE
============================================================

[INPUT] Original data: 53 rows, 13 columns

[STEP 1] Removing duplicates...
  [OK] Removed 3 duplicate rows

[STEP 2] Handling missing values...
  [INFO] Found 93 missing values
  [OK] Filled 6 missing values in 'age' with median: 16.00
  [OK] Filled 8 missing values in 'attendance_rate' with median: 0.88
  [OK] Filled 8 missing values in 'gpa' with median: 3.45

[STEP 3] Fixing data types...
  [OK] All data types are correct

[STEP 4] Standardizing categorical values...
  [OK] Standardized 6 categorical values

[STEP 5] Handling outliers...
  [OK] Capped 2 outliers in 'attendance_rate'
  [OK] Capped 1 outliers in 'gpa'
  [OK] Capped 1 outliers in 'absences_per_month'

[STEP 6] Adding derived features...
  [OK] Added 5 derived features:
      - age_group
      - engagement_level
      - engagement_category
      - academic_risk_score
      - academic_risk_category

[STEP 7] Validating data ranges...
  [OK] Fixed 1 out-of-range values in 'attendance_rate'

[OUTPUT] Cleaned data: 50 rows, 18 columns
[INFO] Rows removed: 3
[INFO] Columns added: 5

============================================================
CLEANING COMPLETE
============================================================
```

## Derived Features Added

### For Student/Education Domains

1. **age_group** - Categorical age grouping
   - Child (0-12), Teen (13-17), Young Adult (18-25), etc.

2. **academic_risk_score** - Numeric risk indicator (0-1)
   - Combines GPA and attendance into single risk measure
   - Higher = more at risk

3. **academic_risk_category** - Risk level
   - Low Risk, Medium Risk, High Risk

### For Donor Domains

1. **donation_size_category**
   - Small (<$50), Medium ($50-$200), Large ($200-$1000), Major (>$1000)

2. **donor_type**
   - One-time, Occasional, Regular, Major Donor

### For All Domains

1. **engagement_level** - Overall engagement score
2. **engagement_category** - Low, Medium, High

## Best Practices

### 1. **Always Keep Original Data**
The cleaner creates a new file (`_cleaned.csv`), so your original data is safe.

### 2. **Review the Cleaning Report**
Check what was changed to ensure it makes sense for your data.

### 3. **Validate Domain-Specific Logic**
Review derived features to ensure they align with your organization's definitions.

### 4. **Handle Privacy Concerns**
Remove personally identifiable information (PII) before sharing cleaned data.

### 5. **Use Cleaned Data for ML Training**
The cleaned CSV can be directly used with:
```bash
python prepare_data.py cleaned_data.csv
python train_model.py
```

## Advanced: Using in Python Code

```python
from data_cleaner import DataCleaner

# Create cleaner
cleaner = DataCleaner()

# Load your data
import pandas as pd
df = pd.read_csv('your_data.csv')

# Clean it
df_clean, report = cleaner.clean_dataset(df, domain_type='student')

# Generate quality report
cleaner.generate_quality_report(df_clean)

# Save
df_clean.to_csv('cleaned_data.csv', index=False)
```

## Troubleshooting

### Issue: Too Many Missing Values

```
[WARNING] Column 'optional_field' has 85.0% missing - consider removing
```

**Solution:** If a column has >50% missing, consider:
1. Removing the column entirely
2. Collecting this data more consistently going forward
3. Using a different imputation strategy

### Issue: Outliers Being Capped

```
[OK] Capped 10 outliers in 'donation_amount'
```

**Review:** Check if these are:
- Real extreme values (keep them) - edit the code to adjust thresholds
- Data entry errors (good to cap)

### Issue: Categories Not Standardizing

```
economic_status: 9 unique values  # Should be 3
```

**Solution:** Manually check and standardize before cleaning:
```python
df['economic_status'] = df['economic_status'].replace({
    'low income': 'Low Income',
    'LOW INCOME': 'Low Income',
    # ... other mappings
})
```

## File Outputs

After running the cleaner:

```
your_data.csv                  # Original (unchanged)
your_data_cleaned.csv          # Cleaned version
```

## Integration with Training Pipeline

1. **Clean your data:**
   ```bash
   python data_cleaner.py raw_data.csv student
   ```

2. **Prepare for training:**
   ```bash
   python prepare_data.py raw_data_cleaned.csv
   ```

3. **Train model:**
   ```bash
   python train_model.py
   ```

## Data Quality Metrics

The cleaner provides these quality metrics:

- **Completeness:** % of non-missing values
- **Validity:** % of values within expected ranges
- **Consistency:** Standardization of formats
- **Uniqueness:** Duplicate detection and removal
- **Accuracy:** Outlier detection and handling

## Demo Dataset

Try the cleaner with our demo messy data:

```bash
# Generate messy demo data
python create_messy_demo_data.py

# Clean it
python data_cleaner.py messy_student_data.csv student

# Compare before and after
# - messy_student_data.csv (original)
# - messy_student_data_cleaned.csv (cleaned)
```

## Summary

The Data Cleaning Toolkit:

✓ Handles missing values intelligently
✓ Standardizes inconsistent formatting
✓ Fixes invalid and out-of-range values
✓ Removes duplicates
✓ Adds valuable derived features
✓ Provides detailed cleaning reports
✓ Works with all NGO domains
✓ Preserves original data
✓ Ready for ML training

**Result:** Clean, consistent, analysis-ready data that improves model performance!

---

## Support

For issues or questions:
1. Check the cleaning report for details
2. Review this guide's troubleshooting section
3. Examine the before/after CSV files
4. Adjust thresholds in `data_cleaner.py` if needed

**Remember:** Data cleaning is the foundation of good machine learning. Clean data = Better predictions!
