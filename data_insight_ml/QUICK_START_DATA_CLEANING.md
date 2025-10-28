# Data Cleaning - Quick Start Guide

## One-Line Commands

### Clean Your Data
```bash
python data_cleaner.py your_data.csv <domain>
```

### Domain Types
- `student` - Education/student data
- `donor` - Fundraising/donor data
- `child` - Child wellbeing data
- `program` - Program completion data
- `grant` - Grant applications
- `customer` - Member/customer data

### Example
```bash
python data_cleaner.py student_records.csv student
```
Output: `student_records_cleaned.csv`

---

## What Gets Fixed

| Problem | Solution | Example |
|---------|----------|---------|
| Missing values | Filled with median/mode | `age: NaN` → `age: 16.0` |
| Inconsistent text | Standardized | `low` → `Low` |
| Invalid values | Fixed to valid range | `gpa: 5.0` → `gpa: 4.0` |
| Outliers | Capped | `absences: 50` → `absences: 15` |
| Duplicates | Removed | 100 rows → 95 rows |
| Missing features | Added | +5 new columns |

---

## Try the Demo

```bash
# 1. Create messy test data
python create_messy_demo_data.py

# 2. Clean it
python data_cleaner.py messy_student_data.csv student

# 3. Compare
# - messy_student_data.csv (before)
# - messy_student_data_cleaned.csv (after)
```

---

## Full Workflow

```bash
# Step 1: Clean your raw data
python data_cleaner.py raw_data.csv student

# Step 2: Prepare for ML training
python prepare_data.py raw_data_cleaned.csv

# Step 3: Train model
python train_model.py

# Step 4: Start API
python app.py
```

---

## What You Get

### Before (Messy Data)
- 53 rows, 13 columns
- 93 missing values
- Inconsistent formatting
- Invalid ranges
- 3 duplicates

### After (Clean Data)
- 50 rows, 18 columns
- 0 critical missing values
- Standardized formats
- Valid ranges
- 0 duplicates
- +5 useful features

---

## New Features Added

### Student Data
- `age_group` - Child, Teen, Adult, etc.
- `academic_risk_score` - 0-1 risk indicator
- `academic_risk_category` - Low/Medium/High Risk
- `engagement_level` - Overall engagement
- `engagement_category` - Low/Medium/High

### Donor Data
- `donation_size_category` - Small/Medium/Large/Major
- `donor_type` - One-time/Occasional/Regular/Major

---

## Quick Tips

✓ **Keep original data** - Cleaned version is saved separately
✓ **Review the report** - Shows exactly what was changed
✓ **Check derived features** - Make sure they make sense
✓ **Use for training** - Cleaned CSV is ready for ML

---

## Common Issues

### Too Many Missing Values
```
[WARNING] Column 'x' has 85% missing
```
→ Consider removing this column or collecting better data

### Unexpected Categories
```
economic_status: 9 unique values (expected 3)
```
→ Check for typos and standardize manually first

---

## Output Files

```
your_data.csv              # Original (safe)
your_data_cleaned.csv      # Cleaned (use this!)
```

---

## Need Help?

1. Read full guide: `DATA_CLEANING_GUIDE.md`
2. Check the cleaning report output
3. Compare before/after CSV files

**Remember:** Good data = Good predictions!
