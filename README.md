# Data Insight ML - Multi-Domain NGO Analytics Platform

**JP Morgan Data for Good Hackathon 2025 - Team 2**

A comprehensive machine learning platform designed for NGOs and nonprofits, featuring multi-domain predictions, interactive data cleaning with user approval, and real-time visualizations.

![Status](https://img.shields.io/badge/Status-Production_Ready-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Flask](https://img.shields.io/badge/Flask-3.0.0-green)
![Chart.js](https://img.shields.io/badge/Chart.js-4.4.0-orange)

🔗 **Live Repository**: https://github.com/weilalicia7/-data-insight-ml

---

## 📊 Project Overview

### The Challenge
NGOs struggle with:
- **Messy data** with missing values, inconsistencies, and quality issues
- **Multiple prediction needs** across different domains (donors, students, programs)
- **Lack of transparency** in data cleaning and ML predictions
- **Limited resources** for custom ML solutions

### Our Solution
A **unified ML platform** that provides:
- ✅ **6 specialized domains** - Donor retention, student dropout, program completion, grant scoring, customer churn, child wellbeing
- ✅ **Interactive data cleaning** - User-approved suggestions with before/after visualization
- ✅ **Real-time visualizations** - Quality metrics, prediction distributions, confidence analysis
- ✅ **CSV batch processing** - Upload files with IDs/names for bulk predictions
- ✅ **Downloadable reports** - Cleaned data and prediction results

---

## 🎯 Key Features

### 1. Multi-Domain Support
Six specialized ML models for different NGO use cases:
- **Donor Retention** - Predict which donors will continue giving
- **Student Dropout** - Identify at-risk students early
- **Program Completion** - Forecast program success rates
- **Grant Scoring** - Rank grant applications automatically
- **Customer Churn** - Predict member/customer retention
- **Child Wellbeing** - Assess child health and development

### 2. Interactive Data Cleaning Workflow
**7-step user-controlled process:**
1. Upload CSV → Automatic quality analysis
2. View data quality score and issue breakdown
3. Review AI-generated cleaning suggestions
4. Approve/reject each suggestion with checkboxes
5. Add custom cleaning notes
6. Apply cleaning & generate predictions
7. Download cleaned CSV and prediction report

**Detects 6 types of issues:**
- Missing values (intelligent imputation)
- Duplicate rows
- Outliers (IQR-based detection)
- Invalid ranges (domain-specific validation)
- Categorical inconsistencies
- Missing derived features

### 3. Visual Analytics Dashboard
**Data Quality Visualizations:**
- Quality Score Gauge (color-coded: green/orange/red)
- Issues Breakdown Pie Chart
- Missing Values by Column Bar Chart

**Prediction Analytics:**
- Prediction Distribution (positive/negative split)
- Confidence Distribution Histogram (5 bins)

### 4. CSV Upload with ID/Name Preservation
- Automatically detects ID columns (student_id, donor_id, etc.)
- Preserves names and identifiers throughout processing
- Returns predictions with "ID - Name" format
- Supports batch processing of 100+ records

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/weilalicia7/-data-insight-ml.git
cd -data-insight-ml

# Install dependencies
cd data_insight_ml
pip install -r requirements.txt

# Start Flask API server
python app.py
```

Server runs on: **http://localhost:5000**

### Using the Interface

1. **Open** `data_insight_ml/demo.html` in your browser
2. **Select** a domain (e.g., "Student Dropout Risk")
3. **Upload** your CSV file (or try `messy_student_data.csv` demo)
4. **Toggle ON** "Enable Interactive Data Cleaning"
5. **Review** quality analysis and suggestions
6. **Approve** cleaning steps and click "Apply Cleaning & Predict"
7. **View** predictions with visualizations
8. **Download** cleaned data and prediction report

### Try the Demo

```bash
cd data_insight_ml

# 1. Create demo messy data
python create_messy_demo_data.py

# 2. Test the data cleaner standalone
python data_cleaner.py messy_student_data.csv student

# 3. Open demo.html and upload messy_student_data.csv
```

---

## 📂 Project Structure

```
-data-insight-ml/
│
├── data_insight_ml/              # Main application directory
│   ├── app.py                    # Flask REST API server
│   ├── demo.html                 # Interactive web interface
│   ├── domain_manager.py         # Multi-domain ML orchestrator
│   ├── data_cleaner.py          # Standalone data cleaning tool
│   │
│   ├── domains/                  # Domain-specific configurations
│   │   ├── student_dropout/
│   │   ├── donor_retention/
│   │   ├── program_completion/
│   │   ├── grant_scoring/
│   │   ├── customer_churn/
│   │   └── child_wellbeing/
│   │
│   ├── models/                   # Pre-trained ML models
│   │   ├── best_model.pkl
│   │   ├── scaler.pkl
│   │   ├── feature_columns.pkl
│   │   └── model_info.yaml
│   │
│   └── messy_student_data.csv   # Demo file with data quality issues
│
├── backend/                      # Alternative backend implementation
├── frontend/                     # Alternative frontend files
├── DATA_CLEANING_SUMMARY.md     # Data cleaning documentation
├── INTERACTIVE_CLEANING_*.md    # Interactive cleaning guides
├── README.md                     # This file
└── requirements.txt              # Python dependencies
```

---

## 🔧 API Endpoints

### Core Prediction Endpoints

```bash
# Health check
GET /api/health

# Get available domains
GET /api/domains

# Single prediction
POST /api/predict
Body: {
  "age": 16,
  "attendance_rate": 0.85,
  "gpa": 3.2,
  ...
}

# CSV batch upload
POST /api/upload-predict
Form-Data:
  - file: CSV file
  - domain: "student_dropout"
```

### Interactive Cleaning Endpoints

```bash
# Analyze CSV for quality issues
POST /api/analyze-csv
Form-Data:
  - file: CSV file
  - domain: "student"

# Clean and predict with approved suggestions
POST /api/clean-and-predict
Form-Data:
  - file: CSV file
  - domain: "student"
  - suggestions: JSON array of approved suggestions

# Download cleaned CSV
GET /api/download-cleaned-csv/<filename>

# Download prediction report
POST /api/download-prediction-report
Body: Prediction results JSON
```

---

## 📊 Data Quality Improvements

### Example: Messy Student Data

**Before Cleaning:**
- 53 rows, 13 columns
- 97 missing values (17.5% of data)
- 3 duplicate rows
- 2 invalid ranges
- 4 outliers
- Inconsistent formatting

**After Cleaning:**
- 50 rows, 18 columns
- 1 missing value (99.8% complete)
- 0 duplicates
- All values within valid ranges
- Outliers capped
- 5 new derived features (age_group, risk_score, etc.)

**Quality Score:** 82.5% → 99.8% (+17.3%)

---

## 🎨 Visualizations

### Data Quality Charts
1. **Quality Gauge** - Doughnut chart with center score percentage
2. **Issues Breakdown** - Pie chart showing distribution of problem types
3. **Missing Values** - Bar chart of top 10 columns with missing data

### Prediction Analytics
1. **Prediction Distribution** - Positive vs Negative split
2. **Confidence Distribution** - Histogram across 5 confidence bins (0-20%, 20-40%, etc.)

All charts use **Chart.js 4.4.0** with:
- Dark theme compatibility
- Interactive tooltips
- Responsive design
- Professional color scheme

---

## 🧪 Testing

### Test Files Included
- `messy_student_data.csv` - Demo file with data quality issues
- `demo_upload_*.csv` - Example CSVs for all 6 domains
- `test_api.py` - API endpoint tests
- `quick_test.py` - Quick validation script

### Run Tests

```bash
# Test API health
curl http://localhost:5000/api/health

# Test data cleaning
python data_cleaner.py messy_student_data.csv student

# View before/after
# Original: messy_student_data.csv
# Cleaned: messy_student_data_cleaned.csv
```

---

## 📚 Documentation

- **DATA_CLEANING_SUMMARY.md** - Complete data cleaning toolkit guide
- **INTERACTIVE_CLEANING_SUMMARY.md** - Interactive cleaning feature overview
- **INTERACTIVE_CLEANING_IMPLEMENTATION.md** - Technical implementation details
- **FILE_GUIDE.md** - Detailed file-by-file documentation
- **PROJECT_CONTEXT_AND_CLARIFICATION.md** - Project context and requirements

---

## 🔒 Privacy & Security

- All data processing happens locally
- No data stored on servers
- Temporary cleaned files auto-deleted after download
- CORS enabled for local development
- Use environment variables for production secrets

---

## 🛠️ Technology Stack

### Backend
- **Flask 3.0.0** - REST API framework
- **scikit-learn** - ML model (Random Forest, Logistic Regression)
- **pandas** - Data manipulation
- **numpy** - Numerical operations
- **PyYAML** - Configuration management

### Frontend
- **HTML5/CSS3** - Modern responsive design
- **JavaScript (ES6+)** - Dynamic interactions
- **Chart.js 4.4.0** - Data visualizations
- **FormData API** - File uploads

### ML Pipeline
- **Random Forest** - Primary model type
- **StandardScaler** - Feature normalization
- **Domain-specific features** - Custom feature engineering per domain
- **IQR outlier detection** - Statistical outlier handling

---

## 🎓 Use Cases

### For NGOs
- Clean messy survey data before analysis
- Predict donor retention to focus outreach
- Identify at-risk students for interventions
- Score grant applications automatically
- Forecast program completion rates

### For Data Teams
- Rapid prototyping of ML solutions
- Automated data quality assessment
- Batch prediction processing
- Downloadable cleaned datasets for retraining

### For Researchers
- Reproducible data cleaning workflows
- Transparent ML predictions with confidence scores
- Domain-specific feature engineering examples
- Visualization of data quality metrics

---

## 🤝 Contributing

This project was developed for the JP Morgan Data for Good Hackathon 2025.

**Team 2 Members:**
- Development and implementation
- ML model training and optimization
- Interactive UI/UX design
- Documentation and testing

---

## 📝 License

This project is open-source and available for nonprofit and educational use.

---

## 🙏 Acknowledgments

- **JP Morgan** - Data for Good Hackathon 2025
- **Chart.js** - Beautiful visualizations
- **Flask** - Lightweight web framework
- **scikit-learn** - Powerful ML toolkit

---

## 📧 Support

For issues, questions, or contributions:
- 🔗 GitHub: https://github.com/weilalicia7/-data-insight-ml
- 📁 Documentation: See `/docs` folder
- 🐛 Report issues: GitHub Issues tab

---

## ⭐ Quick Links

- [Installation Guide](#quick-start)
- [API Documentation](#api-endpoints)
- [Data Cleaning Guide](DATA_CLEANING_SUMMARY.md)
- [Interactive Cleaning Guide](INTERACTIVE_CLEANING_SUMMARY.md)
- [Project Structure](#project-structure)
- [Try the Demo](#try-the-demo)

---

**Built with ❤️ for nonprofits and social good initiatives**

*Transforming messy data into actionable insights through transparent, interactive machine learning*
