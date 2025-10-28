# Data Insight ML - Free ML Analytics Toolkit for NGOs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Free for NGOs](https://img.shields.io/badge/Free%20for-NGOs-green.svg)](https://github.com/weilalicia7/-data-insight-ml)
[![Social Impact](https://img.shields.io/badge/Social-Impact-red.svg)](https://github.com/weilalicia7/-data-insight-ml)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/weilalicia7/-data-insight-ml/pulls)

**A free, open-source machine learning toolkit designed for non-profits and NGOs to analyze their data and make predictions.**

## What is this?

Data Insight ML is a ready-to-use machine learning system that allows organizations to:
- Upload their own CSV data
- Automatically train prediction models
- Get insights and predictions through a web interface
- No ML expertise required!

## Quick Start (5 minutes)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data

Place your CSV file in the `data_insight_ml` folder and run:

```bash
python prepare_data.py your_data.csv
```

This will analyze your data and prepare it for training.

### 3. Train Your Model

```bash
python train_model.py
```

This trains a machine learning model on your data (takes 2-5 minutes).

### 4. Start the Prediction API

```bash
python app.py
```

Your API is now running at http://localhost:5000

### 5. Use the Web Interface

Open `demo.html` in your web browser to interact with your model!

## Features

- **Auto-detection**: Automatically detects data types (numeric, categorical, dates)
- **Feature Engineering**: Creates useful features from your data
- **Multiple Algorithms**: Random Forest, Logistic Regression, XGBoost
- **Model Evaluation**: Cross-validation, accuracy metrics, feature importance
- **REST API**: Easy-to-use API for predictions
- **Web Interface**: User-friendly demo page
- **Free Forever**: Open source, no costs, no limitations

## Data Requirements

Your CSV file should:
- Have a **target column** (the thing you want to predict, e.g., "success", "failure", "approved")
- Have at least 100 rows (more is better)
- Have relevant features (columns that might influence the outcome)

Example data structure:

```csv
applicant_id,age,income,education,region,credit_score,approved
1,25,35000,Bachelor,North,680,1
2,45,85000,Master,South,720,1
3,22,28000,High School,East,590,0
...
```

## File Structure

```
data_insight_ml/
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── prepare_data.py        # Data preparation script
├── train_model.py         # Model training script
├── app.py                 # Flask API server
├── demo.html              # Web interface
├── quick_test.py          # Test your setup
├── config.yaml            # Configuration (optional)
├── models/                # Trained models saved here
└── data/                  # Your data files go here
```

## Configuration (Optional)

Edit `config.yaml` to customize:
- Target column name
- Columns to exclude
- Model hyperparameters
- Feature engineering options

## API Endpoints

### Health Check
```
GET /api/health
```

### Make Prediction
```
POST /api/predict
Content-Type: application/json

{
  "feature1": value1,
  "feature2": value2,
  ...
}
```

## Advanced Usage

### Custom Feature Engineering

Edit `prepare_data.py` to add custom features specific to your domain.

### Model Selection

The toolkit automatically tries multiple models and selects the best one. You can customize this in `train_model.py`.

### Deployment

For production use:
1. Use `gunicorn` instead of Flask development server
2. Add authentication to your API
3. Set up HTTPS
4. Monitor model performance over time

## Common Use Cases

- **Donor Prediction**: Predict which donors are likely to give again
- **Program Success**: Predict which participants will complete programs
- **Resource Allocation**: Predict which areas need most resources
- **Risk Assessment**: Identify high-risk cases early
- **Application Scoring**: Score grant/aid applications automatically

## Troubleshooting

**Issue**: Model accuracy is low
- **Solution**: Add more data, check data quality, add relevant features

**Issue**: API won't start
- **Solution**: Check if port 5000 is available, verify all dependencies installed

**Issue**: Predictions seem wrong
- **Solution**: Verify your data format matches training data structure

## Support

This is a free toolkit developed for the social good. For issues or questions:
- Check documentation in `/docs` folder
- Review example notebooks
- Open an issue on GitHub

## License

MIT License - Free to use for any purpose, including commercial.

## Credits

Developed by open-source contributors for social good.
Designed to democratize ML for social impact organizations.

---

**Remember**: This tool helps you make predictions, but always combine ML insights with human judgment and domain expertise!
