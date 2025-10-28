# Usage Guide for NGOs

This guide explains how to use Data Insight ML for common NGO use cases.

## Use Case 1: Donor Retention Prediction

**Goal**: Predict which donors are likely to donate again

### Step 1: Prepare Your Data

Create a CSV file with donor information:

```csv
donor_id,last_donation_amount,donation_frequency,years_since_first,email_opens,age,region,donated_again
1,100,5,3,12,45,North,1
2,50,2,1,3,28,South,0
3,250,8,5,20,55,East,1
...
```

### Step 2: Configure

Edit `config.yaml`:

```yaml
data:
  target_column: "donated_again"
  id_columns: ["donor_id"]
```

### Step 3: Run

```bash
python prepare_data.py donor_data.csv
python train_model.py
python app.py
```

### Step 4: Use Predictions

Now you can predict for new donors:

```python
import requests

new_donor = {
    "last_donation_amount": 75,
    "donation_frequency": 3,
    "years_since_first": 2,
    "email_opens": 8,
    "age": 35,
    "region_North": 0,
    "region_South": 1,
    "region_East": 0
}

response = requests.post('http://localhost:5000/api/predict', json=new_donor)
print(response.json())
# Output: {"prediction": 1, "confidence": 0.82}
```

**Interpretation**:
- Prediction = 1 means likely to donate again
- Confidence = 82% means model is fairly certain

## Use Case 2: Program Completion Prediction

**Goal**: Identify participants at risk of dropping out

### Data Structure

```csv
participant_id,age,education,income_level,attendance_rate,engagement_score,mentor_assigned,completed
1,22,High School,Low,0.85,7,1,1
2,25,Bachelor,Medium,0.60,4,0,0
3,30,Master,High,0.95,9,1,1
...
```

### Key Steps

1. Set `target_column: "completed"` in config.yaml
2. Run preparation and training
3. Use predictions to identify at-risk participants early

### Actionable Insights

If prediction = 0 (won't complete) and confidence > 70%:
- Assign additional mentorship
- Increase check-ins
- Provide extra resources

## Use Case 3: Grant Application Scoring

**Goal**: Automatically score grant applications

### Data Structure

```csv
application_id,org_size,years_operating,budget,previous_grants,mission_alignment,proposal_quality,approved
1,Small,5,50000,2,High,8,1
2,Medium,10,200000,5,Medium,6,1
3,Large,2,500000,0,Low,4,0
...
```

### Workflow

1. Train model on historical approved/rejected applications
2. Use API to score new applications as they arrive
3. Review high-scoring applications first

### Code Example

```python
import pandas as pd
import requests

# Load new applications
new_apps = pd.read_csv('new_applications.csv')

# Score each application
for _, app in new_apps.iterrows():
    response = requests.post('http://localhost:5000/api/predict',
                            json=app.to_dict())

    result = response.json()

    print(f"Application {app['application_id']}:")
    print(f"  Approval probability: {result['confidence']*100:.1f}%")

    if result['prediction'] == 1 and result['confidence'] > 0.7:
        print("  ✓ High priority - Review soon")
    elif result['prediction'] == 0 and result['confidence'] > 0.7:
        print("  ✗ Low priority")
    else:
        print("  ? Borderline - Manual review recommended")
```

## Use Case 4: Resource Allocation

**Goal**: Predict which regions/programs need most resources

### Approach

1. **Historical Analysis**: Train on past allocation vs outcomes
2. **Feature Engineering**: Include population, needs assessment, past performance
3. **Prediction**: Predict resource needs for next period

### Example Features

- Population size
- Poverty rate
- Past program success rate
- Current capacity
- Geographic accessibility
- Community engagement

## Best Practices

### Data Quality

1. **Minimum Samples**: Aim for at least 500 samples for reliable models
2. **Balanced Classes**: Try to have similar numbers of positive/negative cases
3. **Representative Data**: Ensure data covers all scenarios you'll encounter

### Feature Selection

**Good Features**:
- Directly related to outcome (e.g., attendance → completion)
- Measurable and available at prediction time
- Not biased or discriminatory

**Avoid**:
- Protected characteristics (race, religion) unless legally justified
- Data not available when making predictions
- Redundant information

### Model Interpretation

1. **Use Feature Importance**: Check `models/feature_importance.csv`
2. **Validate Predictions**: Test on known cases first
3. **Set Thresholds**: Determine what confidence level you trust

### Continuous Improvement

```bash
# Monthly: Add new data and retrain
python prepare_data.py updated_data.csv
python train_model.py

# Compare: Check if accuracy improved
# File: models/model_info.yaml shows test accuracy
```

## Integration Examples

### Google Sheets Integration

```python
import gspread
import requests
from oauth2client.service_account import ServiceAccountCredentials

# Connect to Google Sheets
scope = ['https://spreadsheets.google.com/feeds']
creds = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scope)
client = gspread.authorize(creds)

# Open sheet
sheet = client.open("Donor Database").sheet1

# Get data
donors = sheet.get_all_records()

# Make predictions
for donor in donors:
    response = requests.post('http://localhost:5000/api/predict', json=donor)
    prediction = response.json()

    # Update sheet with prediction
    sheet.update_cell(donor['row'], 'prediction_column', prediction['prediction'])
```

### Email Automation

```python
import smtplib
import requests

# Get at-risk participants
participants = get_participants()  # Your data source

for participant in participants:
    response = requests.post('http://localhost:5000/api/predict',
                            json=participant)

    if response.json()['prediction'] == 0:  # At risk
        send_email(
            to=participant['email'],
            subject="We're here to help!",
            body="We noticed you might need extra support..."
        )
```

### Database Integration

```python
import psycopg2
import requests

# Connect to database
conn = psycopg2.connect("dbname=ngo_db user=postgres")
cur = conn.cursor()

# Get records to predict
cur.execute("SELECT * FROM applications WHERE status='pending'")
applications = cur.fetchall()

# Make predictions and update database
for app in applications:
    response = requests.post('http://localhost:5000/api/predict',
                            json=dict(app))

    prediction = response.json()

    cur.execute("""
        UPDATE applications
        SET ml_score = %s, ml_confidence = %s
        WHERE id = %s
    """, (prediction['prediction'], prediction['confidence'], app['id']))

conn.commit()
```

## Troubleshooting Common Issues

### Low Accuracy (< 60%)

**Possible causes**:
- Not enough data
- Poor feature selection
- Wrong model type for your data

**Solutions**:
- Collect more data (aim for 1000+ samples)
- Add more relevant features
- Try adjusting hyperparameters in config.yaml

### Biased Predictions

**Check for bias**:
```python
# Group predictions by demographic
df['prediction'] = df.apply(lambda row: predict(row), axis=1)
print(df.groupby('demographic_group')['prediction'].mean())
```

**Fix**:
- Remove biased features
- Ensure balanced training data
- Consult ethics guidelines

### API Errors

**"Model not found"**: Run `python train_model.py` first

**"Connection refused"**: Check if `python app.py` is running

**"Invalid input"**: Verify feature names match training data

## Getting Help

For issues specific to your use case:
1. Check data quality first
2. Review similar examples in this guide
3. Adjust configuration settings
4. Consider domain-specific feature engineering

## Ethical Considerations

When using ML for social impact:

1. **Transparency**: Explain to stakeholders how predictions are made
2. **Fairness**: Regularly check for bias in predictions
3. **Privacy**: Protect sensitive data, comply with regulations
4. **Human Oversight**: ML assists decisions, humans make final calls
5. **Accountability**: Document decisions and model versions

## Success Metrics

Track these metrics to measure impact:

- **Model Performance**: Accuracy, precision, recall
- **Business Impact**: Time saved, better outcomes, cost reduction
- **User Adoption**: How often predictions are used
- **Feedback Loop**: How often predictions are correct in practice

Remember: ML is a tool to augment human decision-making, not replace it!
