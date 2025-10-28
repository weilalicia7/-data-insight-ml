# Getting Started with Data Insight ML

Welcome! This toolkit makes machine learning accessible to NGOs and non-profits. No ML expertise required!

## What Can This Do?

This toolkit helps you:
- **Predict outcomes** (will a donor give again? will a participant complete the program?)
- **Score applications** (which grant applications to prioritize?)
- **Identify risks** (who is at risk of dropping out?)
- **Optimize resources** (where to allocate limited resources?)

## 5-Minute Quick Test

Want to see it in action before using your own data? Generate example data:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate example datasets
python example_data_generator.py

# 3. Try with donor retention example
python prepare_data.py example_donor_retention.csv
python train_model.py
python app.py

# 4. Open demo.html in your browser
```

## Using Your Own Data

### Step 1: Prepare Your CSV

Your data should look like this:

```csv
id,feature1,feature2,feature3,...,target
1,value1,value2,value3,...,outcome1
2,value1,value2,value3,...,outcome2
...
```

**Requirements**:
- At least 100 rows (more is better - aim for 500+)
- One column as your "target" (what you want to predict)
- Other columns as "features" (information that might help predict)
- No sensitive/private data unless properly secured

### Step 2: Configure

Edit `config.yaml` and set:

```yaml
data:
  target_column: "your_target_column_name"  # e.g., "approved", "completed", "churned"
```

### Step 3: Run the Pipeline

```bash
# Prepare your data
python prepare_data.py your_data.csv

# Train model
python train_model.py

# Start API
python app.py
```

### Step 4: Make Predictions

**Option A - Web Interface**:
Open `demo.html` in your browser

**Option B - Python Code**:
```python
import requests

data = {
    "feature1": value1,
    "feature2": value2,
    ...
}

response = requests.post('http://localhost:5000/api/predict', json=data)
print(response.json())
```

**Option C - cURL**:
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"feature1": 10, "feature2": "value"}'
```

## Understanding Your Results

### Prediction
- The model's guess for this case
- Example: `"prediction": 1` means "yes" or "positive" class

### Confidence
- How certain the model is (0-100%)
- 90%+ = Very confident
- 70-90% = Fairly confident
- 50-70% = Uncertain - use caution
- <50% = Not confident - needs manual review

### Using Predictions Wisely

**HIGH confidence (>80%)**:
- Can inform decisions
- Still review edge cases

**MEDIUM confidence (60-80%)**:
- Use as one input among many
- Combine with human judgment

**LOW confidence (<60%)**:
- Don't rely on predictions
- Focus on improving model or collecting more data

## Common Workflows

### Workflow 1: Batch Scoring

Score many cases at once:

```python
import pandas as pd
import requests

# Load your data
df = pd.read_csv('cases_to_score.csv')

# Score each case
predictions = []
for _, row in df.iterrows():
    response = requests.post('http://localhost:5000/api/predict',
                            json=row.to_dict())
    predictions.append(response.json())

# Add predictions to dataframe
df['prediction'] = [p['prediction'] for p in predictions]
df['confidence'] = [p['confidence'] for p in predictions]

# Save results
df.to_csv('scored_cases.csv', index=False)
```

### Workflow 2: Real-time Predictions

Integrate into your application:

```python
from flask import Flask, request
import requests

app = Flask(__name__)

@app.route('/submit-application', methods=['POST'])
def handle_application():
    application_data = request.json

    # Get ML prediction
    ml_response = requests.post('http://localhost:5000/api/predict',
                               json=application_data)
    prediction = ml_response.json()

    # Use prediction to prioritize
    if prediction['confidence'] > 0.8:
        if prediction['prediction'] == 1:
            priority = 'HIGH'
        else:
            priority = 'LOW'
    else:
        priority = 'MEDIUM'

    # Store application with priority
    save_application(application_data, priority)

    return {'status': 'success', 'priority': priority}
```

### Workflow 3: Monthly Model Updates

Keep your model fresh:

```bash
# 1. Export new data from your database
# (your export script)

# 2. Combine with previous data
cat old_data.csv new_data.csv > updated_data.csv

# 3. Retrain
python prepare_data.py updated_data.csv
python train_model.py

# 4. Restart API
# Stop the running API (Ctrl+C)
python app.py
```

## File Overview

| File | Purpose |
|------|---------|
| `README.md` | Full documentation |
| `QUICKSTART.md` | 5-step quick guide |
| `USAGE_GUIDE.md` | Detailed use cases for NGOs |
| `config.yaml` | Configuration settings |
| `prepare_data.py` | Prepare your data |
| `train_model.py` | Train ML models |
| `app.py` | Prediction API server |
| `demo.html` | Web interface |
| `quick_test.py` | Verify setup |
| `example_data_generator.py` | Generate test data |
| `requirements.txt` | Python dependencies |

## Troubleshooting

### "ModuleNotFoundError"
```bash
pip install -r requirements.txt
```

### "Target column not found"
Edit `config.yaml` and set the correct target column name.

### "Model not loaded"
Run `python train_model.py` first.

### "Connection refused"
Make sure `python app.py` is running.

### Low Accuracy
- Collect more data (aim for 1000+ samples)
- Add more relevant features
- Check data quality
- See USAGE_GUIDE.md for tips

## Next Steps

1. **Try the examples**: Run `python example_data_generator.py` and test with sample data
2. **Use your data**: Follow Step 1-4 above
3. **Read use cases**: Check USAGE_GUIDE.md for your specific use case
4. **Customize**: Adjust config.yaml and scripts for your needs
5. **Deploy**: Move to production when ready

## Need Help?

- **Setup issues**: Run `python quick_test.py` to diagnose
- **Usage questions**: Check USAGE_GUIDE.md
- **Configuration**: Review config.yaml comments
- **Examples**: Run example_data_generator.py

## Important Reminders

- **ML assists, doesn't replace**: Always combine ML with human judgment
- **Privacy matters**: Protect sensitive data
- **Check for bias**: Regularly audit predictions for fairness
- **Keep learning**: Models need new data to stay accurate
- **Document decisions**: Keep track of what models you use and why

## Contributing Back

If this toolkit helps your organization:
- Share your success story (helps other NGOs)
- Contribute improvements back to the project
- Spread the word to other non-profits

## Mission

This toolkit was created to democratize machine learning for social good. It's completely free and open-source. Use it to maximize your impact!

---

**Ready to start?** Run: `python example_data_generator.py`
