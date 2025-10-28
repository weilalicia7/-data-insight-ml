# Quick Start Guide

Get up and running with Data Insight ML in 5 steps!

## Step 1: Install Dependencies (2 minutes)

Open terminal/command prompt and run:

```bash
pip install -r requirements.txt
```

This installs all necessary Python packages.

## Step 2: Verify Installation (30 seconds)

```bash
python quick_test.py
```

This checks if everything is installed correctly.

## Step 3: Prepare Your Data (1 minute)

Place your CSV file in this folder, then run:

```bash
python prepare_data.py your_data.csv
```

Example:
```bash
python prepare_data.py customer_data.csv
```

The script will:
- Auto-detect your data types
- Handle missing values
- Engineer features
- Save prepared data

**Interactive**: The script will ask questions if it needs clarification.

## Step 4: Train Model (3-5 minutes)

```bash
python train_model.py
```

This will:
- Train multiple ML models
- Compare their performance
- Select the best one
- Save the trained model

You'll see accuracy metrics and model comparisons.

## Step 5: Start API & Make Predictions (30 seconds)

Start the API server:

```bash
python app.py
```

The API is now running at http://localhost:5000

### Option A: Use Web Interface

Open `demo.html` in your web browser.

### Option B: Use API Directly

Test with curl:

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"feature1": 25, "feature2": "value2"}'
```

### Option C: Python Code

```python
import requests

response = requests.post('http://localhost:5000/api/predict', json={
    'feature1': 25,
    'feature2': 'CategoryA',
    'feature3': 100
})

print(response.json())
```

## Common Issues

### Issue: "Module not found"

**Solution**: Install requirements
```bash
pip install -r requirements.txt
```

### Issue: "File not found" when training

**Solution**: Run data preparation first
```bash
python prepare_data.py your_data.csv
```

### Issue: "Port 5000 already in use"

**Solution**: Change port in `config.yaml`:
```yaml
api:
  port: 5001  # or any other available port
```

### Issue: Low model accuracy

**Solutions**:
- Add more training data
- Check data quality
- Add more relevant features
- Adjust model parameters in `config.yaml`

## Next Steps

Once everything is working:

1. **Customize Configuration**: Edit `config.yaml` to adjust settings
2. **Add Custom Features**: Modify `prepare_data.py` for domain-specific features
3. **Tune Models**: Adjust hyperparameters in `config.yaml`
4. **Build Your App**: Use the API to build your own application

## Getting Help

- Check `README.md` for detailed documentation
- Review example files in `examples/` folder (if available)
- Check configuration options in `config.yaml`

## Success!

If you've completed all 5 steps, your ML system is ready to use!

The typical workflow is:
1. Collect more data
2. Re-run `prepare_data.py`
3. Re-run `train_model.py`
4. Restart `app.py`

This updates your model with new data.
