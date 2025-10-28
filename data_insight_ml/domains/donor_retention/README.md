# Donor Retention Prediction

## Description
Predict which donors are likely to donate again

## Use Case
Optimize fundraising by focusing on high-probability donors

## Model Information
- **Model Type:** Random Forest
- **Accuracy:** 74.50%
- **Precision:** 74.52%
- **Recall:** 74.50%
- **F1-Score:** 0.7449

## Target Variable
- **Name:** donated_again
- **Description:** 1 = Will donate again, 0 = Will not donate

## Features
Total features: 8

## Example Data
Rows available: 1000

## Getting Started

```python
from domain_manager import DomainManager

# Load domain
dm = DomainManager()
dm.load_domain('donor_retention')

# Make prediction
result = dm.predict({
    # ... your feature values
})

print(result)
```

## Sample Predictions

[
  {
    "name": "High-value engaged donor",
    "description": "Regular donor with high engagement",
    "expected": "Will donate (high confidence)"
  },
  {
    "name": "At-risk donor",
    "description": "Infrequent donor with low engagement",
    "expected": "Will not donate (high confidence)"
  }
]

## Generated
2025-10-28 19:28:29
