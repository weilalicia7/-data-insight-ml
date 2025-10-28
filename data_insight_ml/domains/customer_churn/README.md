# Member Churn Prediction

## Description
Predict which members are at risk of churning

## Use Case
Retain members through proactive engagement

## Model Information
- **Model Type:** Random Forest
- **Accuracy:** 87.00%
- **Precision:** 87.06%
- **Recall:** 87.00%
- **F1-Score:** 0.8699

## Target Variable
- **Name:** churned
- **Description:** 1 = Churned, 0 = Retained

## Features
Total features: 9

## Example Data
Rows available: 1000

## Getting Started

```python
from domain_manager import DomainManager

# Load domain
dm = DomainManager()
dm.load_domain('customer_churn')

# Make prediction
result = dm.predict({
    # ... your feature values
})

print(result)
```

## Sample Predictions

[
  {
    "name": "Engaged member",
    "description": "Active usage and long tenure",
    "expected": "Low churn risk"
  },
  {
    "name": "At-risk member",
    "description": "Declining activity and support tickets",
    "expected": "High churn risk"
  }
]

## Generated
2025-10-28 19:28:32
