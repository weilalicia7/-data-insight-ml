# Grant Application Scoring

## Description
Score and prioritize grant applications automatically

## Use Case
Efficiently evaluate applications and identify top candidates

## Model Information
- **Model Type:** Random Forest
- **Accuracy:** 71.00%
- **Precision:** 73.94%
- **Recall:** 71.00%
- **F1-Score:** 0.7127

## Target Variable
- **Name:** approved
- **Description:** 1 = Approved, 0 = Rejected

## Features
Total features: 9

## Example Data
Rows available: 500

## Getting Started

```python
from domain_manager import DomainManager

# Load domain
dm = DomainManager()
dm.load_domain('grant_scoring')

# Make prediction
result = dm.predict({
    # ... your feature values
})

print(result)
```

## Sample Predictions

[
  {
    "name": "Strong application",
    "description": "Established org with strong proposal",
    "expected": "Approve (high confidence)"
  },
  {
    "name": "Weak application",
    "description": "New org with unclear objectives",
    "expected": "Reject (medium confidence)"
  }
]

## Generated
2025-10-28 19:28:31
