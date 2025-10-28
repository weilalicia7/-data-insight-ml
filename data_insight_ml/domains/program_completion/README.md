# Program Completion Prediction

## Description
Predict which program participants will complete successfully

## Use Case
Identify at-risk participants early and provide targeted support

## Model Information
- **Model Type:** Random Forest
- **Accuracy:** 83.12%
- **Precision:** 83.38%
- **Recall:** 83.12%
- **F1-Score:** 0.8309

## Target Variable
- **Name:** completed
- **Description:** 1 = Completed program, 0 = Did not complete

## Features
Total features: 10

## Example Data
Rows available: 800

## Getting Started

```python
from domain_manager import DomainManager

# Load domain
dm = DomainManager()
dm.load_domain('program_completion')

# Make prediction
result = dm.predict({
    # ... your feature values
})

print(result)
```

## Sample Predictions

[
  {
    "name": "High-attendance participant",
    "description": "Strong engagement and mentor support",
    "expected": "Will complete (high confidence)"
  },
  {
    "name": "At-risk participant",
    "description": "Low attendance and engagement",
    "expected": "At risk of dropout (high confidence)"
  }
]

## Generated
2025-10-28 19:28:30
