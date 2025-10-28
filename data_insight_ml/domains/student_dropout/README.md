# Student Dropout Risk Prediction

## Description
Identify students at risk of dropping out early

## Use Case
Provide early intervention to keep students in school

## Model Information
- **Model Type:** Random Forest
- **Accuracy:** 80.00%
- **Precision:** 80.54%
- **Recall:** 80.00%
- **F1-Score:** 0.7991

## Target Variable
- **Name:** at_risk
- **Description:** 1 = At risk of dropout, 0 = On track

## Features
Total features: 12

## Example Data
Rows available: 600

## Getting Started

```python
from domain_manager import DomainManager

# Load domain
dm = DomainManager()
dm.load_domain('student_dropout')

# Make prediction
result = dm.predict({
    # ... your feature values
})

print(result)
```

## Sample Predictions

[
  {
    "name": "High-risk student",
    "description": "Low attendance, poor grades, limited parent involvement",
    "expected": "At risk (high confidence)"
  },
  {
    "name": "Thriving student",
    "description": "Good attendance, strong GPA, active in extracurriculars",
    "expected": "On track (high confidence)"
  }
]

## Generated
2025-10-28 19:28:33
