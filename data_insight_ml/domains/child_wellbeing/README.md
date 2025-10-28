# Child Wellbeing Risk Assessment

## Description
Identify children who need additional support

## Use Case
Ensure all children receive necessary care and support

## Model Information
- **Model Type:** Random Forest
- **Accuracy:** 79.00%
- **Precision:** 79.10%
- **Recall:** 79.00%
- **F1-Score:** 0.7898

## Target Variable
- **Name:** needs_support
- **Description:** 1 = Needs support, 0 = Doing well

## Features
Total features: 17

## Example Data
Rows available: 500

## Getting Started

```python
from domain_manager import DomainManager

# Load domain
dm = DomainManager()
dm.load_domain('child_wellbeing')

# Make prediction
result = dm.predict({
    # ... your feature values
})

print(result)
```

## Sample Predictions

[
  {
    "name": "Thriving child",
    "description": "Good nutrition, health checkups, supportive home",
    "expected": "Doing well (high confidence)"
  },
  {
    "name": "At-risk child",
    "description": "Poor nutrition, missed checkups, challenging home environment",
    "expected": "Needs support (high confidence)"
  }
]

## Generated
2025-10-28 19:28:33
