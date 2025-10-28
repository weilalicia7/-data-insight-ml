# Multi-Domain System - Quick Start

## 🎯 What is This?

The multi-domain system allows NGOs to:
- **Choose their use case** (donor retention, program completion, grants, etc.)
- **Try demo data instantly** (no setup needed!)
- **See pre-trained model predictions**
- **Upload their own data** for that specific domain
- **Get domain-specific recommendations**

---

## 🚀 Setup (One-Time, 10 Minutes)

### Step 1: Generate Demo Domains

Run this command to create 4 pre-configured domains with synthetic data:

```bash
python setup_domains.py
```

**What this does:**
- Creates `domains/` folder
- Generates synthetic data for each domain (1000+ rows each)
- Trains models for each domain
- Saves pre-trained models, scalers, and metadata
- Creates documentation for each domain

**Output:**
```
domains/
├── donor_retention/
│   ├── example_data.csv (1000 donors)
│   ├── model.pkl (Random Forest, 77% accuracy)
│   ├── scaler.pkl
│   ├── features.pkl
│   ├── metadata.json
│   └── README.md
├── program_completion/
│   └── ... (similar structure)
├── grant_scoring/
│   └── ...
└── customer_churn/
    └── ...
```

**Time:** 5-10 minutes (trains 4 models)

### Step 2: Test Domain Manager

```bash
python domain_manager.py
```

**Output:**
```
Available Domains:
  💰 Donor Retention Prediction: ✓ Ready
     Predict which donors are likely to donate again
  🎓 Program Completion Prediction: ✓ Ready
     Predict which program participants will complete
  📋 Grant Application Scoring: ✓ Ready
     Score and prioritize grant applications
  👥 Member Churn Prediction: ✓ Ready
     Predict which members are at risk of churning

✓ Successfully loaded donor_retention
  Model: Random Forest
  Accuracy: 77.00%
  Features: 31
```

---

## 📊 Available Domains

### 1. 💰 Donor Retention
**Question:** Will this donor give again?

**Sample Input:**
- Last donation: $150
- Donation frequency: 5 times/year
- Email opens: 15/month
- Age: 45

**Sample Output:**
```json
{
  "prediction": "Will not donate",
  "confidence": 80.6%,
  "recommendation": "Initiate re-engagement campaign"
}
```

---

### 2. 🎓 Program Completion
**Question:** Will this participant complete the program?

**Sample Input:**
- Attendance rate: 85%
- Engagement score: 7/10
- Has mentor: Yes
- Previous programs: 2

**Sample Output:**
```json
{
  "prediction": "Will complete",
  "confidence": 91%,
  "recommendation": "Continue current support level"
}
```

---

### 3. 📋 Grant Scoring
**Question:** Should we approve this grant application?

**Sample Input:**
- Organization age: 5 years
- Budget requested: $50,000
- Previous grants: 2
- Mission alignment: High
- Proposal quality: 8/10

**Sample Output:**
```json
{
  "prediction": "Approve",
  "confidence": 85%,
  "recommendation": "Fast-track for review board"
}
```

---

### 4. 👥 Member Churn
**Question:** Will this member cancel their membership?

**Sample Input:**
- Months as member: 18
- Monthly spending: $75
- Support tickets: 1
- Feature usage: High
- Referrals: 3

**Sample Output:**
```json
{
  "prediction": "Low churn risk",
  "confidence": 88%,
  "recommendation": "Standard retention efforts"
}
```

---

## 💻 Using in Code

### Quick Prediction

```python
from domain_manager import quick_predict

# Make a prediction
result = quick_predict('donor_retention', {
    'last_donation_amount': 150,
    'donation_frequency': 5,
    'years_since_first': 3,
    'email_opens': 15,
    'age': 45
})

print(result)
# {
#   'prediction': 0,
#   'confidence': 0.806,
#   'recommendation': 'Initiate re-engagement campaign',
#   'domain': 'donor_retention'
# }
```

### Managing Multiple Domains

```python
from domain_manager import DomainManager

dm = DomainManager()

# List available domains
domains = dm.list_domains()
for domain in domains:
    print(f"{domain['icon']} {domain['display_name']}")

# Load a specific domain
dm.load_domain('program_completion')

# Get example data
examples = dm.get_example_data('program_completion', n_rows=5)
print(examples)

# Make predictions
result = dm.predict({
    'attendance_rate': 0.85,
    'engagement_score': 7,
    'mentor_assigned': 1
})

print(result)

# Switch to different domain
dm.load_domain('grant_scoring')
result = dm.predict({...})
```

### Get Domain Information

```python
from domain_manager import DomainManager

dm = DomainManager()

# Get detailed info about a domain
info = dm.get_domain_info('donor_retention')

print(f"Accuracy: {info['accuracy']:.2%}")
print(f"Features: {info['features_count']}")
print(f"Model: {info['model_type']}")
print(f"Use Case: {info['use_case']}")
```

---

## 🌐 Using with API (Coming Soon)

```python
# New API endpoints will support:

# List domains
GET /api/domains

# Load domain
POST /api/domain/donor_retention/load

# Get example data
GET /api/domain/donor_retention/example

# Make prediction
POST /api/domain/donor_retention/predict
```

---

## 📁 Domain Structure

Each domain folder contains:

```
donor_retention/
├── example_data.csv          # 1000 rows of synthetic data
├── model.pkl                  # Pre-trained Random Forest
├── scaler.pkl                 # Feature scaler
├── features.pkl               # List of feature names
├── metadata.json              # Domain configuration
└── README.md                  # Domain documentation
```

**metadata.json example:**
```json
{
  "domain": "donor_retention",
  "display_name": "Donor Retention Prediction",
  "description": "Predict which donors are likely to donate again",
  "icon": "💰",
  "target_variable": "donated_again",
  "model_type": "Random Forest",
  "accuracy": 0.77,
  "features_count": 31,
  "recommendations": {
    "action_positive": "Prioritize for next campaign",
    "action_negative": "Initiate re-engagement campaign"
  }
}
```

---

## 🔄 Adding Your Own Domain

### Option 1: Use Setup Script

```python
from setup_domains import create_domain
from your_data_generator import generate_your_data

create_domain(
    domain_name='your_domain',
    display_name='Your Domain Name',
    description='What it predicts',
    icon='🎯',
    use_case='How NGOs will use it',
    data_generator=generate_your_data,
    target_column='your_target',
    target_description='0 = X, 1 = Y',
    sample_inputs=[...],
    recommendations={...}
)
```

### Option 2: Manual Setup

1. Create folder: `domains/your_domain/`
2. Add your data: `example_data.csv`
3. Train model:
   ```python
   from setup_domains import train_domain_model
   trained = train_domain_model(df, 'target_column', 'your_domain')
   ```
4. Save artifacts (model.pkl, scaler.pkl, features.pkl)
5. Create metadata.json

---

## 🎨 UI Integration (Future)

The enhanced UI will have:

```
┌─────────────────────────────────────────┐
│  Select Your Use Case                   │
│                                         │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌────┐ │
│  │  💰  │  │  🎓  │  │  📋  │  │ 👥 │ │
│  │Donor │  │Prog. │  │Grant │  │Churn│ │
│  └──────┘  └──────┘  └──────┘  └────┘ │
│                                         │
│  [Try Demo Data]  [Upload Your Data]   │
└─────────────────────────────────────────┘
```

---

## ✅ Benefits for NGOs

### Before (Single Model):
```
1. Upload CSV
2. Hope the model understands your data
3. Get generic predictions
4. Unclear what to do with results
```

### After (Multi-Domain):
```
1. Choose "Donor Retention"
2. See demo predictions instantly
3. Understand how it works
4. Upload your donor data
5. Get donor-specific predictions
6. Get actionable recommendations
```

---

## 📊 Performance

All pre-trained models achieve:
- **70-80% accuracy** on test data
- **< 2 second** prediction time
- **< 1 second** domain switching
- **100% private** (runs locally)

---

## 🔐 Privacy

All features of multi-domain system maintain privacy:
- ✅ Domains stored locally
- ✅ Models run on your computer
- ✅ Demo data is synthetic (fake)
- ✅ No internet connection needed
- ✅ Your uploaded data stays local

---

## 🐛 Troubleshooting

### "Domains directory not found"
Run: `python setup_domains.py`

### "Domain not ready"
Check that domain folder has:
- model.pkl
- scaler.pkl
- features.pkl
- metadata.json

### "Error loading domain"
Check metadata.json format and all files exist

---

## 📚 Next Steps

1. **Setup domains:** `python setup_domains.py`
2. **Test domains:** `python domain_manager.py`
3. **Try predictions:** See code examples above
4. **Read architecture:** MULTI_DOMAIN_ARCHITECTURE.md
5. **Integrate with API:** (Coming soon)

---

## 🎯 Summary

```
✅ 4 pre-configured domains
✅ Synthetic demo data (1000+ rows each)
✅ Pre-trained models (70-80% accuracy)
✅ Domain-specific recommendations
✅ Easy to add new domains
✅ 100% local and private
✅ Ready to use immediately!
```

**Setup time:** 10 minutes
**Usage:** Instant predictions
**Privacy:** Complete
**Cost:** Free forever

🚀 **Ready to democratize ML for social good!**
