# Multi-Domain ML System Architecture

## 🎯 Vision

Transform Data Insight ML into a **multi-domain platform** where NGOs can:
1. Choose their use case (donor retention, grants, programs, etc.)
2. Try demo data instantly (no setup needed)
3. See pre-trained model predictions
4. Upload their own data for that domain
5. Get domain-specific insights

---

## 🏗️ Architecture Design

### Current (Single Model)
```
data_insight_ml/
├── models/
│   ├── best_model.pkl (one model for everything)
│   ├── scaler.pkl
│   └── features.pkl
├── app.py
└── demo.html
```

**Limitations:**
- One model only
- Generic features
- No domain context
- Users must train from scratch

### Enhanced (Multi-Domain)
```
data_insight_ml/
├── domains/
│   ├── donor_retention/
│   │   ├── example_data.csv (1000 rows)
│   │   ├── model.pkl (pre-trained)
│   │   ├── scaler.pkl
│   │   ├── features.pkl
│   │   ├── config.yaml (domain-specific)
│   │   └── metadata.json
│   ├── program_completion/
│   │   ├── example_data.csv
│   │   ├── model.pkl
│   │   └── ...
│   ├── grant_scoring/
│   ├── customer_churn/
│   └── loan_default/
├── domain_manager.py (new)
├── app_multi_domain.py (enhanced)
└── demo_multi_domain.html (enhanced)
```

**Benefits:**
✅ Multiple pre-trained models
✅ Domain-specific features
✅ Instant demo capability
✅ Context-aware predictions
✅ Easy domain switching
✅ Example data for learning

---

## 📊 Supported Domains

### 1. Donor Retention
**Use Case:** Predict which donors will give again

**Features:**
- Last donation amount
- Donation frequency
- Years since first donation
- Email engagement
- Age, Region
- Previous campaign response

**Target:** `will_donate_again` (0/1)

**Sample Prediction:**
```json
{
  "prediction": "Will donate",
  "confidence": 78%,
  "recommendation": "Send personalized thank you email"
}
```

### 2. Program Completion
**Use Case:** Predict which participants will complete programs

**Features:**
- Attendance rate
- Engagement score
- Mentor assigned
- Previous program history
- Education level
- Work status

**Target:** `completed_program` (0/1)

**Sample Prediction:**
```json
{
  "prediction": "At risk of dropout",
  "confidence": 82%,
  "recommendation": "Assign additional mentor support"
}
```

### 3. Grant Application Scoring
**Use Case:** Score and prioritize grant applications

**Features:**
- Organization size
- Years operating
- Budget requested
- Previous grants received
- Mission alignment
- Proposal quality score

**Target:** `approved` (0/1)

**Sample Prediction:**
```json
{
  "prediction": "Approve",
  "confidence": 85%,
  "priority": "High",
  "recommendation": "Fast-track for review"
}
```

### 4. Customer/Member Churn
**Use Case:** Predict member retention

**Features:**
- Months as member
- Monthly spending
- Support tickets
- Feature usage
- Referrals made
- Account type

**Target:** `churned` (0/1)

**Sample Prediction:**
```json
{
  "prediction": "High churn risk",
  "confidence": 91%,
  "recommendation": "Offer retention discount"
}
```

### 5. Loan Default Risk
**Use Case:** Assess loan repayment risk (microfinance)

**Features:**
- Loan amount
- Income level
- Credit history
- Collateral value
- Employment status
- Loan purpose

**Target:** `defaulted` (0/1)

**Sample Prediction:**
```json
{
  "prediction": "Low risk",
  "confidence": 73%,
  "recommendation": "Approve with standard terms"
}
```

---

## 🔧 Technical Implementation

### Phase 1: Domain Manager Class

```python
# domain_manager.py

import os
import json
import pickle
import pandas as pd

class DomainManager:
    """Manages multiple ML domains with pre-trained models"""

    def __init__(self, domains_dir='domains'):
        self.domains_dir = domains_dir
        self.available_domains = self.scan_domains()
        self.current_domain = None
        self.current_model = None

    def scan_domains(self):
        """Scan and list all available domains"""
        domains = {}

        for domain_name in os.listdir(self.domains_dir):
            domain_path = os.path.join(self.domains_dir, domain_name)

            if os.path.isdir(domain_path):
                metadata_path = os.path.join(domain_path, 'metadata.json')

                if os.path.exists(metadata_path):
                    with open(metadata_path) as f:
                        metadata = json.load(f)

                    domains[domain_name] = {
                        'path': domain_path,
                        'metadata': metadata,
                        'has_model': os.path.exists(
                            os.path.join(domain_path, 'model.pkl')
                        ),
                        'has_data': os.path.exists(
                            os.path.join(domain_path, 'example_data.csv')
                        )
                    }

        return domains

    def list_domains(self):
        """Return list of available domains"""
        return [
            {
                'name': name,
                'display_name': info['metadata']['display_name'],
                'description': info['metadata']['description'],
                'use_case': info['metadata']['use_case'],
                'ready': info['has_model'] and info['has_data']
            }
            for name, info in self.available_domains.items()
        ]

    def load_domain(self, domain_name):
        """Load a specific domain's model and config"""
        if domain_name not in self.available_domains:
            raise ValueError(f"Domain '{domain_name}' not found")

        domain_info = self.available_domains[domain_name]
        domain_path = domain_info['path']

        # Load model
        model_path = os.path.join(domain_path, 'model.pkl')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Load scaler
        scaler_path = os.path.join(domain_path, 'scaler.pkl')
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

        # Load features
        features_path = os.path.join(domain_path, 'features.pkl')
        with open(features_path, 'rb') as f:
            features = pickle.load(f)

        self.current_domain = domain_name
        self.current_model = {
            'model': model,
            'scaler': scaler,
            'features': features,
            'metadata': domain_info['metadata']
        }

        return self.current_model

    def get_example_data(self, domain_name, n_rows=10):
        """Get example data for a domain"""
        domain_path = self.available_domains[domain_name]['path']
        data_path = os.path.join(domain_path, 'example_data.csv')

        df = pd.read_csv(data_path)
        return df.head(n_rows).to_dict('records')

    def predict(self, domain_name, input_data):
        """Make prediction using domain's model"""
        if self.current_domain != domain_name:
            self.load_domain(domain_name)

        # Preprocess and predict
        # ... (similar to existing predict logic)

        return prediction
```

### Phase 2: Enhanced API Endpoints

```python
# app_multi_domain.py

from domain_manager import DomainManager

domain_manager = DomainManager()

@app.route('/api/domains', methods=['GET'])
def list_domains():
    """List all available domains"""
    domains = domain_manager.list_domains()
    return jsonify({
        'success': True,
        'domains': domains
    })

@app.route('/api/domain/<domain_name>/load', methods=['POST'])
def load_domain(domain_name):
    """Load a specific domain"""
    try:
        model_info = domain_manager.load_domain(domain_name)
        return jsonify({
            'success': True,
            'domain': domain_name,
            'features': model_info['features'],
            'metadata': model_info['metadata']
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/domain/<domain_name>/example', methods=['GET'])
def get_domain_example(domain_name):
    """Get example data for domain"""
    n_rows = request.args.get('rows', 10, type=int)
    examples = domain_manager.get_example_data(domain_name, n_rows)

    return jsonify({
        'success': True,
        'domain': domain_name,
        'examples': examples
    })

@app.route('/api/domain/<domain_name>/predict', methods=['POST'])
def predict_domain(domain_name):
    """Make prediction using domain model"""
    input_data = request.get_json()

    prediction = domain_manager.predict(domain_name, input_data)

    return jsonify({
        'success': True,
        'domain': domain_name,
        'prediction': prediction
    })
```

### Phase 3: Enhanced UI

```html
<!-- Domain Selector -->
<div class="card">
    <h2>Select Your Use Case</h2>

    <div class="domain-grid">
        <!-- Donor Retention -->
        <div class="domain-card" onclick="selectDomain('donor_retention')">
            <div class="domain-icon">💰</div>
            <h3>Donor Retention</h3>
            <p>Predict which donors will give again</p>
            <span class="badge">Ready</span>
        </div>

        <!-- Program Completion -->
        <div class="domain-card" onclick="selectDomain('program_completion')">
            <div class="domain-icon">🎓</div>
            <h3>Program Completion</h3>
            <p>Identify at-risk participants</p>
            <span class="badge">Ready</span>
        </div>

        <!-- Grant Scoring -->
        <div class="domain-card" onclick="selectDomain('grant_scoring')">
            <div class="domain-icon">📋</div>
            <h3>Grant Scoring</h3>
            <p>Score and prioritize applications</p>
            <span class="badge">Ready</span>
        </div>

        <!-- Member Churn -->
        <div class="domain-card" onclick="selectDomain('customer_churn')">
            <div class="domain-icon">👥</div>
            <h3>Member Churn</h3>
            <p>Predict member retention</p>
            <span class="badge">Ready</span>
        </div>
    </div>
</div>

<!-- Demo Data Section -->
<div class="card" id="demoSection" style="display:none;">
    <h2>Try Demo Data</h2>

    <p>See how the model works with example data</p>

    <button class="btn" onclick="loadDemoData()">
        Load Example Predictions
    </button>

    <div id="demoResults"></div>
</div>
```

---

## 📁 Domain Directory Structure

### Example: donor_retention/

```
donor_retention/
├── example_data.csv           # 1000 synthetic donors
├── model.pkl                  # Pre-trained Random Forest
├── scaler.pkl                 # StandardScaler fitted
├── features.pkl               # Feature list (31 features)
├── config.yaml                # Domain-specific config
├── metadata.json              # Domain information
└── README.md                  # Domain documentation
```

**metadata.json:**
```json
{
  "domain": "donor_retention",
  "display_name": "Donor Retention Prediction",
  "description": "Predict which donors are likely to donate again",
  "icon": "💰",
  "use_case": "Optimize fundraising by focusing on high-probability donors",
  "target_variable": "donated_again",
  "target_description": "1 = Will donate, 0 = Won't donate",
  "features_count": 31,
  "example_rows": 1000,
  "model_type": "Random Forest",
  "accuracy": 0.77,
  "precision": 0.75,
  "recall": 0.79,
  "f1_score": 0.77,
  "training_date": "2025-10-28",
  "recommendations": {
    "high_confidence_threshold": 0.8,
    "action_positive": "Prioritize for next campaign",
    "action_negative": "Re-engagement campaign needed"
  },
  "sample_inputs": [
    {
      "name": "High-value donor",
      "last_donation_amount": 250,
      "donation_frequency": 12,
      "years_since_first": 5,
      "email_opens": 18,
      "age": 55,
      "expected_prediction": "Will donate",
      "expected_confidence": 0.85
    },
    {
      "name": "At-risk donor",
      "last_donation_amount": 25,
      "donation_frequency": 1,
      "years_since_first": 0.5,
      "email_opens": 2,
      "age": 28,
      "expected_prediction": "Won't donate",
      "expected_confidence": 0.82
    }
  ]
}
```

---

## 🎨 Enhanced UI Workflow

### 1. Domain Selection Screen
```
┌────────────────────────────────────────┐
│  Select Your Use Case                  │
│                                        │
│  ┌──────┐  ┌──────┐  ┌──────┐        │
│  │  💰  │  │  🎓  │  │  📋  │        │
│  │Donor │  │Prog. │  │Grant │        │
│  │Reten.│  │Comp. │  │Score │        │
│  └──────┘  └──────┘  └──────┘        │
│                                        │
│  Or upload your own data...            │
│  [Upload CSV]                          │
└────────────────────────────────────────┘
```

### 2. Domain Loaded Screen
```
┌────────────────────────────────────────┐
│  💰 Donor Retention                    │
│  Predict which donors will give again  │
│                                        │
│  Model: Random Forest | Accuracy: 77%  │
│                                        │
│  [Try Demo Data] [Upload Your Data]   │
└────────────────────────────────────────┘
```

### 3. Demo Data Results
```
┌────────────────────────────────────────┐
│  Example Predictions                   │
│                                        │
│  High-value donor:                     │
│  ✓ Will donate (85% confidence)       │
│  → Prioritize for next campaign        │
│                                        │
│  At-risk donor:                        │
│  ✗ Won't donate (82% confidence)      │
│  → Re-engagement campaign needed       │
└────────────────────────────────────────┘
```

---

## 🔄 User Journey

### NGO First-Time User:

```
1. Open Data Insight ML
2. See domain selection screen
3. Click "Donor Retention" (their use case)
4. See domain loaded with 77% accuracy
5. Click "Try Demo Data"
6. See 3-5 example predictions instantly
7. Understand how it works
8. Click "Upload Your Data"
9. Upload their donor CSV
10. Train custom model on their data
11. Get predictions specific to their donors
```

### Power User:

```
1. Open app
2. Select domain from dropdown
3. Upload CSV
4. Auto-detect domain if possible
5. Train model
6. Make predictions
7. Switch to different domain
8. Repeat
```

---

## 💾 Data Generation Strategy

### Synthetic Data Requirements:

Each domain needs **realistic synthetic data** with:
- Proper correlations (features → target)
- Realistic value distributions
- Class imbalance (if relevant)
- 1000+ rows for training
- Clear patterns for ML to learn

### Enhanced example_data_generator.py:

```python
def generate_all_domains():
    """Generate synthetic data for all domains"""

    domains = {
        'donor_retention': generate_donor_data(1000),
        'program_completion': generate_program_data(800),
        'grant_scoring': generate_grant_data(500),
        'customer_churn': generate_churn_data(1000),
        'loan_default': generate_loan_data(600)
    }

    for domain_name, df in domains.items():
        # Save to domain folder
        domain_path = f'domains/{domain_name}'
        os.makedirs(domain_path, exist_ok=True)

        df.to_csv(f'{domain_path}/example_data.csv', index=False)

        # Create metadata
        metadata = create_metadata(domain_name, df)
        with open(f'{domain_path}/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ Generated {domain_name}: {len(df)} rows")
```

---

## 🚀 Implementation Plan

### Week 1: Foundation
- [x] Create domain architecture design
- [ ] Enhance example_data_generator.py
- [ ] Create DomainManager class
- [ ] Generate 5 domain datasets

### Week 2: Backend
- [ ] Add domain API endpoints
- [ ] Pre-train models for each domain
- [ ] Test domain switching
- [ ] Add domain metadata

### Week 3: Frontend
- [ ] Create domain selector UI
- [ ] Add demo data viewer
- [ ] Implement domain switching
- [ ] Add domain-specific help

### Week 4: Polish
- [ ] Create domain documentation
- [ ] Add example predictions
- [ ] Test with NGO users
- [ ] Refine based on feedback

---

## 🎯 Success Criteria

### For NGOs:
✅ Can choose relevant domain in < 5 seconds
✅ See working demo in < 30 seconds
✅ Understand predictions without training
✅ Easy to upload own data
✅ Get domain-specific recommendations

### Technical:
✅ 5+ domains available
✅ < 2 second domain switching
✅ 75%+ accuracy per domain
✅ < 100MB total size
✅ Works offline

---

## 📊 ROI for NGOs

### Before (Generic ML):
- Upload data → ?
- Train model → What features?
- Get prediction → What does it mean?
- Unclear value

### After (Domain-Specific):
- Select "Donor Retention"
- See demo predictions
- Understand immediately
- Upload donor data
- Get actionable insights
- Clear value!

---

## 🔮 Future Enhancements

### Phase 5: Custom Domain Builder
- NGO can create their own domain
- Upload training data
- System auto-generates features
- Save as new domain

### Phase 6: Domain Marketplace
- Community-contributed domains
- Industry-specific models
- Regional variations
- Best practices sharing

### Phase 7: Transfer Learning
- Start with pre-trained domain
- Fine-tune on NGO's data
- Better accuracy with less data
- Faster training

---

## Summary

This multi-domain architecture transforms Data Insight ML from:

**Single Generic Tool** → **Specialized Platform**

With:
- 5+ ready-to-use domains
- Instant demo capability
- Pre-trained models
- Domain expertise built-in
- Actionable recommendations
- Still 100% private and local!

**Next:** Implement enhanced example_data_generator.py and DomainManager class.
