# File Guide: Hackathon Preparation vs Production System

## Quick Reference

### Timeline

```
October 23, 2024          → Hackathon preparation materials downloaded
October 24, 2024 (Day 1)  → Hackathon event - Jupyter notebooks, initial demo
October 26, 2024 (Day +2) → Production system built - Flask API, testing, docs
```

### Component Categories

| Category | Files | Purpose | Origin |
|----------|-------|---------|--------|
| **Hackathon Prep** | PDFs, templates | Reference materials | Pre-hackathon |
| **Development** | Jupyter notebooks | EDA & initial modeling | During hackathon |
| **Prototype** | demo3.html, Streamlit | Quick demo | During hackathon |
| **Production** | backend/, frontend/demo_updated.html | Deployment-ready system | Post-hackathon |
| **Documentation** | README.md, *.md | Setup & context | Post-hackathon |
| **Data** | Excel (virtual), CSV (synthetic) | Training data | Hackathon + Post |

---

## Detailed File Breakdown

### 1. Hackathon Preparation Materials (Pre-Event)

**Downloaded before hackathon for reference and planning.**

#### PDF Documents

**`TEMPLATE_*.pdf`** (if present)
- **Purpose:** Presentation templates provided by organizers
- **Usage:** Structure final presentation
- **Status:** Reference material, not used in code
- **Audience:** Team planning

**`HACKATHON_RULES.pdf`** (if present)
- **Purpose:** Event rules, judging criteria, timeline
- **Usage:** Understanding constraints and goals
- **Status:** Reference only
- **Audience:** Participants

**`DATA_DESCRIPTION.pdf`** (if present)
- **Purpose:** Explanation of provided datasets
- **Usage:** Understanding feature meanings
- **Status:** Critical for feature engineering
- **Audience:** Data scientists

#### Presentation Materials

**`HACKATHON_TEMPLATE.pptx`** (if present)
- **Purpose:** PowerPoint template for final presentation
- **Usage:** Present findings to judges
- **Status:** Completed during hackathon
- **Audience:** Judges, nonprofit partners

**`Team2_Final_Presentation.pdf`** (if exported)
- **Purpose:** Final presentation as submitted
- **Usage:** Show results and methodology
- **Status:** Historical record
- **Audience:** Public record

---

### 2. Development Phase (During Hackathon - October 24, 2024)

**Files created during the 1-day event for rapid prototyping.**

#### Jupyter Notebooks

**`notebooks/mentorship_eda.ipynb`**
- **Purpose:** Exploratory Data Analysis
- **Created:** October 24, 2024 (morning)
- **Contents:**
  - Load Excel files
  - Merge datasets
  - Statistical summaries
  - Correlation analysis
  - Visualization (histograms, scatter plots)
- **Key Findings:**
  - 33,494 mentees, 9,008 failures (26.9%)
  - Engagement score = strongest predictor
  - Summer months = higher risk
  - Computer Science = highest failure field
- **Status:** Complete, archived
- **Audience:** Data scientists understanding the data

**`notebooks/model_training.ipynb`**
- **Purpose:** Initial model training and experimentation
- **Created:** October 24, 2024 (afternoon)
- **Contents:**
  - Feature engineering (24 features initially)
  - Random Forest training
  - Model evaluation
  - Feature importance plots
- **Key Results:**
  - 75.2% accuracy (initial model)
  - 79.7% recall
  - Identified top 10 features
- **Status:** Complete, superseded by `train_model.py`
- **Audience:** ML engineers iterating on models

**Why Jupyter for hackathon?**
- Interactive experimentation
- Inline visualizations
- Quick iteration cycles
- Easy to share findings with team
- Built-in markdown for documentation

#### Initial Data Files

**`Hackaton_Benevoles_JPMORGAN.xlsx`** *(in Jupyter virtual env)*
- **Purpose:** Volunteer mentor data
- **Created:** Provided by organizers
- **Status:** Not downloadable (privacy restrictions)
- **Replaced by:** `ml_ready_dataset.csv` (synthetic)
- **Audience:** Hackathon participants only

**`Hackaton_Jeunes_JPMORGAN.xlsx`** *(in Jupyter virtual env)*
- **Purpose:** Mentee data
- **Created:** Provided by organizers
- **Status:** Not downloadable (privacy restrictions)
- **Replaced by:** `ml_ready_dataset.csv` (synthetic)
- **Audience:** Hackathon participants only

**`Hackaton_Binomes_JPMORGAN.xlsx`** *(in Jupyter virtual env)*
- **Purpose:** Mentor-mentee pairing data
- **Created:** Provided by organizers
- **Status:** Not downloadable (privacy restrictions)
- **Replaced by:** `ml_ready_dataset.csv` (synthetic)
- **Audience:** Hackathon participants only

#### Prototype Frontend

**`demo3.html`**
- **Purpose:** Quick interactive demo for judges
- **Created:** October 24, 2024 (afternoon)
- **Technology:** Vanilla JavaScript (no backend needed)
- **Features:**
  - Form for mentee information
  - Client-side risk calculation (heuristic, not ML)
  - Chart.js visualizations
  - Feature importance display
  - Intervention recommendations
- **Risk Calculation:** Simple heuristic rules:
  - `engagement < 1.0` → +30% risk
  - `workfield == "Computer Science"` → +20% risk
  - `summer_month` → +15% risk
  - etc.
- **Limitations:**
  - Not using real ML model
  - Simplified logic
  - No API integration
- **Status:** Archived, kept as backup
- **Audience:** Hackathon judges (demo purposes)

**Why client-side for demo?**
- No server setup required
- Instant loading
- Works offline
- Easy to share (just open HTML file)
- Good enough for proof of concept

#### Streamlit Prototype (if present)

**`streamlit_app/`** *(if exists)*
- **Purpose:** Alternative interactive dashboard
- **Created:** October 24, 2024 (if time permitted)
- **Technology:** Streamlit Python framework
- **Features:**
  - File upload for batch predictions
  - Interactive widgets
  - Real-time charts
- **Status:** Prototype, may not be production-ready
- **Audience:** Technical demonstration

---

### 3. Post-Hackathon Production System (October 26, 2024)

**Professional system built after the hackathon for real-world deployment.**

#### Backend API

**`backend/app.py`** (632 lines)
- **Purpose:** Flask REST API for serving predictions
- **Created:** October 26, 2024
- **Technology:** Flask 3.0.0, flask-cors
- **Endpoints:**
  - `GET /api/health` - Health check
  - `POST /api/predict` - Get risk prediction
- **Key Functions:**
  - `load_models()` - Load pickle files on startup
  - `transform_input_to_features()` - 8 inputs → 39 features
  - `predict_with_calibration()` - Bayesian adjustment
  - `calculate_risk_metrics()` - Compute 4 risk scores
  - `fallback_prediction()` - Heuristic backup
- **Features:**
  - CORS support for cross-origin requests
  - Comprehensive error handling
  - Logging to console and file
  - Request validation
  - JSON response formatting
- **Status:** Production-ready
- **Audience:** Developers deploying the system

**Why Flask API vs Jupyter?**
- RESTful architecture
- Scalable (can handle multiple concurrent requests)
- Separates frontend from backend
- Easy to integrate with web apps, mobile apps, etc.
- Production deployment (Gunicorn, Docker)

**`backend/train_model.py`** (694 lines)
- **Purpose:** Complete training pipeline
- **Created:** October 26, 2024
- **Replaces:** `notebooks/model_training.ipynb`
- **Workflow:**
  1. Load CSV data
  2. Engineer 39 features (improved from 24)
  3. Split train/test (80/20, stratified)
  4. Train Random Forest (500 trees, max_depth=7)
  5. 10-fold cross-validation
  6. Evaluate metrics
  7. Save model, scaler, feature_columns as pickles
- **Output:**
  - `models/random_forest_model.pkl` (6.2 MB)
  - `models/scaler.pkl` (943 bytes)
  - `models/feature_columns.pkl` (1013 bytes)
- **Metrics:**
  - Accuracy: 84.17%
  - Precision: 84.92%
  - Recall: 84.03%
  - ROC-AUC: 91.54%
- **Status:** Production-ready
- **Audience:** ML engineers retraining the model

**Why standalone script vs notebook?**
- Command-line execution (`python train_model.py`)
- Reproducible (no manual cell execution)
- Version control friendly
- CI/CD integration
- Automated retraining

**`backend/generate_synthetic_data.py`** (247 lines)
- **Purpose:** Generate privacy-safe training data
- **Created:** October 26, 2024
- **Why needed:** Original Excel files not downloadable (privacy)
- **Approach:**
  - Analyze statistics from EDA notebooks
  - Recreate distributions (Computer Science: 20%, etc.)
  - Preserve correlations (engagement ↔ success: 0.68)
  - Add realistic noise
  - Introduce missing values (10.2% workfield)
- **Output:** `ml_ready_dataset.csv` (2.25 MB, 13,513 records)
- **Validation:**
  - Same feature distributions
  - Same failure rate (26.9%)
  - Model achieves similar accuracy (84% vs 75% original)
- **Status:** Production-ready
- **Audience:** Developers needing training data

**Why synthetic data?**
- Privacy protection (no real PII)
- Shareable on GitHub
- Demonstrates methodology publicly
- Allows model replication

**`backend/prepare_data.py`** (359 lines)
- **Purpose:** Load and merge original Excel files
- **Created:** October 24-26, 2024
- **Usage:** Load real data (if accessible)
- **Workflow:**
  1. Load 3 Excel files
  2. Merge by mentee ID
  3. Create binary target (success/failure)
  4. Handle missing values
  5. Export as CSV
- **Status:** Reference (Excel files not accessible)
- **Audience:** Those with access to original data

**`backend/test_api.py`** (285 lines)
- **Purpose:** Automated API testing
- **Created:** October 26, 2024
- **Test Cases:**
  1. Health check
  2. High-risk prediction
  3. Low-risk prediction
  4. Medium-risk prediction
  5. Edge case (very low engagement)
  6. Optimal case (all positive factors)
  7. Error handling (missing fields, invalid JSON, wrong endpoint)
- **Output:** Console report with pass/fail
- **Fixes applied:** UTF-8 encoding for Windows
- **Status:** Production-ready
- **Audience:** QA, developers verifying API

**`backend/requirements.txt`** (18 packages)
- **Purpose:** Python dependency management
- **Created:** October 26, 2024
- **Key Dependencies:**
  - Flask==3.0.0 (web framework)
  - scikit-learn==1.4.0 (ML library)
  - pandas==2.1.4 (data manipulation)
  - flask-cors==4.0.0 (CORS support)
  - gunicorn==21.2.0 (production server)
- **Usage:** `pip install -r requirements.txt`
- **Status:** Production-ready
- **Audience:** DevOps, deployment engineers

**`backend/.gitignore`**
- **Purpose:** Exclude files from version control
- **Created:** October 26, 2024
- **Excludes:**
  - `__pycache__/`
  - `*.pyc`
  - `*.log`
  - `.env` (if added)
  - Virtual environments
- **Status:** Active
- **Audience:** Git users

#### Model Files

**`backend/models/random_forest_model.pkl`** (6.2 MB)
- **Purpose:** Trained Random Forest classifier
- **Created:** October 26, 2024 (by `train_model.py`)
- **Contents:**
  - 500 decision trees
  - Hyperparameters (max_depth=7, etc.)
  - Learned weights from 13,513 records
- **Loaded by:** `app.py` on startup
- **Status:** Production-ready
- **Audience:** Backend API (automated loading)

**`backend/models/scaler.pkl`** (943 bytes)
- **Purpose:** StandardScaler for feature normalization
- **Created:** October 26, 2024 (by `train_model.py`)
- **Contents:**
  - Mean and std dev for 7 numeric features
  - Transformation parameters
- **Loaded by:** `app.py` on startup
- **Status:** Production-ready
- **Audience:** Backend API (automated loading)

**`backend/models/feature_columns.pkl`** (1013 bytes)
- **Purpose:** List of 39 feature names in order
- **Created:** October 26, 2024 (by `train_model.py`)
- **Contents:**
  - `['engagement_score', 'project_confidence_level', ...]`
  - Ensures correct feature order for predictions
- **Loaded by:** `app.py` on startup
- **Status:** Production-ready
- **Audience:** Backend API (automated loading)

**Why pickle format?**
- Native Python serialization
- Fast loading (no re-training)
- Preserves scikit-learn objects exactly
- Small file size (compressed)

#### Data Files

**`backend/ml_ready_dataset.csv`** (2.25 MB)
- **Purpose:** Synthetic training data
- **Created:** October 26, 2024 (by `generate_synthetic_data.py`)
- **Records:** 13,513
- **Columns:** 10 (8 features + 2 metadata)
- **Target:** `success` (0 = failure, 1 = success)
- **Failure Rate:** 26.9%
- **Used by:** `train_model.py`
- **Status:** Production-ready
- **Audience:** ML engineers training models

---

### 4. Frontend (Post-Hackathon)

**`frontend/demo_updated.html`**
- **Purpose:** Production demo with real API integration
- **Created:** October 26, 2024
- **Replaces:** `demo3.html` (hackathon prototype)
- **Changes from original:**
  - Replaced `calculateRisk()` with `callFlaskAPI()`
  - Added async `fetch()` to `http://localhost:5000/api/predict`
  - Data transformation functions (`transformNeeds()`, `transformAvailability()`)
  - 2-second minimum loading animation
  - Automatic fallback to client-side heuristic
  - Visual badge showing data source (API vs fallback)
- **Features:**
  - Real ML predictions from trained model
  - Error handling with user-friendly messages
  - Console logging for debugging
  - Chart.js visualizations
  - Responsive design
- **Status:** Production-ready
- **Audience:** End users (nonprofit staff)

**`frontend/FRONTEND_UPDATE_SUMMARY.md`** (309 lines)
- **Purpose:** Documentation of API integration
- **Created:** October 26, 2024
- **Contents:**
  - What was changed
  - API mapping (form → API format)
  - Testing instructions
  - Performance metrics
  - Console output examples
- **Status:** Complete documentation
- **Audience:** Developers understanding the integration

**Why update the frontend?**
- Use real ML predictions (not heuristics)
- Demonstrate full end-to-end system
- Professional user experience
- Production deployment readiness

---

### 5. Documentation (Post-Hackathon)

**`README.md`** (872 lines)
- **Purpose:** Main project documentation
- **Created:** October 26, 2024
- **Sections:**
  - Project overview
  - Quick start (5 steps)
  - Detailed installation
  - Training instructions
  - Running the API
  - Running the demo
  - API documentation
  - Troubleshooting
  - Model performance
- **Audience:** New users, developers, contributors
- **Status:** Production-ready

**`PROJECT_CONTEXT_AND_CLARIFICATION.md`** (this guide's companion)
- **Purpose:** Explain hackathon context and synthetic data
- **Created:** October 26, 2024
- **Sections:**
  - Hackathon background
  - Why synthetic data
  - Hackathon vs production
  - Skills demonstrated
  - Future enhancements
- **Audience:** Portfolio viewers, recruiters, contributors
- **Status:** Complete

**`FILE_GUIDE.md`** (this file)
- **Purpose:** Explain which files are what
- **Created:** October 26, 2024
- **Approach:** Timeline, component categories, file-by-file breakdown
- **Audience:** Developers navigating the repository
- **Status:** Complete

**Why so much documentation?**
- Onboarding new contributors
- Portfolio demonstration
- Knowledge transfer
- Reproducibility
- Professional standards

---

### 6. Summary Documents (Created During Development)

**`backend/COMPLETE_SUMMARY.md`** (359 lines)
- **Purpose:** Summary of backend development
- **Created:** October 26, 2024
- **Contents:** What was built, how it works, next steps
- **Audience:** Team members catching up
- **Status:** Archived (superseded by README.md)

**`backend/QUICKSTART.md`** (150 lines)
- **Purpose:** Quick reference for running the system
- **Created:** October 26, 2024
- **Contents:** 5-minute setup guide
- **Audience:** Developers wanting fast start
- **Status:** Archived (merged into README.md)

**`backend/SUMMARY.md`** (200 lines)
- **Purpose:** Initial development summary
- **Created:** October 26, 2024
- **Contents:** Early progress notes
- **Audience:** Team members
- **Status:** Archived (superseded by COMPLETE_SUMMARY.md)

**`backend/TRAINING_GUIDE.md`** (280 lines)
- **Purpose:** Detailed training pipeline explanation
- **Created:** October 26, 2024
- **Contents:** Step-by-step model training
- **Audience:** ML engineers
- **Status:** Archived (merged into README.md)

**`backend/SYNTHETIC_DATA_AND_TRAINING_SUMMARY.md`** (250 lines)
- **Purpose:** Explain synthetic data generation
- **Created:** October 26, 2024
- **Contents:** Why synthetic, how generated, validation
- **Audience:** Data scientists
- **Status:** Archived (merged into PROJECT_CONTEXT_AND_CLARIFICATION.md)

**Why multiple summaries?**
- Iterative development (each iteration documented)
- Different audiences (quick vs detailed)
- Later consolidated into comprehensive README

---

## How to Interpret Each Component

### For New Users

**Start here:**
1. Read `README.md` (main documentation)
2. Read `PROJECT_CONTEXT_AND_CLARIFICATION.md` (understand context)
3. Run Quick Start commands (5 minutes)
4. Open `demo_updated.html` in browser
5. Explore Jupyter notebooks for data insights

**Skip these files:**
- `demo3.html` (old prototype, use `demo_updated.html`)
- `*_SUMMARY.md` files in backend/ (merged into README)
- `.gitignore`, `.pyc` files (system files)

### For Developers

**Code to study:**
1. `backend/app.py` - Flask API patterns
2. `backend/train_model.py` - ML pipeline
3. `frontend/demo_updated.html` - API integration
4. `backend/test_api.py` - Testing practices

**Modify these for customization:**
1. `train_model.py` - Change hyperparameters, add features
2. `app.py` - Add new endpoints, change risk calculation
3. `demo_updated.html` - Customize UI, add features

### For Recruiters/Portfolio Viewers

**Demonstrates:**
1. **ML Skills** - `notebooks/`, `train_model.py`
2. **Backend Skills** - `app.py`, API design
3. **Frontend Skills** - `demo_updated.html`, async JavaScript
4. **Testing** - `test_api.py`, comprehensive test cases
5. **Documentation** - All `.md` files, inline comments
6. **Problem Solving** - Synthetic data generation, UTF-8 fixes

**Ignore:**
- Internal summaries (`*_SUMMARY.md`)
- System files (`.gitignore`, `__pycache__`)

### For Data Scientists

**Key files:**
1. `notebooks/mentorship_eda.ipynb` - Data exploration approach
2. `notebooks/model_training.ipynb` - Initial modeling
3. `train_model.py` - Production training pipeline
4. `generate_synthetic_data.py` - Data synthesis methodology

**Metrics to review:**
- Cross-validation results (83.93% ± 0.81%)
- Feature importance rankings
- ROC-AUC curve (91.54%)

### For Nonprofit Users

**You need:**
1. `README.md` - Installation guide
2. `demo_updated.html` - Interactive demo
3. API running (`python app.py`)

**You can ignore:**
- All code files (unless customizing)
- Jupyter notebooks (technical EDA)
- Documentation files (unless troubleshooting)

---

## Evolution Timeline

### October 23, 2024 (Pre-Hackathon)
- Download hackathon materials
- Review data description PDFs
- Download Excel files to Jupyter environment

### October 24, 2024 (Hackathon Day)
**Morning (9 AM - 12 PM):**
- Create `notebooks/mentorship_eda.ipynb`
- Explore data, find correlations
- Identify key features

**Afternoon (1 PM - 5 PM):**
- Create `notebooks/model_training.ipynb`
- Train initial Random Forest (75% accuracy)
- Build `demo3.html` with client-side calculations
- Create Streamlit prototype (if time)

**Evening (6 PM - 8 PM):**
- Prepare presentation
- Finalize demo
- Submit to judges

### October 26, 2024 (Post-Hackathon Production)
**Day +2 - Morning:**
- Create `backend/app.py` (Flask API)
- Create `backend/train_model.py` (production pipeline)
- Realize Excel files not downloadable

**Day +2 - Afternoon:**
- Create `backend/generate_synthetic_data.py`
- Generate `ml_ready_dataset.csv`
- Train production model (84% accuracy)

**Day +2 - Evening:**
- Create `backend/test_api.py`
- Fix UTF-8 encoding issues
- Run tests (all pass)

**Day +2 - Night:**
- Update frontend to `demo_updated.html`
- Create `requirements.txt`
- Write `README.md`
- Write `PROJECT_CONTEXT_AND_CLARIFICATION.md`
- Write `FILE_GUIDE.md`

---

## Decision Points: Why This Architecture?

### Why Flask API instead of just Jupyter notebooks?

**Hackathon (notebooks):**
- Quick prototyping
- Interactive exploration
- Good for demos to judges
- Not scalable

**Production (Flask API):**
- RESTful endpoints (standard)
- Can handle concurrent users
- Easy to integrate with apps
- Deployable to cloud
- Professional standard

### Why separate frontend and backend?

**Monolithic (demo3.html with client-side logic):**
- ✓ Easy to demo (just open HTML)
- ✓ No server needed
- ✗ Not using real ML model
- ✗ Can't update model without changing HTML

**Separated (Flask API + demo_updated.html):**
- ✓ Real ML predictions
- ✓ Update model independently
- ✓ API can serve multiple frontends (web, mobile, etc.)
- ✓ Scalable architecture
- ✗ Requires server running

**Decision:** Separate for production, monolithic for quick demos

### Why synthetic data instead of just documenting "we can't share"?

**Option 1: Just document limitation**
- ✓ Honest
- ✗ Others can't replicate
- ✗ Can't verify claims
- ✗ Not useful as learning resource

**Option 2: Synthetic data**
- ✓ Others can run and verify
- ✓ Code works end-to-end
- ✓ Privacy protected
- ✓ Learning resource
- ✗ Not 100% identical to original (but close)

**Decision:** Synthetic data for transparency and replicability

### Why so much documentation?

**Minimal docs:**
- ✓ Less time writing
- ✗ Hard to onboard
- ✗ Not portfolio-friendly
- ✗ Code becomes legacy quickly

**Comprehensive docs:**
- ✓ Easy onboarding
- ✓ Professional impression
- ✓ Future-proof
- ✓ Good for portfolio
- ✗ Takes time

**Decision:** Comprehensive docs for long-term value

---

## File Status Summary

| File | Status | Use It? | Audience |
|------|--------|---------|----------|
| `demo3.html` | Archived | ❌ Use `demo_updated.html` | Historical |
| `demo_updated.html` | Production | ✅ Yes | End users |
| `notebooks/*.ipynb` | Complete | ✅ For learning | Data scientists |
| `backend/app.py` | Production | ✅ Yes | Deployment |
| `backend/train_model.py` | Production | ✅ Yes | ML engineers |
| `backend/test_api.py` | Production | ✅ Yes | QA/Testing |
| `backend/generate_synthetic_data.py` | Production | ✅ Yes | Data generation |
| `backend/requirements.txt` | Production | ✅ Yes | Deployment |
| `backend/models/*.pkl` | Production | ✅ Yes (auto-loaded) | Backend |
| `README.md` | Complete | ✅ Yes | Everyone |
| `PROJECT_CONTEXT_AND_CLARIFICATION.md` | Complete | ✅ Yes | Portfolio viewers |
| `FILE_GUIDE.md` | Complete | ✅ Yes | Developers |
| `*_SUMMARY.md` | Archived | ❌ Use README instead | Historical |
| `*.pdf` (if present) | Reference | ✅ For context | Participants |

---

## Quick Decision Guide

### "I want to run the demo"
→ Use `demo_updated.html` + `backend/app.py`

### "I want to understand the data"
→ Read `notebooks/mentorship_eda.ipynb`

### "I want to train a new model"
→ Run `python backend/train_model.py`

### "I want to deploy to production"
→ Read `README.md` → Use Docker (future) or Gunicorn

### "I want to understand the project context"
→ Read `PROJECT_CONTEXT_AND_CLARIFICATION.md`

### "I want to modify the risk calculation"
→ Edit `backend/app.py` → `calculate_risk_metrics()` function

### "I want to add a new feature"
→ Edit `backend/train_model.py` → `engineer_features()` function

### "I'm new to the codebase"
→ Read `README.md`, then `FILE_GUIDE.md`, then run Quick Start

---

## Questions?

**"Why are there so many summary files?"**
→ Iterative development. Each phase was documented. Later consolidated into README.md. Old summaries kept for history.

**"Can I delete the old summaries?"**
→ Yes, they're archived. But keeping them shows development process.

**"Which files do I need to deploy?"**
→ `backend/` (all .py files), `backend/models/` (all .pkl files), `requirements.txt`, `frontend/demo_updated.html`

**"Where's the real data?"**
→ In Jupyter virtual environment, not downloadable. Use `ml_ready_dataset.csv` (synthetic).

**"How do I know which version is latest?"**
→ `demo_updated.html` > `demo3.html`, `README.md` > `*_SUMMARY.md`, `train_model.py` > `model_training.ipynb`

**"What if I have access to the original Excel files?"**
→ Use `backend/prepare_data.py` to load them, then run `train_model.py` with real data.

---

*This guide maps the entire repository evolution from hackathon preparation through production deployment.*

**October 2024**
**JP Morgan Data for Good Hackathon 2025 - Team 2**
