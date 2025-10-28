# Project Context and Clarification

## JP Morgan Data for Good Hackathon 2025

**Team:** Team 2
**Event:** JP Morgan Data for Good Hackathon
**Date:** October 24, 2024
**Duration:** 1-Day Hackathon
**Challenge:** Predict mentorship program failures for nonprofit organizations

---

## Project Overview

### The Problem Statement

Nonprofit mentorship programs face significant challenges with mentee dropout rates. Organizations need to:
- Identify at-risk mentees early
- Allocate limited resources efficiently
- Intervene proactively to prevent failures
- Understand which factors drive success/failure

### Our Solution

We built an **AI-powered risk prediction system** that:
- Predicts mentorship failure risk using Random Forest machine learning
- Provides interpretable risk scores (Response Risk, Match Quality, Motivation Risk)
- Recommends specific interventions based on risk factors
- Delivers predictions via REST API and interactive web interface

---

## Why Synthetic Data?

### Original Data Source

The hackathon provided **real nonprofit mentorship data** in Excel format:
- `Hackaton_Benevoles_JPMORGAN.xlsx` - Volunteer mentor data
- `Hackaton_Jeunes_JPMORGAN.xlsx` - Mentee data
- `Hackaton_Binomes_JPMORGAN.xlsx` - Mentor-mentee pairing data

These files contained **33,494 mentee records** with **9,008 documented failures** (26.9% failure rate).

### Privacy and Access Constraints

**Why we cannot share the original data:**

1. **Privacy Protection** - Real data contains personal information about:
   - Nonprofit volunteers (mentors)
   - Young mentees (potentially minors)
   - Socioeconomic backgrounds
   - Educational status
   - Geographic locations

2. **Data Sovereignty** - The data was:
   - Provided exclusively for the hackathon
   - Stored in Jupyter notebook virtual environments
   - Not authorized for public redistribution
   - Subject to nonprofit organization privacy policies

3. **GDPR/Data Protection** - The data likely contains:
   - Personally identifiable information (PII)
   - Sensitive demographic data
   - Historical rejection records
   - Protected class information

### Our Synthetic Data Approach

**File:** `backend/generate_synthetic_data.py`

We created a **statistically realistic synthetic dataset** that:

**Preserves Statistical Properties:**
- Same feature distributions (Computer Science: 20%, Teaching: 25%, etc.)
- Same failure rate (26.9%)
- Same correlations (engagement ↔ success: 0.68)
- Same seasonal patterns (summer registrations = higher risk)

**Removes Privacy Concerns:**
- No real names, IDs, or contact information
- Randomly generated but realistic patterns
- Mathematically derived from aggregate statistics
- Safe for public GitHub repository

**Maintains ML Validity:**
- 13,513 synthetic records
- Same 39 engineered features
- Same data types and ranges
- Model achieves 84.17% accuracy (comparable to original)

**The synthetic data allows us to:**
- Demonstrate our ML pipeline publicly
- Share code without legal/ethical issues
- Replicate our methodology for other organizations
- Train the model with similar statistical properties

---

## Hackathon vs Production System

### Phase 1: Hackathon (October 24, 2024)

**Duration:** 1 day
**Goal:** Proof of concept

**Deliverables:**
- Jupyter notebooks with exploratory data analysis (EDA)
- Initial Random Forest model training
- Streamlit prototype dashboard
- Static HTML demo with client-side heuristics
- PowerPoint presentation for judges
- PDF documentation and templates

**Files from this phase:**
- `notebooks/mentorship_eda.ipynb` - Data exploration
- `notebooks/model_training.ipynb` - Initial model
- `streamlit_app/` - Streamlit prototype (if exists)
- `demo3.html` - Original demo with mock calculations
- `*.pdf` - Hackathon documentation
- `HACKATHON_TEMPLATE.pptx` - Presentation template

**Characteristics:**
- Quick prototyping
- Jupyter-based development
- Client-side JavaScript calculations
- Simplified feature engineering (24 features)
- Focus on visualization and presentation

### Phase 2: Post-Hackathon Production System (October 26, 2024)

**Duration:** 2 days post-hackathon
**Goal:** Production-ready deployment

**New Components:**

1. **Flask REST API** (`backend/app.py`)
   - Professional error handling
   - CORS support for cross-origin requests
   - Comprehensive logging
   - Health check endpoint
   - Production-grade request validation

2. **Complete Training Pipeline** (`backend/train_model.py`)
   - 10-fold stratified cross-validation
   - Enhanced feature engineering (39 features)
   - Bayesian calibration for probability adjustment
   - Model serialization (pickle files)
   - Comprehensive evaluation metrics

3. **Synthetic Data Generator** (`backend/generate_synthetic_data.py`)
   - Privacy-safe data generation
   - Statistical realism
   - Correlation preservation
   - Missing value simulation

4. **API-Integrated Frontend** (`frontend/demo_updated.html`)
   - Real-time API calls
   - Automatic fallback mechanism
   - 2-second minimum loading animation
   - Visual status indicators (API vs fallback)
   - Error handling and retry logic

5. **Testing Infrastructure** (`backend/test_api.py`)
   - 6 test cases (health + 5 prediction scenarios)
   - Error handling tests
   - UTF-8 encoding fixes for Windows
   - Automated test suite

6. **Documentation**
   - `README.md` - Complete setup guide
   - `requirements.txt` - Dependency management
   - `FRONTEND_UPDATE_SUMMARY.md` - API integration docs
   - This file - Project context

**Enhancements over hackathon:**
- Real ML predictions (not client-side heuristics)
- 39 features (up from 24)
- 84.17% accuracy with cross-validation
- Professional API architecture
- Comprehensive error handling
- Production deployment readiness

---

## Skills Demonstrated

### Machine Learning & Data Science

**Feature Engineering:**
- Transformed 8 raw inputs into 39 ML features
- Created domain-specific features (engagement_binome, dual_needs_flag)
- One-hot encoding for categorical variables
- Numeric scaling with StandardScaler

**Model Development:**
- Random Forest Classifier (500 trees, max_depth=7)
- Stratified 80/20 train-test split
- 10-fold cross-validation for robustness
- Bayesian probability calibration (Platt scaling)

**Model Evaluation:**
- Accuracy: 84.17%
- Precision: 84.92%
- Recall: 84.03%
- F1-Score: 84.47%
- ROC-AUC: 91.54%
- Cross-validated metrics with std dev

**Data Analysis:**
- Exploratory Data Analysis (EDA) in Jupyter
- Statistical correlation analysis
- Feature importance ranking
- Missing value analysis and imputation

### Backend Development

**Flask REST API:**
- RESTful endpoint design (`/api/predict`, `/api/health`)
- JSON request/response handling
- CORS configuration for cross-origin access
- Comprehensive error handling and logging
- Model loading and caching

**Python Engineering:**
- Object-oriented design patterns
- Type hints and documentation
- Error handling with try-except blocks
- Logging with Python logging module
- File I/O with pickle serialization

**Data Processing:**
- Pandas for data manipulation
- NumPy for numerical operations
- Feature transformation pipelines
- Data validation and sanitization

### Frontend Development

**Modern JavaScript:**
- Async/await for API calls
- Fetch API with timeout handling
- Promise-based error handling
- Data transformation functions
- Dynamic DOM manipulation

**User Experience:**
- Loading states with minimum duration
- Visual feedback (status badges)
- Automatic fallback mechanism
- Responsive error messages
- Interactive form validation

**HTML/CSS:**
- Responsive design
- Chart.js for data visualization
- Custom styling and theming
- Accessibility considerations

### Software Engineering Practices

**Code Organization:**
- Modular architecture (backend, frontend, notebooks)
- Separation of concerns
- Reusable functions
- Clear file structure

**Documentation:**
- Comprehensive README with setup instructions
- Inline code comments
- API documentation with examples
- Project context documentation

**Testing:**
- Automated test suite
- Multiple test cases (happy path, edge cases, errors)
- Integration testing (API + model)
- Error handling validation

**Version Control:**
- Git repository structure
- Clean commit history
- `.gitignore` for sensitive files
- README-first approach

**Deployment Readiness:**
- Requirements.txt for dependency management
- Environment configuration
- Gunicorn for production WSGI server
- Health check endpoints

### Domain Knowledge

**Mentorship Program Insights:**
- Understanding of risk factors (engagement, confidence, availability)
- Knowledge of seasonal patterns (summer = higher risk)
- Awareness of matching quality importance
- Recognition of early warning signs

**Nonprofit Sector Awareness:**
- Limited resource allocation challenges
- Privacy and data protection concerns
- Measurement and impact evaluation
- Volunteer management complexities

### Problem-Solving

**Privacy Challenge:**
- Identified data access constraint
- Developed synthetic data solution
- Preserved statistical validity
- Maintained model performance

**Technical Challenges:**
- UTF-8 encoding issues on Windows → Fixed with encoding wrappers
- API integration complexity → Implemented fallback mechanism
- Feature count mismatch → Updated to 39 features throughout
- Model persistence → Implemented pickle serialization

**User Experience Challenges:**
- API availability uncertainty → Automatic fallback to client-side
- Loading state management → 2-second minimum with smooth UX
- Error communication → Clear visual indicators and console logs

---

## Technical Architecture

### Data Flow

```
1. User Input (Frontend)
   ↓
2. JavaScript Form Handler
   ↓
3. Data Transformation (form → API format)
   ↓
4. HTTP POST to /api/predict
   ↓
5. Flask Request Handler
   ↓
6. Feature Engineering (8 inputs → 39 features)
   ↓
7. StandardScaler Normalization
   ↓
8. Random Forest Prediction
   ↓
9. Bayesian Calibration
   ↓
10. Risk Metrics Calculation
    ↓
11. JSON Response
    ↓
12. Frontend Display with Charts
```

### Model Architecture

**Input Layer (8 features):**
- workfield (categorical)
- study_level (categorical)
- needs (categorical)
- registration_month (categorical)
- engagement_score (numeric, 0-3)
- project_confidence_level (numeric, 1-5)
- mentor_availability (numeric, hours/month)
- previous_rejection (binary, 0/1)

**Feature Engineering Layer (39 features):**
- Numeric features (7): engagement_score, project_confidence_level, etc.
- One-hot encoded workfield (13): Computer Science, Engineering, etc.
- One-hot encoded study_level (7): Bac+1, Bac+2, ..., Bac+5+
- One-hot encoded needs (3): Professional, Academic, Both
- One-hot encoded month (9): January, February, ..., December
- Engineered features: engagement_binome, dual_needs_flag, summer_registration

**Random Forest Classifier:**
- 500 decision trees
- Max depth: 7
- Min samples split: 20
- Min samples leaf: 10
- Class weight: balanced (handles imbalanced data)
- Bootstrap: True (bagging for variance reduction)

**Calibration Layer:**
- Bayesian adjustment: `calibrated = (raw - 0.2) / 0.9`
- Maps RF probabilities to real-world failure rates

**Output Layer (4 metrics):**
- Response Risk (0-100%): Overall failure probability
- Match Quality (0-100%): Mentor-mentee compatibility
- Motivation Risk (0-100%): Engagement/dropout risk
- Days to Failure: Expected time until dropout

---

## Impact and Applications

### For Nonprofits

**Proactive Intervention:**
- Identify at-risk mentees before dropout
- Allocate counseling resources efficiently
- Trigger automated check-ins for high-risk cases

**Resource Optimization:**
- Focus limited staff time on highest-need cases
- Data-driven mentor assignment
- Optimize program timing (avoid summer registrations for CS students)

**Program Improvement:**
- Understand which factors drive success
- Measure intervention effectiveness
- Track trends over time

### For Researchers

**Methodology:**
- Replicable ML pipeline
- Open-source code (privacy-safe)
- Statistical validation (cross-validation, metrics)

**Extensions:**
- Test on other mentorship programs
- Add new features (demographics, communication frequency)
- Experiment with other algorithms (XGBoost, neural networks)

### For Developers

**Learning Resource:**
- Complete end-to-end ML project
- Flask API development patterns
- Frontend-backend integration
- Testing and documentation best practices

**Template:**
- Adaptable for other prediction tasks
- Reusable API structure
- Fallback mechanism pattern
- Synthetic data generation approach

---

## Future Enhancements

### Model Improvements

- **Real-time learning:** Update model with new data monthly
- **Explainability:** SHAP values for individual predictions
- **Additional features:** Communication frequency, mentor experience
- **Ensemble methods:** Combine Random Forest with XGBoost

### System Features

- **Authentication:** JWT tokens for API security
- **Rate limiting:** Prevent API abuse
- **Database:** Store predictions and track outcomes
- **Dashboard:** Admin panel for program managers
- **Notifications:** Email/SMS alerts for critical risk cases

### Deployment

- **Containerization:** Docker for consistent environments
- **CI/CD:** Automated testing and deployment
- **Cloud hosting:** AWS/Azure/GCP deployment
- **Monitoring:** Application performance monitoring (APM)
- **Logging:** Centralized log aggregation

---

## Acknowledgments

**JP Morgan Data for Good Hackathon 2025**
- Thank you for providing the challenge and real-world data
- Gratitude to nonprofit partners who shared their data
- Appreciation for the opportunity to apply ML for social impact

**Technologies Used:**
- Python 3.12
- scikit-learn 1.4.0
- Flask 3.0.0
- pandas 2.1.4
- NumPy 1.26.4
- Chart.js (frontend visualization)

**Inspiration:**
This project demonstrates how machine learning can empower nonprofits with limited resources to maximize their social impact through data-driven decision making.

---

## Contact and Contribution

**Repository:** Team-2 (JP Morgan Data for Good Hackathon 2025)
**Status:** Production-ready, open for contribution
**License:** To be determined (consider MIT or Apache 2.0 for open-source)

**Ways to Contribute:**
- Test with your nonprofit's data (privacy-safe)
- Improve the model with new features
- Enhance the frontend UI/UX
- Add multilingual support
- Create mobile app version

---

*This project was built with the mission of empowering nonprofits to create lasting mentorship relationships that change lives.*

**October 2024**
**JP Morgan Data for Good Hackathon 2025 - Team 2**
