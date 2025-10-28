# Interactive Data Cleaning Implementation Guide

## Overview

I've added a complete backend API for interactive data cleaning with user approval workflow. The system now provides:

1. **Analyze uploaded CSV** - Detect all data quality issues
2. **Show suggestions** - Present cleaning recommendations to user
3. **Get approval** - User reviews and approves/rejects suggestions
4. **Clean data** - Apply approved cleaning steps
5. **Generate predictions** - Predict on cleaned data
6. **Download results** - Get cleaned CSV and prediction report

## Backend API - COMPLETED ✓

### New Endpoints Added

#### 1. `/api/analyze-csv` (POST)
**Purpose:** Analyze uploaded CSV for data quality issues

**Input:**
- `file`: CSV file (multipart/form-data)
- `domain`: Domain name (student, donor, etc.)

**Output:**
```json
{
  "success": true,
  "analysis": {
    "total_rows": 53,
    "total_columns": 13,
    "quality_score": 85.9,
    "duplicates": 3,
    "missing_values": {
      "age": {"count": 7, "percentage": 13.2},
      "gpa": {"count": 8, "percentage": 15.1}
    },
    "outliers": {
      "attendance_rate": 2,
      "absences_per_month": 1
    },
    "invalid_values": {
      "attendance_rate": 2,
      "gpa": 1
    },
    "categorical_issues": {
      "parent_involvement": "Inconsistent formatting"
    },
    "suggestions": [
      {
        "type": "missing_values",
        "column": "age",
        "issue": "7 missing values (13.2%)",
        "action": "Fill with median",
        "recommended": true
      },
      {
        "type": "duplicates",
        "issue": "3 duplicate rows found",
        "action": "Remove duplicate rows",
        "recommended": true
      }
      // ... more suggestions
    ]
  }
}
```

#### 2. `/api/clean-and-predict` (POST)
**Purpose:** Clean CSV with approved suggestions and generate predictions

**Input:**
- `file`: CSV file
- `domain`: Domain name
- `suggestions`: JSON array of suggestions with `approved: true/false`
- `custom_notes`: Optional custom cleaning notes

**Example suggestions input:**
```json
[
  {
    "type": "duplicates",
    "issue": "3 duplicate rows found",
    "action": "Remove duplicate rows",
    "approved": true
  },
  {
    "type": "missing_values",
    "column": "age",
    "issue": "7 missing values",
    "action": "Fill with median",
    "approved": true
  },
  {
    "type": "outliers",
    "column": "attendance_rate",
    "issue": "2 extreme outliers detected",
    "action": "Cap outliers using IQR method",
    "approved": false
  }
]
```

**Output:**
```json
{
  "success": true,
  "cleaning_log": [
    "Removed 3 duplicate rows",
    "Filled 7 missing values in age with median",
    "Capped 2 outliers in attendance_rate",
    "Standardized categorical values in parent_involvement",
    "Added derived features for student"
  ],
  "results": [
    {
      "identifiers": {
        "student_id": "STU5001",
        "student_name": "Emma Martin"
      },
      "prediction": 0,
      "confidence": 0.677,
      "recommendation": "Manual review recommended"
    }
    // ... more predictions
  ],
  "count": 50,
  "domain": "student",
  "cleaned_filename": "cleaned_messy_student_data.csv",
  "original_rows": 53,
  "cleaned_rows": 50
}
```

#### 3. `/api/download-cleaned-csv/<filename>` (GET)
**Purpose:** Download the cleaned CSV file

**Response:** CSV file download

#### 4. `/api/download-prediction-report` (POST)
**Purpose:** Generate and download prediction report

**Input:**
```json
{
  "results": [...],  // Array of prediction results
  "domain": "student",
  "cleaning_log": [...]  // Array of cleaning actions taken
}
```

**Response:** CSV file with predictions and metadata

---

## Frontend Implementation - TODO

### Step 1: Add CSS Styles (Add to `<style>` section)

```css
/* Toggle Switch */
.toggle-container {
    display: flex;
    align-items: center;
    margin: 20px 0;
    padding: 15px;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 8px;
}

.toggle-switch {
    position: relative;
    width: 50px;
    height: 24px;
    margin-right: 12px;
}

.toggle-switch input {
    opacity: 0;
    width: 0;
    height: 0;
}

.toggle-slider {
    position: absolute;
    cursor: pointer;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: #555;
    transition: .4s;
    border-radius: 24px;
}

.toggle-slider:before {
    position: absolute;
    content: "";
    height: 18px;
    width: 18px;
    left: 3px;
    bottom: 3px;
    background-color: white;
    transition: .4s;
    border-radius: 50%;
}

input:checked + .toggle-slider {
    background-color: #42A5F5;
}

input:checked + .toggle-slider:before {
    transform: translateX(26px);
}

/* Analysis Panel */
.analysis-panel {
    display: none;
    margin: 20px 0;
    padding: 20px;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.analysis-panel.show {
    display: block;
}

.analysis-summary {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 15px;
    margin-bottom: 20px;
}

.analysis-stat {
    padding: 12px;
    background: rgba(66, 165, 245, 0.1);
    border-radius: 6px;
    border-left: 3px solid #42A5F5;
}

.analysis-stat-value {
    font-size: 1.5em;
    font-weight: bold;
    color: #42A5F5;
}

.analysis-stat-label {
    font-size: 0.9em;
    color: #B2BAC2;
    margin-top: 5px;
}

/* Suggestions List */
.suggestions-list {
    margin: 20px 0;
}

.suggestion-item {
    display: flex;
    align-items: start;
    padding: 12px;
    margin: 8px 0;
    background: rgba(255, 255, 255, 0.03);
    border-radius: 6px;
    border-left: 3px solid #FFA726;
}

.suggestion-item.recommended {
    border-left-color: #66BB6A;
}

.suggestion-checkbox {
    margin-right: 12px;
    margin-top: 3px;
}

.suggestion-content {
    flex: 1;
}

.suggestion-issue {
    color: #FFA726;
    font-weight: 500;
    margin-bottom: 5px;
}

.suggestion-action {
    color: #B2BAC2;
    font-size: 0.9em;
}

/* Custom Notes */
.custom-notes {
    width: 100%;
    padding: 12px;
    margin: 15px 0;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 6px;
    color: #E0E0E0;
    font-family: inherit;
    resize: vertical;
    min-height: 60px;
}

/* Cleaning Log */
.cleaning-log {
    margin: 20px 0;
    padding: 15px;
    background: rgba(76, 175, 80, 0.1);
    border-radius: 6px;
    border-left: 3px solid #4CAF50;
}

.cleaning-log-item {
    padding: 5px 0;
    color: #B2BAC2;
    font-size: 0.9em;
}

.cleaning-log-item:before {
    content: "✓ ";
    color: #4CAF50;
    margin-right: 8px;
}

/* Download Section */
.download-section {
    display: flex;
    gap: 10px;
    margin: 20px 0;
    flex-wrap: wrap;
}

.download-btn {
    flex: 1;
    min-width: 200px;
    padding: 12px 20px;
    background: #42A5F5;
    color: white;
    border: none;
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.95em;
    transition: background 0.3s;
}

.download-btn:hover {
    background: #1E88E5;
}

.download-btn:disabled {
    background: #555;
    cursor: not-allowed;
}
```

### Step 2: Add HTML Elements (Add inside upload tab, after upload-zone)

```html
<!-- Add after the upload-zone div -->
<div class="toggle-container">
    <label class="toggle-switch">
        <input type="checkbox" id="cleaningToggle">
        <span class="toggle-slider"></span>
    </label>
    <span>Enable Interactive Data Cleaning</span>
</div>

<!-- Analysis Panel -->
<div class="analysis-panel" id="analysisPanel">
    <h3>Data Quality Analysis</h3>

    <div class="analysis-summary" id="analysisSummary"></div>

    <h4>Cleaning Suggestions</h4>
    <div class="suggestions-list" id="suggestionsList"></div>

    <label for="customNotes">Additional Notes (Optional):</label>
    <textarea id="customNotes" class="custom-notes" placeholder="Enter any specific cleaning instructions..."></textarea>

    <button class="btn" id="approveBtn" style="margin-top: 15px;">Apply Cleaning & Predict</button>
</div>

<!-- Cleaning Log (shown after cleaning) -->
<div class="cleaning-log" id="cleaningLog" style="display: none;">
    <h4>Cleaning Applied:</h4>
    <div id="cleaningLogContent"></div>
</div>

<!-- Download Section -->
<div class="download-section" id="downloadSection" style="display: none;">
    <button class="download-btn" id="downloadCleanedBtn" disabled>
        Download Cleaned CSV
    </button>
    <button class="download-btn" id="downloadReportBtn" disabled>
        Download Prediction Report
    </button>
</div>
```

### Step 3: Add JavaScript Functions (Add to `<script>` section)

```javascript
let analysisData = null;
let cleaningResults = null;
let cleanedFilename = null;

// Toggle cleaning mode
document.getElementById('cleaningToggle').addEventListener('change', function() {
    if (this.checked && uploadedFile) {
        analyzeUploadedFile();
    } else {
        document.getElementById('analysisPanel').classList.remove('show');
    }
});

// Analyze uploaded file
async function analyzeUploadedFile() {
    if (!uploadedFile || !currentDomain) return;

    showSpinner();

    const formData = new FormData();
    formData.append('file', uploadedFile);
    formData.append('domain', currentDomain);

    try {
        const response = await fetch(`${API_BASE}/analyze-csv`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        hideSpinner();

        if (data.success) {
            analysisData = data.analysis;
            displayAnalysis(data.analysis);
        } else {
            showError(data.error || 'Analysis failed');
        }
    } catch (error) {
        hideSpinner();
        showError('Error analyzing file: ' + error.message);
    }
}

// Display analysis results
function displayAnalysis(analysis) {
    document.getElementById('analysisPanel').classList.add('show');

    // Display summary statistics
    const summaryDiv = document.getElementById('analysisSummary');
    summaryDiv.innerHTML = `
        <div class="analysis-stat">
            <div class="analysis-stat-value">${analysis.quality_score}%</div>
            <div class="analysis-stat-label">Data Quality</div>
        </div>
        <div class="analysis-stat">
            <div class="analysis-stat-value">${analysis.total_rows}</div>
            <div class="analysis-stat-label">Total Rows</div>
        </div>
        <div class="analysis-stat">
            <div class="analysis-stat-value">${analysis.duplicates}</div>
            <div class="analysis-stat-label">Duplicates</div>
        </div>
        <div class="analysis-stat">
            <div class="analysis-stat-value">${Object.keys(analysis.missing_values).length}</div>
            <div class="analysis-stat-label">Columns w/ Missing</div>
        </div>
    `;

    // Display suggestions
    const suggestionsDiv = document.getElementById('suggestionsList');
    if (analysis.suggestions.length === 0) {
        suggestionsDiv.innerHTML = '<p style="color: #4CAF50;">No issues detected - data is clean!</p>';
        return;
    }

    suggestionsDiv.innerHTML = analysis.suggestions.map((sug, index) => `
        <div class="suggestion-item ${sug.recommended ? 'recommended' : ''}">
            <input type="checkbox"
                   class="suggestion-checkbox"
                   id="sug-${index}"
                   ${sug.recommended ? 'checked' : ''}>
            <div class="suggestion-content">
                <div class="suggestion-issue">${sug.issue}</div>
                <div class="suggestion-action">Action: ${sug.action}</div>
            </div>
        </div>
    `).join('');
}

// Apply cleaning and predict
document.getElementById('approveBtn').addEventListener('click', async function() {
    if (!uploadedFile || !currentDomain || !analysisData) return;

    showSpinner();

    // Collect approved suggestions
    const approvedSuggestions = analysisData.suggestions.map((sug, index) => ({
        ...sug,
        approved: document.getElementById(`sug-${index}`).checked
    }));

    const customNotes = document.getElementById('customNotes').value;

    const formData = new FormData();
    formData.append('file', uploadedFile);
    formData.append('domain', currentDomain);
    formData.append('suggestions', JSON.stringify(approvedSuggestions));
    formData.append('custom_notes', customNotes);

    try {
        const response = await fetch(`${API_BASE}/clean-and-predict`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        hideSpinner();

        if (data.success) {
            cleaningResults = data;
            cleanedFilename = data.cleaned_filename;

            // Show cleaning log
            displayCleaningLog(data.cleaning_log);

            // Show results
            displayBatchResults(data);

            // Enable download buttons
            document.getElementById('downloadSection').style.display = 'flex';
            document.getElementById('downloadCleanedBtn').disabled = false;
            document.getElementById('downloadReportBtn').disabled = false;
        } else {
            showError(data.error || 'Cleaning and prediction failed');
        }
    } catch (error) {
        hideSpinner();
        showError('Error: ' + error.message);
    }
});

// Display cleaning log
function displayCleaningLog(log) {
    const logDiv = document.getElementById('cleaningLog');
    const contentDiv = document.getElementById('cleaningLogContent');

    if (log && log.length > 0) {
        contentDiv.innerHTML = log.map(item =>
            `<div class="cleaning-log-item">${item}</div>`
        ).join('');
        logDiv.style.display = 'block';
    }
}

// Download cleaned CSV
document.getElementById('downloadCleanedBtn').addEventListener('click', function() {
    if (cleanedFilename) {
        window.location.href = `${API_BASE}/download-cleaned-csv/${cleanedFilename}`;
    }
});

// Download prediction report
document.getElementById('downloadReportBtn').addEventListener('click', async function() {
    if (!cleaningResults) return;

    try {
        const response = await fetch(`${API_BASE}/download-prediction-report`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                results: cleaningResults.results,
                domain: cleaningResults.domain,
                cleaning_log: cleaningResults.cleaning_log
            })
        });

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `prediction_report_${cleaningResults.domain}_${new Date().getTime()}.csv`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
    } catch (error) {
        showError('Error downloading report: ' + error.message);
    }
});
```

---

## Testing the Backend API

### Test 1: Analyze CSV
```bash
curl -X POST \
  -F "file=@messy_student_data.csv" \
  -F "domain=student" \
  http://localhost:5000/api/analyze-csv
```

**Result:** Returns analysis with 85.9% quality score, 3 duplicates, 97 missing values, and 15+ cleaning suggestions.

### Test 2: Clean and Predict
```bash
curl -X POST \
  -F "file=@messy_student_data.csv" \
  -F "domain=student" \
  -F 'suggestions=[{"type":"duplicates","approved":true},{"type":"missing_values","column":"age","approved":true}]' \
  http://localhost:5000/api/clean-and-predict
```

**Result:** Returns cleaned data (50 rows), cleaning log, and 50 predictions with identifiers.

---

## User Workflow

1. **Upload CSV** → File selected
2. **Toggle ON** → "Enable Interactive Data Cleaning"
3. **Analyze** → System analyzes CSV automatically
4. **Review** → User sees:
   - Quality score (85.9%)
   - Issues found (duplicates, missing values, outliers, etc.)
   - Suggestions with checkboxes (pre-checked for recommended items)
5. **Customize** → User can:
   - Check/uncheck suggestions
   - Add custom notes
6. **Approve** → Click "Apply Cleaning & Predict"
7. **View Results** → See:
   - Cleaning log (what was done)
   - Predictions (with IDs and names)
8. **Download** → Get:
   - Cleaned CSV file
   - Prediction report with metadata

---

## Benefits

✓ **Transparent** - User sees exactly what will be cleaned
✓ **Controllable** - User approves/rejects each suggestion
✓ **Traceable** - Cleaning log shows what was done
✓ **Downloadable** - Get both cleaned data and predictions
✓ **ML-Ready** - Cleaned data ready for training
✓ **Professional** - Proper workflow for production use

---

## Backend Status: ✓ COMPLETE

- ✓ Analysis endpoint working
- ✓ Clean-and-predict endpoint working
- ✓ Download endpoints working
- ✓ Temp file storage working
- ✓ Tested with messy student data

## Frontend Status: TODO

- ☐ Add CSS styles
- ☐ Add HTML elements
- ☐ Add JavaScript functions
- ☐ Test full workflow

---

## Next Steps

1. Copy the CSS, HTML, and JavaScript code above into `demo.html`
2. Test the workflow with messy_student_data.csv
3. Verify downloads work
4. Customize styling as needed

The backend is fully functional and ready to use!
