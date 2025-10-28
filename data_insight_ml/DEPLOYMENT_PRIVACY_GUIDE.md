# Deployment & Privacy Guide for NGOs

## 🔒 Privacy-First Deployment

This guide ensures your data **never leaves your organization** and runs completely offline.

---

## Understanding the Architecture

### Current Setup (Privacy-Safe)
```
Your Computer
├── Data files (local storage only)
├── Python scripts (runs locally)
├── Flask API (localhost:5000 - not accessible from internet)
└── Browser (connects to localhost only)
```

**Key Privacy Features:**
- ✅ All data stays on your computer
- ✅ No internet connection required (after installation)
- ✅ No cloud services
- ✅ No external API calls
- ✅ No data transmission

---

## Deployment Options

### Option 1: Single User Desktop (Recommended for Most NGOs)

**Best for:** Individual analysts, small teams, sensitive data

**Setup:**
```bash
# 1. Install once
pip install -r requirements.txt

# 2. Run whenever needed
python app.py

# 3. Open in browser
# Visit: http://localhost:5000
```

**Privacy Level:** ⭐⭐⭐⭐⭐ (Maximum)
- Only you can access it
- No network exposure
- Data never leaves your machine

---

### Option 2: Local Network Deployment (Team Access)

**Best for:** Teams in same office, shared analysis

**Setup:**
```bash
# In config.yaml, change:
api:
  host: "0.0.0.0"  # Allows local network access
  port: 5000

# Run the server
python app.py

# Team members access via:
# http://YOUR_COMPUTER_IP:5000
# Example: http://192.168.1.100:5000
```

**Privacy Level:** ⭐⭐⭐⭐ (High)
- Only accessible on your office network
- Data stays within organization
- Not accessible from internet

**Security Recommendations:**
1. Use on trusted office WiFi only
2. Add password protection (see below)
3. Monitor who has access

---

### Option 3: Secure Server Deployment (Advanced)

**Best for:** Large NGOs, multiple locations, need remote access

**Requirements:**
- Dedicated server or cloud VM (AWS, Azure, DigitalOcean)
- HTTPS certificate
- Strong authentication

**Setup Overview:**
```bash
# 1. Deploy on private server
# 2. Set up HTTPS
# 3. Add authentication
# 4. Restrict IP access
# 5. Enable logging
```

**Privacy Level:** ⭐⭐⭐ (Medium-High)
- Depends on server configuration
- Requires IT expertise
- Can be made very secure

See "Advanced Deployment" section below for details.

---

## Enhanced Privacy Features

### 1. Data Upload Security

**Problem:** Currently need to manually place files
**Solution:** Secure file upload with automatic cleanup

We'll add:
- ✅ Web-based file upload
- ✅ Automatic file encryption (optional)
- ✅ Auto-delete after 24 hours
- ✅ No file name logging
- ✅ Isolated storage per session

### 2. Access Control

**Add Password Protection:**

```python
# Add to app.py
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import check_password_hash

auth = HTTPBasicAuth()

users = {
    "ngo_admin": "hashed_password_here"  # Change this!
}

@auth.verify_password
def verify_password(username, password):
    if username in users and check_password_hash(users[username], password):
        return username

# Protect all routes
@app.route('/api/predict', methods=['POST'])
@auth.login_required
def predict():
    # ... existing code
```

**Install:**
```bash
pip install Flask-HTTPAuth
```

### 3. Data Retention Policy

**Automatic Cleanup Script:**

Create `cleanup_old_data.py`:
```python
import os
import time

# Delete files older than 24 hours
MAX_AGE = 24 * 60 * 60  # 24 hours in seconds

for folder in ['uploads', 'models', 'temp']:
    if os.path.exists(folder):
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            if time.time() - os.path.getmtime(file_path) > MAX_AGE:
                os.remove(file_path)
                print(f"Deleted old file: {file}")
```

**Run daily:**
```bash
# Linux/Mac (cron job)
0 0 * * * python /path/to/cleanup_old_data.py

# Windows (Task Scheduler)
# Create task to run daily at midnight
```

---

## Step-by-Step: Completely Offline Setup

### For Maximum Privacy (Air-Gapped System)

**Step 1: Prepare Online Computer**
```bash
# Download all dependencies
pip download -r requirements.txt -d ./packages

# Copy to USB:
# - data_insight_ml folder
# - packages folder
```

**Step 2: Transfer to Offline Computer**
```bash
# Install from downloaded packages
pip install --no-index --find-links=./packages -r requirements.txt

# Run system (no internet needed)
python app.py
```

**Result:** Completely offline, zero internet exposure

---

## Network Isolation Methods

### Method 1: Firewall Rules

**Windows:**
```powershell
# Block Python from internet (after installation)
New-NetFirewallRule -DisplayName "Block Python Internet" `
  -Direction Outbound -Program "C:\Python\python.exe" -Action Block
```

**Linux:**
```bash
# Using iptables
sudo iptables -A OUTPUT -m owner --uid-owner $(id -u) -j REJECT
```

### Method 2: Virtual Machine

```
Host Computer (with internet)
└── Virtual Machine (isolated)
    └── Data Insight ML (no internet access)
```

**Benefits:**
- Complete isolation
- Snapshot/rollback capability
- Easy to backup

**Tools:** VirtualBox, VMware, Hyper-V

### Method 3: Docker Container (Network Disabled)

```bash
# Run with no network
docker run --network none -p 5000:5000 data-insight-ml
```

---

## Data Upload Workflow (Privacy-Safe)

### Current Workflow (Manual)
```
1. Save CSV to folder
2. Run: python prepare_data.py file.csv
3. Run: python train_model.py
4. Run: python app.py
5. Use predictions
```

### Enhanced Workflow (Web-Based, Still Private)
```
1. Open browser → localhost:5000
2. Drag-and-drop CSV file
3. Click "Prepare & Train" button
4. Wait for progress bar
5. Make predictions immediately
```

**Privacy maintained:**
- File uploaded to local folder only
- Processed on your machine
- Deleted after session (optional)
- No external transmission

---

## Security Checklist for NGOs

### Before Deploying:

- [ ] Change default passwords (if using auth)
- [ ] Disable debug mode in production
- [ ] Set up HTTPS (if network-accessible)
- [ ] Configure firewall rules
- [ ] Test on isolated network first
- [ ] Document who has access
- [ ] Set up access logging
- [ ] Create backup procedure
- [ ] Test recovery process
- [ ] Train staff on security

### Data Handling:

- [ ] Encrypt sensitive data at rest
- [ ] Use strong passwords
- [ ] Limit access to authorized personnel
- [ ] Regular security audits
- [ ] Keep software updated
- [ ] Monitor access logs
- [ ] Secure physical access to server
- [ ] Have data deletion procedure

---

## Compliance Considerations

### GDPR (Europe)
- ✅ Data stays on-premises (Article 44)
- ✅ No third-party processors
- ✅ Full data control
- ⚠ Implement data deletion on request
- ⚠ Document processing activities
- ⚠ Conduct DPIA if high-risk

### HIPAA (Healthcare - US)
- ✅ Local storage only
- ⚠ Need encryption at rest
- ⚠ Need access controls
- ⚠ Need audit logs
- ⚠ Business Associate Agreement not needed (self-hosted)

### General Best Practices
- Anonymize data before analysis
- Remove personally identifiable information (PII)
- Document data flows
- Regular security reviews
- Incident response plan

---

## Troubleshooting Privacy Concerns

### "Is my data being sent anywhere?"

**Check:**
```bash
# Monitor network traffic (Linux)
sudo tcpdump -i any host YOUR_IP

# Monitor network traffic (Windows)
# Use Resource Monitor → Network tab
```

**Expected:** Only local connections (127.0.0.1, localhost)

### "How do I verify it's truly offline?"

**Test:**
```bash
# 1. Disconnect from internet
# 2. Run: python app.py
# 3. Open: http://localhost:5000

# If it works → fully offline ✓
```

### "Can someone hack into it?"

**If properly deployed:**
- Single-user desktop: No external access possible
- Local network: Only office network (same as file shares)
- Server deployment: Requires security configuration

**Security layers:**
1. Firewall (blocks external access)
2. Authentication (username/password)
3. HTTPS (encrypted transmission)
4. Access logs (audit trail)

---

## Recommended Setup for Different NGO Sizes

### Small NGO (1-5 people)
```yaml
Deployment: Single desktop
Privacy: Maximum (localhost only)
Access: One analyst
Setup Time: 5 minutes
Cost: Free
```

### Medium NGO (5-20 people)
```yaml
Deployment: Shared office server
Privacy: High (local network only)
Access: Office WiFi only
Setup Time: 30 minutes
Cost: Free
```

### Large NGO (20+ people)
```yaml
Deployment: Dedicated server with auth
Privacy: Medium-High (with HTTPS + auth)
Access: Authorized users only
Setup Time: 2-4 hours
Cost: Server hosting ($5-20/month)
```

---

## Quick Start: Most Secure Setup

```bash
# 1. Install on air-gapped machine
pip install -r requirements.txt

# 2. Never connect to internet again

# 3. Use USB for data transfer only

# 4. Run locally
python app.py

# 5. Access only via localhost
# Browser: http://localhost:5000

# 6. Delete data after analysis
rm -rf uploads/* models/*
```

**Result:** 100% private, zero internet exposure, maximum security.

---

## Support & Questions

**Common Questions:**

**Q: Is this really free?**
A: Yes, 100% free and open-source.

**Q: Do I need internet after installation?**
A: No, works completely offline.

**Q: Who can see my data?**
A: Only people with physical access to your computer (unless you configure network access).

**Q: What about updates?**
A: Download updates on a separate internet-connected machine, transfer via USB.

**Q: Can I use this for confidential data?**
A: Yes, with proper setup it's suitable for sensitive data.

---

## Next Steps

1. Choose deployment option (recommend: Single Desktop)
2. Follow setup instructions
3. Test with sample data first
4. Implement security measures
5. Train your team
6. Document your setup
7. Regular backups

**Remember:** Privacy is about proper configuration, not just software. Always follow your organization's data policies!
