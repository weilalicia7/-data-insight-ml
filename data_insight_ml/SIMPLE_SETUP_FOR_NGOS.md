# Simple Setup Guide for NGOs (Non-Technical)

## 🎯 What You'll Achieve

After following this guide, you'll have a **private, web-based ML tool** running on your computer that:
- Never sends your data anywhere
- Works completely offline
- No coding required
- No command line after initial setup

---

## ⏱️ Time Required: 10 Minutes

---

## Step 1: Install Python (One-Time, 5 minutes)

### Windows:
1. Go to: https://www.python.org/downloads/
2. Click "Download Python 3.12"
3. Run the installer
4. ✅ **IMPORTANT**: Check "Add Python to PATH"
5. Click "Install Now"

### Mac:
1. Open Terminal (Cmd + Space, type "Terminal")
2. Paste: `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
3. Then: `brew install python`

### Linux:
```bash
sudo apt update
sudo apt install python3 python3-pip
```

**Verify Installation:**
Open Terminal/Command Prompt and type:
```bash
python --version
```
Should show: `Python 3.x.x`

---

## Step 2: Download Data Insight ML (2 minutes)

### Option A: Download ZIP (Easier)
1. Go to: https://github.com/weilalicia7/-data-insight-ml
2. Click green "Code" button
3. Click "Download ZIP"
4. Unzip to your desired location (e.g., Desktop)

### Option B: Use Git (If you have it)
```bash
git clone https://github.com/weilalicia7/-data-insight-ml.git
cd -data-insight-ml
```

---

## Step 3: Install Dependencies (2 minutes)

Open Terminal/Command Prompt in the downloaded folder:

**Windows:**
1. Navigate to the folder in File Explorer
2. Right-click in the folder
3. Choose "Open in Terminal" or "Open PowerShell here"

**Mac/Linux:**
```bash
cd /path/to/data-insight-ml
```

Then run:
```bash
pip install -r requirements.txt
```

**Note:** This downloads necessary libraries. **One-time only.**

---

## Step 4: Start the System (30 seconds)

In the same terminal:
```bash
python app.py
```

You should see:
```
* Running on http://127.0.0.1:5000
✓ API Connected | Model Loaded
```

**Leave this window open!** This is your private server.

---

## Step 5: Open in Browser

Open your web browser and go to:
```
http://localhost:5000
```

You should see the **Data Insight ML** interface!

---

## 🎉 You're Done! Now What?

### To Use It:

1. **Prepare Your Data:**
   - Export your data as CSV from Excel
   - Make sure you have a column for what you want to predict (e.g., "donated_again", "completed", "approved")

2. **Current Method (Manual):**
   ```bash
   # Generate example data first to test:
   python example_data_generator.py

   # Then prepare your data:
   python prepare_data.py example_donor_retention.csv

   # Train the model:
   python train_model.py

   # Your model is ready!
   ```

3. **Use the Web Interface:**
   - Go to http://localhost:5000
   - Click "Initialize Model"
   - Fill in values
   - Click "Run Prediction"
   - See results!

---

## 🔒 Privacy & Security

### Is My Data Safe?

**YES!** Here's why:

✅ **No Internet Connection Needed** (after installation)
- Your data never leaves your computer
- No cloud services
- No external API calls

✅ **Local Only**
- Only accessible on your computer
- Not visible to anyone else
- Like a Microsoft Word document on your PC

✅ **You Control Everything**
- You can delete all data anytime
- You can turn it off anytime
- You own the software completely

### Can Others Access It?

**On Your Computer Only:** No, only you can access it

**Want to Share with Team?**
- If on same office WiFi, yes (see DEPLOYMENT_PRIVACY_GUIDE.md)
- Still stays within your organization
- Not accessible from internet

---

## Common Questions

### Q: Do I need to pay for this?
**A:** No, completely free forever.

### Q: Do I need internet to use it?
**A:** Only for initial installation. After that, works offline.

### Q: What if I'm not technical?
**A:** That's okay! After setup, it's just a website you use in your browser.

### Q: Can I use this for confidential data?
**A:** Yes, it's designed for privacy. Data stays on your computer.

### Q: What if something breaks?
**A:** Just close the terminal and restart: `python app.py`

### Q: How do I stop it?
**A:** Press `Ctrl+C` in the terminal window.

---

## Daily Usage

### Every Time You Want to Use It:

```bash
# 1. Open terminal in the folder
# 2. Run:
python app.py

# 3. Open browser:
http://localhost:5000

# 4. When done, press Ctrl+C in terminal to stop
```

---

## Next Steps

### New Features Coming Soon:

We're adding a **drag-and-drop interface** so you can:
- Upload CSV directly in the browser
- Train models with one click
- No command line needed

**Check ENHANCEMENT_PLAN.md for details**

### Current Workaround:

For now, use command line for:
- Uploading data: Save CSV to the folder
- Preparing data: `python prepare_data.py your_file.csv`
- Training: `python train_model.py`

Then use web interface for predictions!

---

## Troubleshooting

### "Command not found: python"
**Solution:** Try `python3` instead of `python`

### "Port 5000 already in use"
**Solution:** Something else is using port 5000. Change in `config.yaml`:
```yaml
api:
  port: 5001  # Or any other number
```

### "Module not found"
**Solution:** Run `pip install -r requirements.txt` again

### "Permission denied"
**Solution (Windows):** Run terminal as Administrator
**Solution (Mac/Linux):** Use `sudo pip install ...`

### Still Having Issues?
1. Check DEPLOYMENT_PRIVACY_GUIDE.md
2. Check ENHANCEMENT_PLAN.md
3. Open an issue on GitHub

---

## Getting Help

### Resources:
- **Setup Issues:** This guide
- **Privacy Questions:** DEPLOYMENT_PRIVACY_GUIDE.md
- **Feature Requests:** ENHANCEMENT_PLAN.md
- **How to Use:** README.md
- **Quick Start:** QUICKSTART.md

### Contact:
- Open an issue on GitHub: https://github.com/weilalicia7/-data-insight-ml/issues
- Describe your problem clearly
- We'll help!

---

## Summary

```
✅ Install Python (one-time, 5 min)
✅ Download Data Insight ML (one-time, 2 min)
✅ Install dependencies (one-time, 2 min)
✅ Run: python app.py (every time, 30 sec)
✅ Open: http://localhost:5000 (every time)
✅ Use the interface!

🔒 All data stays on your computer
🆓 Completely free
📴 Works offline
👥 Share with team (optional)
🎯 No coding needed (soon!)
```

**You're ready to use ML for social good!** 🎉
