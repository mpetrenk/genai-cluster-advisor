# 🛡️ Troubleshooting Guide - CrowdStrike & Security Software Compatibility

This guide helps resolve issues when running the Cluster Recommendation Wizard with enterprise security software like CrowdStrike Falcon.

## 🚨 Common Issues & Solutions

### Issue 1: "Missing frpc file" Error

**Error Message:**
```
Could not create share link. Missing file: /Users/maxpetre/.cache/huggingface/gradio/frpc/frpc_darwin_arm64_v0.3
```

**Root Cause:** CrowdStrike or other security software blocked the download of the frpc (Fast Reverse Proxy Client) file.

**Solutions:**

#### Option A: Manual Download (Recommended)
```bash
# Create directory
mkdir -p /Users/maxpetre/.cache/huggingface/gradio/frpc

# Download file manually
curl -L "https://cdn-media.huggingface.co/frpc-gradio-0.3/frpc_darwin_arm64" \
     -o "/Users/maxpetre/.cache/huggingface/gradio/frpc/frpc_darwin_arm64_v0.3"

# Make executable
chmod +x /Users/maxpetre/.cache/huggingface/gradio/frpc/frpc_darwin_arm64_v0.3
```

#### Option B: Use Secure Launch Script
```bash
./launch_secure.sh
```

#### Option C: Run in Local Mode Only
Use `app_improved.py` which automatically detects missing frpc and runs locally.

### Issue 2: CrowdStrike Blocking Network Connections

**Symptoms:**
- Application starts but can't create public URLs
- Network timeouts during startup
- "Connection refused" errors

**Solutions:**

#### Temporary CrowdStrike Exclusion
1. Contact your IT administrator
2. Request temporary exclusion for:
   - `/Users/maxpetre/.cache/huggingface/`
   - Python process for Gradio
   - Port 7860 (local server)

#### Alternative: Local Mode
```bash
# Run without public sharing
python3 app_improved.py
# Access at: http://127.0.0.1:7860
```

### Issue 3: Python Process Blocked

**Symptoms:**
- Application won't start
- "Permission denied" errors
- Process killed unexpectedly

**Solutions:**

#### Check Process Whitelist
```bash
# Check if Python is allowed
ps aux | grep python
```

#### Use Alternative Python
```bash
# Try with different Python executable
python3 app_improved.py
# or
/usr/bin/python3 app_improved.py
```

### Issue 4: File System Access Denied

**Symptoms:**
- Can't read/write dataset files
- Permission errors on cache directories

**Solutions:**

#### Check File Permissions
```bash
# Fix permissions
chmod 755 "cluster recommendation wizard"
chmod 644 *.py *.csv *.json
```

#### Alternative Data Location
```bash
# Copy dataset to user directory
cp cluster_dataset.csv ~/cluster_dataset.csv
# Update app to use ~/cluster_dataset.csv
```

## 🔧 CrowdStrike-Specific Configuration

### Recommended Exclusions (Request from IT)

**File/Folder Exclusions:**
```
/Users/maxpetre/.cache/huggingface/
/Users/maxpetre/cluster recommendation wizard/
~/cluster_dataset.csv
```

**Process Exclusions:**
```
python3 (when running Gradio applications)
frpc_darwin_arm64_v0.3
```

**Network Exclusions:**
```
Port 7860 (local server)
*.gradio.live (for public sharing)
cdn-media.huggingface.co (for downloads)
```

### Safe Mode Configuration

If you can't get exclusions, use these safe settings:

```bash
# Environment variables for secure mode
export GRADIO_ANALYTICS_ENABLED=false
export GRADIO_DEBUG=false
export GRADIO_SHARE=false

# Launch in local mode only
python3 app_improved.py
```

## 🚀 Launch Options by Security Level

### High Security Environment
```bash
# Local only, no external connections
python3 -c "
import os
os.environ['GRADIO_SHARE'] = 'false'
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'false'
exec(open('app_improved.py').read())
"
```

### Medium Security Environment
```bash
# Use secure launch script
./launch_secure.sh
```

### Standard Environment
```bash
# Full functionality
python3 app.py
```

## 🔍 Diagnostic Commands

### Check CrowdStrike Status
```bash
# Check if CrowdStrike is running
pgrep -f falcon
ps aux | grep -i crowdstrike
```

### Test Network Connectivity
```bash
# Test Hugging Face CDN
curl -I https://cdn-media.huggingface.co/frpc-gradio-0.3/frpc_darwin_arm64

# Test local server
curl -I http://127.0.0.1:7860
```

### Verify File Permissions
```bash
# Check cache directory
ls -la /Users/maxpetre/.cache/huggingface/gradio/frpc/

# Check application files
ls -la "cluster recommendation wizard/"
```

### Test Python Environment
```bash
# Test imports
python3 -c "
import gradio
import pandas
import sklearn
print('All imports successful')
"
```

## 📞 Getting Help

### Internal IT Support
When contacting IT support, provide:

1. **Error Message:** Full error text
2. **Security Software:** CrowdStrike Falcon version
3. **Attempted Solutions:** What you've tried
4. **Business Justification:** Why you need this tool

### Application Logs
```bash
# Run with verbose logging
python3 app_improved.py 2>&1 | tee application.log
```

### System Information
```bash
# Gather system info
echo "OS: $(uname -a)"
echo "Python: $(python3 --version)"
echo "User: $(whoami)"
echo "Working Directory: $(pwd)"
```

## 🛠️ Alternative Solutions

### Option 1: Docker Container
```bash
# Run in isolated container (if Docker allowed)
docker run -p 7860:7860 -v "$(pwd):/app" python:3.9 \
  bash -c "cd /app && pip install -r requirements.txt && python app_improved.py"
```

### Option 2: Virtual Environment
```bash
# Create isolated Python environment
python3 -m venv cluster_wizard_env
source cluster_wizard_env/bin/activate
pip install -r requirements.txt
python app_improved.py
```

### Option 3: Jupyter Notebook
```bash
# Run as Jupyter notebook (often less restricted)
pip install jupyter
jupyter notebook
# Then run code cells from app_improved.py
```

## 📋 Pre-deployment Checklist

Before deploying in a CrowdStrike environment:

- [ ] Test frpc download manually
- [ ] Verify Python process permissions
- [ ] Check network connectivity
- [ ] Test file system access
- [ ] Confirm port 7860 availability
- [ ] Validate dataset generation
- [ ] Test application startup
- [ ] Verify local access works
- [ ] Document any exclusions needed

## 🔄 Recovery Procedures

### If Application Crashes
```bash
# Clean restart
pkill -f "python.*app"
rm -rf /tmp/gradio_*
python3 app_improved.py
```

### If Files Corrupted
```bash
# Regenerate dataset
rm cluster_dataset.csv cluster_dataset.json
python3 generate_dataset.py
```

### If Cache Issues
```bash
# Clear Gradio cache
rm -rf /Users/maxpetre/.cache/huggingface/gradio/
./launch_secure.sh
```

---

**Remember:** Always work with your IT security team to ensure compliance with organizational policies while maintaining functionality.
