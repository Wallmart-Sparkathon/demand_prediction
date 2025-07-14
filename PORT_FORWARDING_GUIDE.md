# Port Forwarding Troubleshooting Guide

## Common Issues and Solutions

### 1. "Unable to forward localhost:5000" Error

**Causes:**
- Server binding to localhost (127.0.0.1) instead of all interfaces
- Port already in use
- Windows Firewall blocking the port
- Antivirus software interference

**Solutions:**

#### A. Use Enhanced Server
```bash
# Use the enhanced server that binds to all interfaces
start_api_enhanced.bat
```

#### B. Check Port Availability
```cmd
# Check if port 5000 is available
netstat -an | find "5000"

# If port is busy, kill the process
netstat -ano | find "5000"
taskkill /PID <process_id> /F
```

#### C. Windows Firewall
```cmd
# Allow port 5000 through Windows Firewall
netsh advfirewall firewall add rule name="Flask API" dir=in action=allow protocol=TCP localport=5000
```

### 2. Alternative Port Configuration

If port 5000 doesn't work, modify the server:

```python
# In api_server.py, change the port
app.run(debug=False, host='0.0.0.0', port=8000)
```

### 3. Network Binding Issues

**Check server binding:**
```bash
# The server should show:
# * Running on all addresses (0.0.0.0)
# * Running on http://0.0.0.0:5000
```

**If showing localhost only:**
- Use `api_server_enhanced.py` instead
- Ensure `host='0.0.0.0'` is set

### 4. Testing Server Accessibility

#### Local Testing
```bash
# Test from same machine
curl http://localhost:5000/health
curl http://127.0.0.1:5000/health
```

#### Network Testing
```bash
# Get your IP address
ipconfig | findstr IPv4

# Test from network IP
curl http://YOUR_IP:5000/health
```

### 5. Port Forwarding Service Configuration

#### For VS Code Port Forwarding:
1. Use `0.0.0.0:5000` as the target
2. Set visibility to "Public" if needed
3. Try manual port mapping

#### For ngrok:
```bash
ngrok http 5000
```

#### For other forwarding tools:
- Ensure target is `0.0.0.0:5000` not `localhost:5000`
- Check if tool supports binding to all interfaces

### 6. Server Configuration Options

#### Option 1: Standard Server
```bash
python api_server.py
```

#### Option 2: Enhanced Server (Recommended)
```bash
python api_server_enhanced.py
```

#### Option 3: Custom Port
```bash
# Modify api_server.py to use different port
app.run(debug=False, host='0.0.0.0', port=8000)
```

### 7. Network Diagnostics

#### Check Network Interfaces
```cmd
# Windows
ipconfig /all

# Show all listening ports
netstat -an | find "LISTENING"
```

#### Test Port Accessibility
```cmd
# Test if port is accessible externally
telnet YOUR_IP 5000
```

### 8. Common Error Messages and Fixes

| Error | Cause | Solution |
|-------|-------|----------|
| `Address already in use` | Port 5000 busy | Use different port or kill process |
| `Permission denied` | Admin rights needed | Run as administrator |
| `Connection refused` | Server not accessible | Check firewall/binding |
| `Forwarding exited with code 1` | Binding/access issue | Use enhanced server |

### 9. Production Deployment

For production deployment, consider using:

#### Gunicorn (Linux/Mac)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 api_server:app
```

#### Waitress (Windows)
```bash
pip install waitress
waitress-serve --host=0.0.0.0 --port=5000 api_server:app
```

### 10. Verification Steps

1. **Start Server:**
   ```bash
   start_api_enhanced.bat
   ```

2. **Check Server Output:**
   - Should show "Running on http://0.0.0.0:5000"
   - Should display local IP address

3. **Test Health Endpoint:**
   ```bash
   curl http://localhost:5000/health
   ```

4. **Test External Access:**
   ```bash
   curl http://YOUR_IP:5000/health
   ```

5. **Setup Port Forwarding:**
   - Use target: `0.0.0.0:5000` or `YOUR_IP:5000`
   - Not `localhost:5000`

### 11. Quick Fix Commands

```cmd
:: Kill any process using port 5000
for /f "tokens=5" %a in ('netstat -aon ^| find ":5000"') do taskkill /PID %a /F

:: Add firewall rule
netsh advfirewall firewall add rule name="Flask API Port 5000" dir=in action=allow protocol=TCP localport=5000

:: Start enhanced server
cd "d:\Code_stuff\demand_prediction"
.spark\Scripts\activate.bat
python api_server_enhanced.py
```

### 12. Alternative Solutions

If port forwarding still doesn't work:

1. **Use Cloud Deployment:**
   - Deploy to Heroku, AWS, or Azure
   - Get public URL automatically

2. **Use Local Tunnel Services:**
   - ngrok: `ngrok http 5000`
   - localtunnel: `npx localtunnel --port 5000`

3. **Use Different Port:**
   - Try ports 8000, 8080, 3000
   - Some firewalls block port 5000

The enhanced server (`api_server_enhanced.py`) should resolve most port forwarding issues by properly binding to all network interfaces and providing better diagnostics.
