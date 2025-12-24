# 🚀 COMPLETE DEPLOYMENT GUIDE

**Date:** December 24, 2025  
**Goal:** Deploy Frontend (Vercel) + Backend (Render)

---

## ⚡ QUICK SUMMARY

- **Frontend (Next.js):** Deploy to Vercel
- **Backend (FastAPI):** Deploy to Render.com (free tier)
- **Why:** Vercel doesn't support Python backends well

---

## 📦 PART 1: DEPLOY FRONTEND TO VERCEL

### Step 1: Prepare Frontend

**Already created:** `project/frontend/vercel.json`

```json
{
  "buildCommand": "npm run build",
  "outputDirectory": ".next",
  "installCommand": "npm install",
  "framework": "nextjs",
  "public": false
}
```

### Step 2: Deploy to Vercel

1. **Go to Vercel Dashboard**
   - Visit: https://vercel.com/new
   - Click "Import Project"

2. **Import from GitHub**
   - Select your repo: `Deep-Learning-Based-Multimodal-MRI-Analysis...`
   - Click "Import"

3. **Configure Build Settings** ⚠️ **CRITICAL**
   ```
   Framework Preset: Next.js
   Root Directory: project/frontend  ← MUST SET THIS
   Build Command: (leave default)
   Output Directory: (leave default)
   Install Command: (leave default)
   ```

4. **Click "Deploy"**

5. **Wait for build** (2-3 minutes)

6. **Your frontend is live!**
   - URL: `https://your-project-name.vercel.app`

---

## 🐍 PART 2: DEPLOY BACKEND TO RENDER

### Why Not Vercel for Backend?

Vercel has limited Python support (serverless functions only, not full FastAPI apps).
Render.com offers **free tier** with full FastAPI support.

### Step 1: Sign up for Render

1. Go to: https://render.com
2. Sign up with GitHub
3. Authorize Render to access your repo

### Step 2: Create New Web Service

1. Click "New +" → "Web Service"
2. Connect your GitHub repo
3. Select your repo

### Step 3: Configure Service

```
Name: neuroscope-backend
Runtime: Python 3
Region: (choose closest to you)
Branch: main
Root Directory: (leave empty)
Build Command: pip install -r requirements.txt
Start Command: gunicorn project.backend.main:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT
Instance Type: Free
```

### Step 4: Add Environment Variables (if needed)

```
PYTHON_VERSION=3.12.0
```

### Step 5: Deploy

1. Click "Create Web Service"
2. Wait for deployment (5-10 minutes first time)
3. Your backend is live!
   - URL: `https://neuroscope-backend.onrender.com`

---

## 🔗 PART 3: CONNECT FRONTEND TO BACKEND

### Update Frontend API Calls

**Create `.env.local` in `project/frontend/`:**

```bash
NEXT_PUBLIC_API_URL=https://neuroscope-backend.onrender.com
```

**Update API calls in frontend:**

```typescript
// Before
const response = await fetch('/api/data');

// After
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const response = await fetch(`${API_URL}/api/data`);
```

### Commit and Redeploy

```bash
git add .
git commit -m "Update API endpoint for production"
git push origin main
```

Vercel will auto-deploy the update.

---

## ✅ VERIFICATION CHECKLIST

### Frontend (Vercel)
- [ ] Build succeeds
- [ ] Homepage loads
- [ ] Documentation page loads
- [ ] 3 markdown files downloadable
- [ ] Navigation works
- [ ] Mobile responsive

### Backend (Render)
- [ ] Build succeeds
- [ ] `/docs` endpoint accessible
- [ ] API responds to requests
- [ ] CORS configured for frontend domain

---

## 🎯 FINAL URLS

After deployment, you'll have:

```
Frontend: https://your-project.vercel.app
Backend:  https://neuroscope-backend.onrender.com
API Docs: https://neuroscope-backend.onrender.com/docs
```

---

## 🐛 TROUBLESHOOTING

### Issue: Vercel build fails with "No public directory"

**Solution:** Make sure Root Directory is set to `project/frontend`

1. Go to Project Settings → General
2. Find "Root Directory"
3. Set to: `project/frontend`
4. Save and redeploy

---

### Issue: Backend doesn't start on Render

**Check:**
1. `requirements.txt` is in root directory ✓
2. Start command includes `project.backend.main:app`
3. Python version is 3.12.0

**Fix Start Command if needed:**
```bash
gunicorn project.backend.main:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT
```

---

### Issue: CORS errors when frontend calls backend

**Add to backend (`project/backend/main.py`):**

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-project.vercel.app"],  # Update with your Vercel URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 🆓 FREE TIER LIMITS

### Vercel (Free)
- Bandwidth: 100GB/month
- Builds: 6,000 minutes/month
- Serverless Function Execution: 100GB-hrs
- **Perfect for your frontend**

### Render (Free)
- 750 hours/month (enough for 1 always-on service)
- Auto-sleeps after 15 min inactivity
- Wakes up on request (takes ~30 seconds)
- **Good for demo/portfolio backend**

---

## 🚨 IMPORTANT NOTES

### Backend Will Sleep on Render Free Tier

The backend will go to sleep after 15 minutes of inactivity.

**First request will be slow** (~30 seconds to wake up).

**Solutions:**
1. Upgrade to paid tier ($7/month) for always-on
2. Use a cron job to ping every 14 minutes (keeps it awake)
3. Accept the sleep behavior for portfolio/demo

---

## 📱 MOBILE DEPLOYMENT

Both Vercel and Render automatically handle:
- ✅ HTTPS (SSL certificates)
- ✅ CDN (global distribution)
- ✅ Auto-scaling
- ✅ GitHub auto-deployments

Every `git push` to main will trigger redeployment!

---

## 🎉 DEPLOYMENT COMPLETE!

Once both are deployed:

1. **Test frontend:** Visit your Vercel URL
2. **Test backend:** Visit backend URL + `/docs`
3. **Test integration:** Frontend calling backend APIs
4. **Share:** Send portfolio links!

---

## 🔄 CONTINUOUS DEPLOYMENT

Both platforms auto-deploy on git push:

```bash
# Make changes
git add .
git commit -m "Update feature"
git push origin main

# Vercel rebuilds frontend automatically
# Render rebuilds backend automatically
```

No manual steps needed! 🎯

---

**Status:** Ready to deploy!  
**Time to deploy:** 15-20 minutes total  
**Cost:** $0 (both free tiers)
