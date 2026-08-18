# Streamlit Community Cloud & WebRTC Deployment Guide

This document details the public HTTPS deployment workflow for **Vision-Driven Site Intelligence** using **Streamlit Community Cloud** and **WebRTC browser camera access**.

---

## 1. Architecture Overview

- **Frontend / UI**: Streamlit 1.30+ with WebRTC browser camera integration (`streamlit-webrtc`).
- **Remote Camera Access**: WebRTC media stream protocol with Google STUN server (`stun:stun.l.google.com:19302`) for NAT traversal.
- **Inference Pipeline**: YOLOv8n object detection + IoU centroid tracker + safety/activity engines.
- **Persistence**: SQLite session database (`data/site.db`).
- **AI Intelligence**: Gemini 2.0 Flash (`st.secrets["GEMINI_API_KEY"]`) with automatic rule-based fallback.

---

## 2. GitHub Repository Requirements

1. **Repository**: `vision_site_intelligence` (or your GitHub repo name).
2. **Branch**: `main`
3. **Entry point**: `app.py`
4. **Dependencies**: `requirements.txt` containing `streamlit-webrtc`, `av`, `opencv-python-headless`, `ultralytics`, `plotly`, `google-generativeai`.

---

## 3. Streamlit Community Cloud Deployment Steps

### Step 1: Push Repository to GitHub
```bash
git init
git add .
git commit -m "Deploy: Vision-Driven Site Intelligence with WebRTC browser camera"
git branch -M main
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/vision_site_intelligence.git
git push -u origin main
```

### Step 2: Deploy on Streamlit Community Cloud
1. Log in to [share.streamlit.io](https://share.streamlit.io).
2. Click **New app**.
3. Select your GitHub repository (`<YOUR_GITHUB_USERNAME>/vision_site_intelligence`).
4. Set **Main file path** to `app.py`.
5. Under **Advanced settings...**:
   - Set **Python version** to `3.10` or higher.
   - Open **Secrets** tab and paste your Gemini API Key:
     ```toml
     GEMINI_API_KEY = "AIzaSy..."
     ```
6. Click **Deploy!**

---

## 4. WebRTC / STUN / Secrets Configuration

- **STUN Server Configuration**: Automatically configured in `app/services/webrtc_engine.py`:
  ```python
  RTC_CONFIGURATION = RTCConfiguration(
      {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
  )
  ```
- **Gemini Secret Access**: Resolved automatically from Streamlit Secrets:
  ```python
  api_key = st.secrets.get("GEMINI_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")
  ```

---

## 5. Camera Permission & Browser Usage

### Desktop Browsers (Chrome, Edge, Firefox, Safari)
1. Open the public HTTPS deployment URL.
2. Navigate to **Live Monitor** or stay on **Dashboard**.
3. Click **START** under the camera panel.
4. When prompted by the browser:
   > *"share.streamlit.io wants to use your camera"*
   Click **Allow**.
5. The live camera feed starts processing real-time object detection and safety zone monitoring.

### Mobile Browsers (Android Chrome, iOS Safari)
1. Open the public HTTPS deployment URL on your smartphone.
2. Tap **START** in the camera card.
3. Grant camera permissions when prompted.
4. The responsive UI automatically adapts layout for touch screens.

---

## 6. Camera Privacy Notice
> 🔒 **Privacy Guarantee**: Camera frames are processed in-memory for real-time computer vision inference. Raw video is never stored, recorded, or uploaded to any server.

---

## 7. Troubleshooting

| Issue | Cause | Resolution |
|---|---|---|
| Camera permission denied | Browser blocked camera access | Click camera lock icon in browser URL bar $\rightarrow$ Permissions $\rightarrow$ Allow Camera $\rightarrow$ Refresh page. |
| WebRTC connection failed | Strict firewall/NAT blocking STUN | Ensure WebRTC is enabled in browser or test on cellular network. |
| Gemini API error | Key missing or invalid | Add `GEMINI_API_KEY = "your_key"` to Streamlit Cloud Secrets. Rule-based summary works automatically as fallback. |
| Slow FPS on Cloud CPU | Cloud server vCPU throttling | Inference FPS throttles automatically to maintain UI responsiveness (~5–10 FPS on cloud). |
