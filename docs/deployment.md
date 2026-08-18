# Streamlit Community Cloud & WebRTC Deployment Guide

This document details the public HTTPS deployment workflow for **Vision-Driven Site Intelligence** using **Streamlit Community Cloud** and **WebRTC browser camera access**.

---

## 1. Architecture Overview

- **Frontend / UI**: Streamlit 1.30+ with WebRTC browser camera integration (`streamlit-webrtc`).
- **Global Camera Runtime**: Persistent single-instance WebRTC stream mounted in top application shell (`global_webrtc_streamer`).
- **Remote Camera Access**: WebRTC media stream protocol with Google STUN server (`stun:stun.l.google.com:19302`) for NAT traversal.
- **Inference Pipeline**: YOLOv8n object detection + IoU centroid tracker + multi-zone safety engine + activity engine.
- **Persistence**: SQLite session database (`data/site.db`).
- **AI Intelligence**: Gemini 2.0 Flash (`st.secrets["GEMINI_API_KEY"]`) with automatic rule-based fallback.

---

## 2. GitHub Repository Requirements

1. **Repository**: `biswal-prem-5677/vision_site_intelligence`
2. **Branch**: `main`
3. **Entry point**: `app.py`
4. **Dependencies**: `requirements.txt` containing `streamlit-webrtc`, `av`, `opencv-python-headless`, `ultralytics`, `plotly`, `google-genai`.

---

## 3. Streamlit Community Cloud Deployment Steps

### Step 1: Push Repository to GitHub
```bash
git add .
git commit -m "Refactor: Final productization & architectural refactor"
git push origin master:main
```

### Step 2: Deploy on Streamlit Community Cloud
1. Log in to [share.streamlit.io](https://share.streamlit.io).
2. Click **New app**.
3. Select your GitHub repository (`biswal-prem-5677/vision_site_intelligence`).
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
- **Gemini Secret Access**: Resolved automatically from Streamlit Secrets or environment variables.

---

## 5. Camera Permission & Browser Usage

### Desktop & Mobile Browsers
1. Open the public HTTPS deployment URL.
2. Under the top persistent transport panel, click **START**.
3. When prompted by the browser:
   > *"share.streamlit.io wants to use your camera"*
   Click **Allow**.
4. Navigate between any of the 9 pages (**Dashboard**, **Live Monitor**, **Safety**, **Safety Zones**, **Assets**, **Events**, **Analytics**, **AI Reports**, **Settings**).
5. The live camera feed remains active and streaming continuously without asking for permission again or reloading the tracker.

---

## 6. Camera Privacy Notice
> 🔒 **Privacy Guarantee**: Camera frames are processed in-memory for real-time computer vision inference. Raw video is never stored, recorded, or uploaded to any server.
