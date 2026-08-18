# Vision-Driven Site Intelligence

**Equipment Utilisation & Safety Monitoring — Remote & Cloud-Ready Industrial Control Center**

A software-only computer vision platform that transforms ordinary camera feeds (webcam, WebRTC browser camera, or video clips) into actionable site intelligence. It automatically detects and tracks workers, monitors restricted danger zones, estimates visual asset activity, identifies safety events, calculates operational metrics, and generates AI executive summaries.

---

## Live Public HTTPS Demo
🔗 **Public Demo URL**: `https://vision-site-intelligence.streamlit.app` *(Deployable to Streamlit Community Cloud)*

---

## Tech Stack & Architecture

- **Core**: Python 3.10+, Streamlit 1.30+
- **Computer Vision**: Ultralytics YOLOv8n (`yolov8n.pt`)
- **Object Tracking**: IoU & Centroid Tracker
- **Remote Camera Access**: `streamlit-webrtc` with Google STUN server (`stun:stun.l.google.com:19302`)
- **Analytics & Visuals**: Plotly Express & Custom Dark Navy CSS Design System
- **Database & Persistence**: SQLite3 (`data/site.db`)
- **Executive AI Reports**: Gemini 2.0 Flash (`google-generativeai`) with automatic rule-based fallback

---

## Important Technical Disclaimer

> ⚠️ **IMPORTANT CLAIM**: This MVP performs vision-based activity and utilisation estimation directly from visual camera observations. It does **NOT** measure physical equipment telemetry (e.g., engine CAN bus, fuel consumption, or hydraulic pressure sensors).

---

## Remote Camera Architecture

```
[ User Browser (Laptop / Android Chrome) ]
                    │
            Camera Permission
                    │
             Browser Webcam
                    │
           WebRTC Stream (HTTPS)
                    │
          Google STUN Server
                    │
     [ Streamlit Community Cloud ]
                    │
           WebRTCVideoProcessor
                    │
             YOLOv8n Detector
                    │
             Simple Tracker
                    │
             Safety Engine  ───►  Zone Violations & Events
                    │
            Activity Engine ───►  Active / Idle Utilisation
                    │
            Executive UI & Gemini AI Summary
```

---

## Remote Usage (No Installation Required)

1. Open the public HTTPS deployment URL in any browser (**Chrome desktop**, **Edge**, or **Android Chrome**).
2. Go to **Live Monitor** or **Dashboard**.
3. Select **Browser Camera (WebRTC)** as the Camera Source.
4. Click **START**.
5. Grant camera permission when prompted by your browser.
6. The live feed will display real-time bounding boxes, track IDs, danger zone boundaries, and safety metrics.

---

## Local Development & Setup

```bash
# 1. Clone repository
git clone https://github.com/<YOUR_GITHUB_USERNAME>/vision_site_intelligence.git
cd vision_site_intelligence

# 2. Create virtual environment
python -m venv .venv

# Windows activation:
.venv\Scripts\activate
# Mac/Linux activation:
# source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Set Gemini API Key
cp .env.example .env
# Edit .env and set GEMINI_API_KEY=your_key_here

# 5. Run application locally
streamlit run app.py
```

---

## Cloud Deployment (Streamlit Community Cloud)

1. Push repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and create a **New App**.
3. Select your repository, set main file to `app.py`, and set Python version to `3.10`.
4. Add `GEMINI_API_KEY` under **Secrets** if available.
5. Click **Deploy!**

Detailed deployment notes are available in [docs/deployment.md](file:///c:/Claude%20Hackathon/vision_site_intelligence/docs/deployment.md).

---

## Camera Privacy Notice
> 🔒 **Camera Privacy**: Camera frames are processed in-memory for real-time computer vision inference. Raw camera video is not recorded or stored by the application.
