# Vision-Driven Site Intelligence

**Equipment Utilisation & Safety Monitoring — Industrial Computer-Vision Command Center**

A unified computer vision platform that transforms camera feeds (webcam, WebRTC browser camera, or video clips) into actionable operational intelligence. It automatically detects and tracks workers, monitors multi-zone safety boundaries (`Crane Swing Area`, `Excavation Zone`, `Restricted Personnel Area`, `Equipment Operating Area`), provides a visual polygon zone editor, estimates visual asset activity, logs safety events, and generates executive AI reports.

---

## Technical Stack & Architecture

- **Core Framework**: Python 3.10+, Streamlit 1.30+
- **Computer Vision**: Ultralytics YOLOv8n (`yolov8n.pt`)
- **Object Tracking**: Centroid & IoU Tracker
- **Remote Camera Access**: `streamlit-webrtc` with Google STUN server (`stun:stun.l.google.com:19302`)
- **Safety Engine**: Ray-Casting Multi-Zone Point-in-Polygon Engine
- **Analytics & Visuals**: Plotly Express & Custom Industrial Dark Theme CSS (`#090d12`)
- **Database & Persistence**: SQLite3 (`data/site.db`)
- **Executive AI Reports**: Gemini 2.0 Flash with automatic rule-based fallback

---

## System Architecture

```
                                GLOBAL APPLICATION SHELL
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    │                                             │
         Persistent WebRTC Transport                     Global Camera Runtime
                    │                                             │
             YOLOv8n Detector                              Centroid Tracker
                    │                                             │
       Multi-Zone Safety Engine                         Activity & Utilisation Engine
  (Crane Swing, Excavation, Restricted)                 (Active / Idle Classification)
                    │                                             │
                    └──────────────────────┬──────────────────────┘
                                           │
                             Industrial Command Center UI
                 (Dashboard, Live Monitor, Safety Zones, Analytics, AI Reports)
```

---

## Core Capabilities & Honest Limitations

1. **Global Camera Runtime**: Camera session is maintained as a single global application resource. Navigating between pages never restarts the WebRTC stream, resets tracker IDs, or reloads models.
2. **Multi-Zone Safety System**: Evaluates 4 categorized Safety Zones simultaneously with independent event lifecycles (`ENTRY` $\rightarrow$ `SUSTAINED` $\rightarrow$ `CLEARED`).
3. **Interactive Visual Zone Editor**: Dedicated **Safety Zones** page allows visual boundary editing over a live camera preview canvas.
4. **Detected Assets (COCO Proxies)**: Equipment tracking uses COCO vehicle classes (`Car`, `Truck`, `Bus`, `Motorcycle`, `Bicycle`) as visual proxies for heavy machinery. Visual activity and idle duration are computed directly from motion.
5. **PPE Detection Disclaimer**: Standard YOLOv8n COCO models detect person bounding boxes. Construction PPE compliance (helmets/vests) requires fine-tuned construction dataset weights; the interface honestly indicates `PPE DETECTION: Not configured` rather than faking compliance metrics.

---

## Local Development & Setup

```bash
# 1. Clone repository
git clone https://github.com/biswal-prem-5677/vision_site_intelligence.git
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
3. Select repository `biswal-prem-5677/vision_site_intelligence`, set branch to `main`, main file to `app.py`.
4. Add `GEMINI_API_KEY` under **Secrets** if available.
5. Click **Deploy!**

Detailed deployment notes are available in [docs/deployment.md](file:///c:/Claude%20Hackathon/vision_site_intelligence/docs/deployment.md).

---

## Camera Privacy Notice
> 🔒 **Camera Privacy**: Camera frames are processed in-memory for real-time computer vision inference. Raw camera video is not recorded or stored by the application.
