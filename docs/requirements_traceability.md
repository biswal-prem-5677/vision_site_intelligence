# Requirements Traceability Matrix

This document maps user problem statements, hackathon constraints, and operational goals directly to codebase implementations.

---

## Traceability Mapping

| Problem Statement / Constraint | Technical Implementation | Code File(s) | Verification Method |
|---|---|---|---|
| **Software-Only Constraint** | Laptop webcam / WebRTC browser camera feed as single live sensor. Zero IoT or external hardware needed. | [webrtc_engine.py](file:///c:/Claude%20Hackathon/vision_site_intelligence/app/services/webrtc_engine.py) & [camera.py](file:///c:/Claude%20Hackathon/vision_site_intelligence/app/services/camera.py) | Verified live stream in browser via WebRTC |
| **Global Camera Session** | Persistent WebRTC component mounted in top shell container (`global_webrtc_streamer`). Single camera lifecycle across all 9 pages. | [app.py](file:///c:/Claude%20Hackathon/vision_site_intelligence/app.py) | Verified camera stays LIVE during page navigation |
| **Multi-Zone Safety Monitoring** | Evaluates 4 Safety Zones (`Crane Swing Area`, `Excavation Zone`, `Restricted Personnel Area`, `Equipment Operating Area`) per `(zone_id, track_id)`. | [safety_engine.py](file:///c:/Claude%20Hackathon/vision_site_intelligence/app/services/safety_engine.py) & [config.py](file:///c:/Claude%20Hackathon/vision_site_intelligence/app/config.py) | Verified independent zone entry/exit events |
| **Visual Polygon Zone Editor** | Dedicated **Safety Zones** page featuring interactive normalized vertex sliders and live camera frame boundary preview. | [app.py](file:///c:/Claude%20Hackathon/vision_site_intelligence/app.py#L740-L790) | Verified live preview canvas draws updated polygon |
| **Asset Utilisation Tracking** | Visual motion tracking calculates active vs idle duration and computes utilisation % without physical sensors. | [activity_engine.py](file:///c:/Claude%20Hackathon/vision_site_intelligence/app/services/activity_engine.py) | Verified utilisation % changes based on object motion |
| **Bounded Safety Scoring** | Dynamic state-based 0–100 safety score assigning `LOW`, `MODERATE`, or `HIGH` site risk levels. | [risk_engine.py](file:///c:/Claude%20Hackathon/vision_site_intelligence/app/services/risk_engine.py) | Verified score drops during active violations and resets when clear |
| **No Event Spam** | Event state machine (`ENTRY` $\rightarrow$ `SUSTAINED` every 5s $\rightarrow$ `CLEARED`) enforces per-track cooldowns. | [safety_engine.py](file:///c:/Claude%20Hackathon/vision_site_intelligence/app/services/safety_engine.py) | Verified exactly 1 entry event, 5s sustained alerts, 1 exit event |
| **AI Executive Summaries** | Formats structured telematics into context payload for Gemini AI API with rule-based fallback when offline. | [gemini_service.py](file:///c:/Claude%20Hackathon/vision_site_intelligence/app/services/gemini_service.py) | Verified executive summary generation |
| **SQLite Data Persistence** | Stores events, detections, and asset metrics in SQLite database (`data/site.db`) with confirmation clear control. | [database.py](file:///c:/Claude%20Hackathon/vision_site_intelligence/app/database.py) | Verified data persists across restarts and clears cleanly |
| **Industrial Command UI** | Dark graphite/navy (`#090d12`) CSS design system with top command header, status badges, and hero canvas layout. | [app.py](file:///c:/Claude%20Hackathon/vision_site_intelligence/app.py) | Visual UI inspection |
