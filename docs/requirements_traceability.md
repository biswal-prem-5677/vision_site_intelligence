# Requirements Traceability Matrix

## Hackathon Problem Statement & Solution Mapping

This document explicitly maps the core hackathon problem statements to the verified software MVP features implemented in the **Vision-Driven Site Intelligence** platform.

---

### Problem 1: Heavy Asset Idle Time & Schedule/Cost Risk
> *"Large sites manage hundreds of heavy assets; idle time adds cost and schedule risk."*

- **Feature Implementation**: Visual Asset Activity + Utilisation Tracking (`app/services/activity_engine.py`)
- **How It Works**:
  - Automatically filters detections to equipment proxy classes (`car`, `truck`, `bus`, `motorcycle`, `bicycle`).
  - Computes Euclidean movement per track ID across frames to classify state as **ACTIVE** or **IDLE**.
  - Calculates real-time asset utilisation percentage: $\text{Utilisation} = \frac{\text{Active Time}}{\text{Active Time} + \text{Idle Time}} \times 100\%$.
  - Displays asset-by-asset status, active seconds, idle seconds, and utilisation on both the Executive Dashboard and dedicated **Assets** control page.
- **Traceability**: `app/services/activity_engine.py` $\rightarrow$ `app/database.py:upsert_asset_metrics` $\rightarrow$ `app.py:render_assets()`

---

### Problem 2: Worker Safety Incidents & Human Error
> *"Worker safety incidents and human-error remain prevalent despite existing controls."*

- **Feature Implementation**: Real-Time Danger Zone Monitoring & Bounded Risk Scoring (`app/services/safety_engine.py`, `app/services/risk_engine.py`)
- **How It Works**:
  - Defines spatial restricted boundaries using normalized polygon coordinates (`DANGER_ZONE_POLYGON`).
  - Performs ray-casting point-in-polygon checks on worker centroids in real-time.
  - Implements state-based 0–100 Safety Score model starting at 100:
    - Active worker in danger zone: $-20$ (1st worker), $-5$ (additional workers).
    - Automatically updates site risk level: **LOW** (90–100), **MODERATE** (70–89), **HIGH** (0–69).
    - Resets dynamically when workers clear the zone.
- **Traceability**: `app/services/safety_engine.py:check_safety` $\rightarrow$ `app/services/risk_engine.py:calculate_safety_score` $\rightarrow$ `app.py:render_safety()`

---

### Problem 3: Labor-Intensive & Reactive Manual Monitoring
> *"Manual monitoring and reporting are labour-intensive and reactive."*

- **Feature Implementation**: Automated Visual Event Engine & Event Lifecycle (`app/services/safety_engine.py`)
- **How It Works**:
  - Replaces manual video watching with automated visual event detection.
  - Manages event lifecycle: `ENTRY` $\rightarrow$ `SUSTAINED` (cooldown interval) $\rightarrow$ `CLEARED`.
  - Prevents frame-by-frame alert spamming through track state management (`zone_alert_sent`, `zone_last_sustained`).
  - Logs all events to SQLite database (`events` table) with timestamp, severity (`HIGH`/`MEDIUM`/`LOW`), track ID, and descriptive message.
- **Traceability**: `app/services/safety_engine.py:flush_events` $\rightarrow$ `app/database.py:insert_event` $\rightarrow$ `app.py:render_events()`

---

### Problem 4: Reactive Management Reporting
> *"Reporting is manual, slow, and reactive."*

- **Feature Implementation**: Gemini AI Site Intelligence Summary (`app/services/gemini_service.py`)
- **How It Works**:
  - System extracts structured CV metrics (worker count, active violations, equipment utilisation, safety score, risk level).
  - Passes structured metrics to Gemini AI (`gemini-2.0-flash`) to generate concise executive summaries with priority actions.
  - Features automatic rule-based fallback if Gemini API key is not configured or offline, ensuring zero UI breakage.
- **Traceability**: `app/services/gemini_service.py:generate_summary` $\rightarrow$ `app.py:render_ai_reports()`

---

## Technical Constraints & Compliance

- **Hardware**: Software-only MVP using laptop webcam (or video file mode).
- **No External Hardware**: Zero Arduino, ESP32, Raspberry Pi, IoT sensors, RFID, or external cameras required.
- **Accuracy**: Class names match underlying YOLO model (COCO classes) without manufacturing false equipment identities.
