"""
Site Intelligence Dashboard — Industrial Operations Command Center
Real-time vision monitoring, multi-zone safety intelligence, and operational analytics.
"""

import os
import time
import copy
import cv2
import numpy as np
import streamlit as st
from dotenv import load_dotenv

from app.config import (
    SAFETY_ZONES,
    DANGER_ZONE_POLYGON,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    PROCESS_FPS,
    CONFIDENCE_THRESHOLD,
    EQUIPMENT_CLASSES,
    PENALTY_DANGER_ZONE,
    PENALTY_NO_HELMET,
    PENALTY_NO_VEST,
    PENALTY_REPEATED_VIOLATION,
)
from app.database import (
    init_db,
    get_recent_events,
    get_all_asset_metrics,
    clear_all_data,
)
from app.schemas import SiteMetrics
from app.services.camera import Camera
from app.services.detector import Detector
from app.services.tracker import SimpleTracker
from app.services.safety_engine import check_safety, flush_events
from app.services.activity_engine import (
    update_activity,
    get_asset_summary,
    get_display_label,
    _is_person,
    _is_equipment,
)
from app.services.risk_engine import calculate_safety_score
from app.services.gemini_service import GeminiService

# ─────────────────────────────────────────────────────────────────────────────
# 1. Page Setup & Initialization
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()
init_db()

st.set_page_config(
    page_title="Site Intelligence | Industrial Command Center",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _get_gemini_api_key() -> str:
    key = ""
    try:
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    if not key:
        key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        for env_file in [".env", ".env.example"]:
            if os.path.exists(env_file):
                try:
                    with open(env_file, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip().startswith("GEMINI_API_KEY"):
                                parts = line.strip().split("=", 1)
                                if len(parts) == 2:
                                    extracted = parts[1].strip("'\" \r\n")
                                    if extracted and "your_gemini" not in extracted.lower():
                                        key = extracted
                                        break
                except Exception:
                    pass
            if key:
                break
    return key


# Session state defaults
_defaults = {
    "camera_source": "Browser Camera (WebRTC)",
    "camera_running": False,
    "frame_buffer": None,
    "last_frame_time": 0.0,
    "session_events": [],
    "metrics": SiteMetrics(),
    "ai_summary": "Start camera to begin real-time site monitoring.",
    "last_summary_time": 0.0,
    "fps": 0.0,
    "inference_fps": 0.0,
    "page": "Dashboard",
    "active_violations": 0,
    "confidence_threshold": 0.5,
    "idle_threshold": 3.0,
    "danger_zone_enabled": True,
    "safety_zones": copy.deepcopy(SAFETY_ZONES),
    "editing_zone_id": "crane_swing",
}
for key, val in _defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

if "cam" not in st.session_state:
    st.session_state.cam = None
if "det" not in st.session_state:
    st.session_state.det = None
if "trk" not in st.session_state:
    st.session_state.trk = None
if "gemini_svc" not in st.session_state:
    st.session_state.gemini_svc = None
if "_last_process_time" not in st.session_state:
    st.session_state._last_process_time = 0.0
if "_frame_skip" not in st.session_state:
    st.session_state._frame_skip = 0


# ─────────────────────────────────────────────────────────────────────────────
# 2. High Density Industrial Command Center CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    :root {
        --bg-primary: #090d12;
        --bg-secondary: #121820;
        --bg-card: #18202c;
        --bg-hover: #222c3c;
        --border: #253040;
        --border-light: #34445c;
        --text-primary: #e8eaed;
        --text-secondary: #a0aec0;
        --text-muted: #64748b;
        --accent-blue: #3b82f6;
        --accent-green: #10b981;
        --accent-amber: #f59e0b;
        --accent-red: #ef4444;
        --accent-purple: #8b5cf6;
    }

    * { box-sizing: border-box; }

    body {
        background: var(--bg-primary);
        color: var(--text-primary);
        font-family: 'Inter', -apple-system, system-ui, sans-serif;
    }

    .main .block-container {
        padding: 0.8rem 1.2rem;
        max-width: 100%;
    }

    /* Target Streamlit Native Bordered Containers */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
        padding: 16px !important;
        margin-bottom: 10px !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        gap: 0.5rem !important;
    }

    /* Command Header */
    .command-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.8rem 1.2rem;
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: 6px;
        margin-bottom: 1rem;
    }
    .command-title {
        font-size: 1.2rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        color: var(--text-primary);
        text-transform: uppercase;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .command-subtitle {
        font-size: 0.72rem;
        color: var(--text-muted);
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-top: 2px;
    }
    .status-badge-container {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .pill-green { background: rgba(16, 185, 129, 0.15); color: var(--accent-green); border: 1px solid rgba(16, 185, 129, 0.3); }
    .pill-amber { background: rgba(245, 158, 11, 0.15); color: var(--accent-amber); border: 1px solid rgba(245, 158, 11, 0.3); }
    .pill-red { background: rgba(239, 68, 68, 0.15); color: var(--accent-red); border: 1px solid rgba(239, 68, 68, 0.3); }
    .pill-gray { background: rgba(100, 116, 139, 0.15); color: var(--text-muted); border: 1px solid rgba(100, 116, 139, 0.3); }

    .panel-title {
        font-size: 0.72rem;
        font-weight: 700;
        color: var(--text-muted);
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }

    /* KPI Cards */
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 12px 14px;
    }
    .metric-label {
        font-size: 0.68rem;
        font-weight: 600;
        color: var(--text-muted);
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 800;
        color: var(--text-primary);
        line-height: 1.2;
        margin: 4px 0 2px;
    }
    .metric-sub {
        font-size: 0.68rem;
        color: var(--text-secondary);
    }

    /* Event Rows */
    .event-row {
        padding: 8px 10px;
        border-radius: 4px;
        margin-bottom: 6px;
        font-size: 0.8rem;
        border-left: 3px solid var(--border-light);
        background: var(--bg-secondary);
    }
    .event-row-high { border-left-color: var(--accent-red); background: rgba(239, 68, 68, 0.08); }
    .event-row-medium { border-left-color: var(--accent-amber); background: rgba(245, 158, 11, 0.08); }
    .event-row-low { border-left-color: var(--accent-green); background: rgba(16, 185, 129, 0.08); }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 2px 6px;
        border-radius: 3px;
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .badge-green { background: rgba(16, 185, 129, 0.2); color: var(--accent-green); }
    .badge-amber { background: rgba(245, 158, 11, 0.2); color: var(--accent-amber); }
    .badge-red { background: rgba(239, 68, 68, 0.2); color: var(--accent-red); }
    .badge-blue { background: rgba(59, 130, 246, 0.2); color: var(--accent-blue); }
    .badge-gray { background: rgba(100, 116, 139, 0.2); color: var(--text-muted); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Global Camera Runtime & Shared Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _start_camera():
    if st.session_state.camera_running:
        return
    try:
        if st.session_state.cam is None:
            st.session_state.cam = Camera()
        if st.session_state.det is None:
            st.session_state.det = Detector()
        if st.session_state.trk is None:
            st.session_state.trk = SimpleTracker()

        if st.session_state.gemini_svc is None:
            api_key = _get_gemini_api_key()
            if api_key and "your_gemini" not in api_key.lower() and len(api_key) > 10:
                st.session_state.gemini_svc = GeminiService(api_key)

        if st.session_state.cam.start():
            st.session_state.camera_running = True
            st.session_state.session_events = []
            st.session_state.trk = SimpleTracker()
            st.session_state.last_frame_time = 0.0
            st.session_state.last_summary_time = 0.0
            st.session_state._last_process_time = 0.0
            st.session_state.ai_summary = "Camera active — visit AI Reports for executive summary."
        else:
            st.error("Cannot open local camera source.")
    except Exception as e:
        st.error(f"Camera error: {e}")


def _stop_camera():
    if st.session_state.cam:
        st.session_state.cam.stop()
    st.session_state.camera_running = False
    st.session_state.frame_buffer = None
    st.session_state.active_violations = 0


def _clear_data():
    clear_all_data()
    st.session_state.session_events = []
    st.session_state.metrics = SiteMetrics()
    st.session_state.ai_summary = "Data cleared. Start camera to begin monitoring."
    st.session_state.trk = SimpleTracker()
    st.session_state.frame_buffer = None
    st.session_state.active_violations = 0
    st.session_state.fps = 0.0
    st.session_state.inference_fps = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 4. Global Persistent Shell Header & WebRTC Mount
# ─────────────────────────────────────────────────────────────────────────────
# Persistent Top Command Header
sys_status_class = "pill-green" if st.session_state.camera_running else "pill-gray"
sys_status_text = "● SYSTEM LIVE" if st.session_state.camera_running else "○ SYSTEM READY"
cam_status_text = f"CAMERA: {st.session_state.camera_source.split()[0]}"

st.markdown(f"""
<div class="command-header">
    <div>
        <div class="command-title">🏗️ SITE INTELLIGENCE COMMAND CENTER</div>
        <div class="command-subtitle">Real-Time Computer Vision • Multi-Zone Safety Engine • Operational Analytics</div>
    </div>
    <div class="status-badge-container">
        <span class="status-pill {sys_status_class}">{sys_status_text}</span>
        <span class="status-pill pill-blue">AI READY</span>
        <span class="status-pill pill-gray">{cam_status_text}</span>
    </div>
</div>
""", unsafe_allow_html=True)


# Persistent WebRTC Component Mount (Single instance across application navigation)
if st.session_state.get("camera_source") == "Browser Camera (WebRTC)":
    try:
        from streamlit_webrtc import webrtc_streamer, WebRtcMode
        from app.services.webrtc_engine import WebRTCVideoProcessor, RTC_CONFIGURATION

        with st.container(border=True):
            st.caption("PERSISTENT WEBRTC LIVE CAMERA STREAM (HD 720p)")
            webrtc_ctx = webrtc_streamer(
                key="global_webrtc_streamer",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                video_processor_factory=WebRTCVideoProcessor,
                media_stream_constraints={
                    "video": {
                        "width": {"ideal": 1280, "max": 1920},
                        "height": {"ideal": 720, "max": 1080},
                        "frameRate": {"ideal": 30},
                    },
                    "audio": False,
                },
                async_processing=True,
            )

        if webrtc_ctx and webrtc_ctx.state.playing and webrtc_ctx.video_processor:
            proc = webrtc_ctx.video_processor
            proc.danger_zone_enabled = st.session_state.danger_zone_enabled
            proc.zones = list(st.session_state.safety_zones)
            with proc.lock:
                st.session_state.camera_running = True
                st.session_state.metrics = proc.metrics
                st.session_state.active_violations = proc.active_violations
                st.session_state.fps = proc.fps
                st.session_state.inference_fps = proc.inference_fps
                if proc.session_events:
                    st.session_state.session_events = list(proc.session_events)
        else:
            if st.session_state.camera_source == "Browser Camera (WebRTC)":
                st.session_state.camera_running = False
    except Exception as e:
        st.error(f"WebRTC Engine Initialization: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Sidebar Navigation & Global Camera Control Center
# ─────────────────────────────────────────────────────────────────────────────
def nav_page(label: str, icon: str) -> bool:
    active = st.session_state.page == label
    if st.button(f"{icon} {label}", key=f"nav_{label}"):
        st.session_state.page = label
        st.rerun()
    return active


with st.sidebar:
    st.markdown(
        '<div class="panel-title" style="padding:4px 0 8px;">NAVIGATION</div>',
        unsafe_allow_html=True,
    )
    nav_page("Dashboard", "📊")
    nav_page("Live Monitor", "📹")
    nav_page("Safety", "🛡️")
    nav_page("Safety Zones", "📍")
    nav_page("Assets", "🚜")
    nav_page("Events", "📋")
    nav_page("Analytics", "📈")
    nav_page("AI Reports", "🤖")
    nav_page("Settings", "⚙️")

    st.markdown("---")

    # Global Camera Control Center
    with st.container(border=True):
        st.markdown('<div class="panel-title">GLOBAL CAMERA CONTROL</div>', unsafe_allow_html=True)

        st.session_state.camera_source = st.selectbox(
            "Source",
            ["Browser Camera (WebRTC)", "Local Webcam / Demo Video"],
            index=0 if st.session_state.camera_source == "Browser Camera (WebRTC)" else 1,
            label_visibility="collapsed",
        )

        if st.session_state.camera_running:
            st.markdown('<span style="color:#10b981; font-size:0.8rem; font-weight:700;">● CAMERA STREAM ACTIVE</span>', unsafe_allow_html=True)
            st.caption(f"Source: {st.session_state.camera_source.split()[0]} | AI: {st.session_state.fps:.1f} FPS")
            if st.session_state.camera_source == "Local Webcam / Demo Video":
                if st.button("⏹ Stop Camera", key="btn_global_stop"):
                    _stop_camera()
                    st.rerun()
        else:
            st.markdown('<span style="color:#64748b; font-size:0.8rem; font-weight:700;">○ CAMERA OFFLINE</span>', unsafe_allow_html=True)
            if st.session_state.camera_source == "Local Webcam / Demo Video":
                if st.button("▶ Start Camera", key="btn_global_start", type="primary"):
                    _start_camera()
                    st.rerun()
            else:
                st.caption("Click START in persistent WebRTC stream box above to grant camera access.")

        st.caption("Master Zone Monitoring")
        st.session_state.danger_zone_enabled = st.checkbox(
            "Enable Zone Engine",
            value=st.session_state.danger_zone_enabled,
        )

    st.markdown("---")
    st.caption("Data Reset")
    confirm_clear = st.checkbox("Confirm data reset", value=False, key="chk_confirm_reset")
    if st.button("Clear All Data", disabled=not confirm_clear):
        _clear_data()
        st.success("All database and session data cleared!")
        time.sleep(0.3)
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# 6. Process One Frame (Local OpenCV Fallback)
# ─────────────────────────────────────────────────────────────────────────────
def process_frame() -> bool:
    if not st.session_state.camera_running:
        return False

    if st.session_state.get("camera_source") == "Browser Camera (WebRTC)":
        return True

    cam = st.session_state.cam
    det = st.session_state.det
    trk = st.session_state.trk

    if cam is None or det is None or trk is None or not getattr(cam, "running", False):
        return False

    ret, frame = cam.read()
    if not ret or frame is None:
        return False

    now = time.time()

    if st.session_state.last_frame_time > 0:
        dt = now - st.session_state.last_frame_time
    else:
        dt = 1.0 / PROCESS_FPS
    st.session_state.last_frame_time = now
    raw_fps = 1.0 / dt if dt > 0 else 0.0
    st.session_state.fps = round(raw_fps, 1)

    elapsed = now - st.session_state._last_process_time
    min_interval = 1.0 / PROCESS_FPS
    if elapsed < min_interval:
        return True

    st.session_state._last_process_time = now
    h, w = frame.shape[:2]

    detections = det.detect(frame)
    tracked = trk.update(detections, w, h)
    update_activity(tracked, dt, w, h)

    active_violating_workers = 0
    active_zone_violations = 0
    if st.session_state.danger_zone_enabled:
        safety_events, active_violating_workers, active_zone_violations = check_safety(
            tracked, w, h, now, zones=st.session_state.safety_zones
        )
        if safety_events:
            flush_events(safety_events)
            st.session_state.session_events.extend(safety_events)
            if len(st.session_state.session_events) > 200:
                st.session_state.session_events = st.session_state.session_events[-200:]
    st.session_state.active_violations = active_violating_workers

    ann = frame.copy()

    # Draw Safety Zones
    if st.session_state.danger_zone_enabled and st.session_state.safety_zones:
        for zone in st.session_state.safety_zones:
            if not zone.get("enabled", True):
                continue
            pts = np.array(
                [[int(p[0] * w), int(p[1] * h)] for p in zone["polygon"]],
                dtype=np.int32,
            )
            bgr = zone.get("color_bgr", (0, 0, 255))
            overlay = ann.copy()
            cv2.fillPoly(overlay, [pts], (int(bgr[0] * 0.3), int(bgr[1] * 0.3), int(bgr[2] * 0.3)))
            cv2.addWeighted(overlay, 0.35, ann, 0.65, 0, ann)
            cv2.polylines(ann, [pts], isClosed=True, color=bgr, thickness=2)
            cv2.putText(
                ann, zone.get("name", "ZONE").upper(),
                (pts[0][0] + 5, max(pts[0][1] - 6, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, bgr, 2,
            )

    # Tracks
    for obj in tracked:
        color = (0, 0, 255) if obj.in_zone else (0, 255, 0)
        cv2.rectangle(
            ann,
            (int(obj.x1), int(obj.y1)),
            (int(obj.x2), int(obj.y2)),
            color, 2,
        )
        label = get_display_label(obj.class_name)
        label += f" #{obj.track_id:02d}"
        if obj.is_idle:
            label += " [IDLE]"
        cv2.putText(
            ann, label,
            (int(obj.x1), int(obj.y1) - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2,
        )

    # Metrics
    asset_s = get_asset_summary(tracked)
    high_incidents = sum(
        1 for e in st.session_state.session_events
        if getattr(e, "severity", "") == "HIGH" or (isinstance(e, dict) and e.get("severity") == "HIGH")
    )
    score, risk = calculate_safety_score(
        active_violations=active_violating_workers,
        session_incidents=high_incidents,
    )
    people_count = sum(1 for o in tracked if _is_person(o.class_name))

    st.session_state.metrics = SiteMetrics(
        worker_count=people_count,
        asset_count=asset_s["count"],
        avg_utilisation=asset_s["avg_util"],
        safety_score=score,
        risk_level=risk,
    )

    st.session_state.frame_buffer = ann
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 7. UI Page Renderers
# ─────────────────────────────────────────────────────────────────────────────

# Page: Dashboard
def render_dashboard():
    page_title("Dashboard", "Industrial operational overview & live conditions")

    m = st.session_state.metrics
    av = st.session_state.active_violations

    # KPI Cards
    _kpis = st.columns(5)
    with _kpis[0]:
        _kpi_card("Workers Detected", str(m.worker_count), "on site", "blue")
    with _kpis[1]:
        _kpi_card("Assets Tracked", str(m.asset_count), "equipment items", "purple")
    with _kpis[2]:
        _kpi_card("Avg Utilisation", f"{m.avg_utilisation:.1f}%", "active equipment", "green")
    with _kpis[3]:
        _kpi_card("Safety Score", str(m.safety_score), "out of 100", "amber")
    with _kpis[4]:
        risk_color = {"LOW": "green", "MODERATE": "amber", "HIGH": "red"}.get(m.risk_level, "gray")
        _kpi_card("Site Risk", m.risk_level, f"{av} active violations", risk_color)

    st.markdown("<br/>", unsafe_allow_html=True)

    cam_col, right = st.columns([2, 1])
    with cam_col:
        with st.container(border=True):
            st.caption("LIVE COMMAND SNAPSHOT")
            _render_camera_feed()

    with right:
        with st.container(border=True):
            st.caption("RECENT SAFETY INCIDENTS")
            events = get_recent_events(6)
            if events:
                for evt in events:
                    _render_event_row(evt)
            else:
                st.caption("No safety incidents recorded.")

        with st.container(border=True):
            st.caption("ACTIVE SAFETY CONDITIONS")
            _render_safety_status()

    st.markdown("<br/>", unsafe_allow_html=True)
    with st.container(border=True):
        st.caption("OPERATIONAL ASSET SUMMARY")
        tracked = _get_tracked_from_session()
        if tracked:
            asset_sum = get_asset_summary(tracked)
            if asset_sum["items"]:
                _asset_table(asset_sum["items"])
            else:
                st.caption("No equipment currently detected on camera field of view.")
        else:
            st.caption("Start camera to track operational assets.")

    st.markdown("<br/>", unsafe_allow_html=True)
    with st.container(border=True):
        st.caption("AI EXECUTIVE SUMMARY")
        st.write(st.session_state.ai_summary)


# Page: Live Monitor (Hero Page)
def render_live_monitor():
    page_title("Live Monitor", "Hero Canvas — Real-Time Computer Vision Stream")

    ctrl = st.columns(5)
    with ctrl[0]:
        cam_state = "LIVE" if st.session_state.camera_running else "OFFLINE"
        st.metric("Camera Stream", cam_state)
    with ctrl[1]:
        st.metric("Stream FPS", f"{st.session_state.fps:.1f}")
    with ctrl[2]:
        st.metric("Inference", f"{st.session_state.inference_fps:.1f} FPS")
    with ctrl[3]:
        st.metric("Active Violations", str(st.session_state.active_violations))
    with ctrl[4]:
        st.metric("Detection Model", "YOLOv8n")

    st.markdown("<br/>", unsafe_allow_html=True)

    cam_col, command_panel = st.columns([2.5, 1])
    with cam_col:
        with st.container(border=True):
            st.caption("LIVE VISION CANVAS")
            _render_camera_feed()

    with command_panel:
        with st.container(border=True):
            st.caption("SITE COMMAND METRICS")
            m = st.session_state.metrics
            st.write(f"• Workers Detected: **{m.worker_count}**")
            st.write(f"• Equipment Tracked: **{m.asset_count}**")
            st.write(f"• Active Violations: **{st.session_state.active_violations}**")
            st.write(f"• Current Risk Level: **{m.risk_level}**")
            st.write(f"• Safety Score: **{m.safety_score}/100**")

        with st.container(border=True):
            st.caption("LIVE INCIDENTS STREAM")
            session_evts = st.session_state.session_events
            if session_evts:
                for evt in list(reversed(session_evts))[:8]:
                    if isinstance(evt, dict):
                        _render_event_row(evt)
                    else:
                        _render_event_row({
                            "timestamp": "Live",
                            "severity": getattr(evt, "severity", "LOW"),
                            "message": getattr(evt, "message", "Event"),
                            "track_id": getattr(evt, "track_id", None),
                        })
            else:
                st.caption("No active incidents in current session.")


# Page: Safety Zones (NEW PAGE & VISUAL ZONE EDITOR)
def render_safety_zones():
    page_title("Safety Zones", "Multi-Zone Safety Management & 4-Point Polygon Editor")

    zones = st.session_state.safety_zones

    with st.container(border=True):
        st.caption("ACTIVE SAFETY ZONES OVERVIEW")
        z_cols = st.columns(len(zones) if zones else 1)
        for idx, z in enumerate(zones):
            with z_cols[idx % len(z_cols)]:
                status = "ACTIVE" if z.get("enabled", True) else "INACTIVE"
                color_hex = z.get("color_hex", "#3b82f6")
                st.markdown(f"""
                <div style="background:var(--bg-secondary); border:1px solid var(--border); border-left:4px solid {color_hex}; border-radius:4px; padding:10px;">
                    <div style="font-weight:700; font-size:0.85rem;">{z.get('icon','📍')} {z['name']}</div>
                    <div style="font-size:0.7rem; color:var(--text-muted); margin-top:2px;">Severity: <strong>{z['severity']}</strong> | Status: <strong>{status}</strong></div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    col_editor, col_preview = st.columns([1.2, 1.8])

    with col_editor:
        with st.container(border=True):
            st.caption("4-POINT SAFETY ZONE POLYGON EDITOR (NORMALIZED COORDINATES)")

            zone_names = [z["name"] for z in zones]
            selected_name = st.selectbox("Select Zone to Edit", zone_names, index=0 if zone_names else 0)

            # Find target zone
            target_zone = next((z for z in zones if z["name"] == selected_name), zones[0] if zones else None)

            if target_zone:
                st.session_state.editing_zone_id = target_zone["id"]
                z_enabled = st.checkbox("Enable Zone Monitoring", value=target_zone.get("enabled", True), key=f"chk_en_{target_zone['id']}")
                target_zone["enabled"] = z_enabled

                st.caption("Normalized Polygon Vertices (0.00 – 1.00):")
                poly = target_zone.get("polygon", [(0.1, 0.1), (0.5, 0.1), (0.5, 0.5), (0.1, 0.5)])

                # Edit 4 vertices
                v1_x = st.slider("Vertex 1 X", 0.0, 1.0, float(poly[0][0]), 0.02, key=f"v1x_{target_zone['id']}")
                v1_y = st.slider("Vertex 1 Y", 0.0, 1.0, float(poly[0][1]), 0.02, key=f"v1y_{target_zone['id']}")

                v2_x = st.slider("Vertex 2 X", 0.0, 1.0, float(poly[1][0]), 0.02, key=f"v2x_{target_zone['id']}")
                v2_y = st.slider("Vertex 2 Y", 0.0, 1.0, float(poly[1][1]), 0.02, key=f"v2y_{target_zone['id']}")

                v3_x = st.slider("Vertex 3 X", 0.0, 1.0, float(poly[2][0]), 0.02, key=f"v3x_{target_zone['id']}")
                v3_y = st.slider("Vertex 3 Y", 0.0, 1.0, float(poly[2][1]), 0.02, key=f"v3y_{target_zone['id']}")

                v4_x = st.slider("Vertex 4 X", 0.0, 1.0, float(poly[3][0]), 0.02, key=f"v4x_{target_zone['id']}")
                v4_y = st.slider("Vertex 4 Y", 0.0, 1.0, float(poly[3][1]), 0.02, key=f"v4y_{target_zone['id']}")

                target_zone["polygon"] = [(v1_x, v1_y), (v2_x, v2_y), (v3_x, v3_y), (v4_x, v4_y)]

                st.success(f"Updated polygon boundaries for {target_zone['name']}")

    with col_preview:
        with st.container(border=True):
            st.caption("LIVE ZONE BOUNDARY PREVIEW CANVAS")
            buf = st.session_state.frame_buffer
            if buf is not None:
                preview_img = buf.copy()
            else:
                preview_img = np.zeros((720, 1280, 3), dtype=np.uint8) + 20

            # Draw all safety zones on preview
            h, w = preview_img.shape[:2]
            for z in zones:
                if not z.get("enabled", True):
                    continue
                pts = np.array([[int(p[0] * w), int(p[1] * h)] for p in z["polygon"]], dtype=np.int32)
                bgr = z.get("color_bgr", (0, 0, 255))
                overlay = preview_img.copy()
                cv2.fillPoly(overlay, [pts], (int(bgr[0] * 0.3), int(bgr[1] * 0.3), int(bgr[2] * 0.3)))
                cv2.addWeighted(overlay, 0.4, preview_img, 0.6, 0, preview_img)
                thickness = 4 if target_zone and z["id"] == target_zone["id"] else 2
                cv2.polylines(preview_img, [pts], isClosed=True, color=bgr, thickness=thickness)
                cv2.putText(
                    preview_img, z["name"].upper(),
                    (pts[0][0] + 5, max(pts[0][1] - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2,
                )

            rgb_preview = cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB)
            st.image(rgb_preview, use_container_width=True, channels="RGB")
            st.caption("The exact normalized polygon coordinates shown above are evaluated by the Safety Engine in real time.")


# Page: Safety
def render_safety():
    page_title("Safety", "Worker Safety Intelligence & Penalty Model")

    m = st.session_state.metrics
    av = st.session_state.active_violations

    risk_info = {
        "LOW": ("Low risk — normal site operations", "green"),
        "MODERATE": ("Moderate risk — active caution required", "amber"),
        "HIGH": ("High risk — immediate safety attention required", "red"),
    }
    risk_text, risk_color = risk_info.get(m.risk_level, ("Unknown", "gray"))

    with st.container(border=True):
        st.markdown(f'<div style="font-weight:700; font-size:0.9rem; color:var(--accent-{risk_color});">CURRENT SITE RISK LEVEL: {m.risk_level}</div>', unsafe_allow_html=True)
        st.write(risk_text)

    st.markdown("<br/>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _kpi_card("Safety Score", str(m.safety_score), "0–100 score", "amber")
    with c2:
        _kpi_card("Workers Detected", str(m.worker_count), "on site", "blue")
    with c3:
        zone_status = "VIOLATION" if av > 0 else "Clear"
        zone_color = "red" if av > 0 else "green"
        _kpi_card("Zone Status", zone_status, "zone monitor", zone_color)
    with c4:
        _kpi_card("Active Violators", str(av), "in zones", "red")

    st.markdown("<br/>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        with st.container(border=True):
            st.caption("CONFIGURED SAFETY ZONES")
            for z in st.session_state.safety_zones:
                status_text = "ENABLED" if z.get("enabled", True) else "DISABLED"
                st.markdown(
                    f"<div style='margin-bottom:8px;'>{z['icon']} <strong>{z['name']}</strong> — "
                    f"<span style='color:{z['color_hex']}; font-weight:600;'>{z['severity']} RISK ({status_text})</span></div>",
                    unsafe_allow_html=True,
                )

    with col_b:
        with st.container(border=True):
            st.caption("SAFETY SCORING & PPE MODEL")
            st.write("• Base Score: **100**")
            st.write("• Safety Zone Entry Penalty: **-20**")
            st.write("• Repeated Zone Violation Penalty: **-5**")
            st.markdown("---")
            st.caption("PPE DETECTION MODEL STATUS")
            st.markdown('<span class="badge badge-amber">PPE DETECTION: Not configured</span>', unsafe_allow_html=True)
            st.caption("Standard YOLOv8n COCO model detects person bounding boxes. Construction PPE compliance requires custom fine-tuned model weights.")


# Page: Assets
def render_assets():
    page_title("Assets", "Equipment Utilisation & Motion Tracking")

    tracked = _get_tracked_from_session()
    asset_s = get_asset_summary(tracked)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _kpi_card("Tracked Equipment", str(asset_s["count"]), "items", "purple")
    with c2:
        _kpi_card("Active Equipment", str(asset_s["active"]), "in motion", "green")
    with c3:
        _kpi_card("Idle Equipment", str(asset_s["idle"]), "stationary", "amber")
    with c4:
        _kpi_card("Avg Utilisation", f"{asset_s['avg_util']:.1f}%", "overall site", "blue")

    st.markdown("<br/>", unsafe_allow_html=True)
    with st.container(border=True):
        st.caption("DETECTED ASSETS (COCO PROXY CLASSES)")
        if asset_s["items"]:
            _asset_table(asset_s["items"])
        else:
            st.caption("No equipment currently detected. Move vehicles or equipment proxies in front of camera.")


# Page: Events
def render_events():
    page_title("Events", "Chronological Safety & Operational Activity Log")

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        severity_filter = st.selectbox("Severity", ["All", "HIGH", "MEDIUM", "LOW"], index=0)
    with f2:
        type_filter = st.selectbox("Event Category", ["All", "Zone Entry", "Zone Sustained", "Zone Cleared"], index=0)
    with f3:
        search = st.text_input("Search", placeholder="Filter by message...")
    with f4:
        limit = st.slider("Records", 10, 100, 30, 10)

    events = get_recent_events(limit)

    if severity_filter != "All":
        events = [e for e in events if e.get("severity") == severity_filter]
    if search:
        events = [e for e in events if search.lower() in (e.get("message", "") or "").lower()]

    with st.container(border=True):
        if events:
            for evt in events[:limit]:
                _render_event_row(evt)
        else:
            st.caption("No historical events match current filter.")


# Page: Analytics
def render_analytics():
    page_title("Analytics", "Operational Intelligence & Metrics Trends")

    m = st.session_state.metrics
    session_events = st.session_state.session_events
    high_risk_session = sum(
        1 for e in session_events
        if (isinstance(e, dict) and e.get("severity") == "HIGH") or (hasattr(e, "severity") and e.severity == "HIGH")
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _kpi_card("Avg Utilisation", f"{m.avg_utilisation:.1f}%", "active session", "green")
    with c2:
        _kpi_card("Session Events", str(len(session_events)), "active session", "blue")
    with c3:
        _kpi_card("High-Risk Events", str(high_risk_session), "active session", "red")
    with c4:
        _kpi_card("Workers Detected", str(m.worker_count), "active session", "purple")

    st.markdown("<br/>", unsafe_allow_html=True)

    tracked = _get_tracked_from_session()
    col_a, col_b = st.columns(2)
    with col_a:
        with st.container(border=True):
            st.caption("ACTIVE vs IDLE TIME (SESSION)")
            _render_active_idle_chart(tracked)

    with col_b:
        with st.container(border=True):
            st.caption("UTILISATION BY ASSET")
            _render_util_chart(tracked)

    st.markdown("<br/>", unsafe_allow_html=True)
    with st.container(border=True):
        st.caption("EVENT SEVERITY DISTRIBUTION (SESSION)")
        event_dicts = []
        for e in session_events:
            if isinstance(e, dict):
                event_dicts.append(e)
            elif hasattr(e, "severity"):
                event_dicts.append({"severity": e.severity})
        _render_event_chart(event_dicts)

    st.markdown("<br/>", unsafe_allow_html=True)
    with st.container(border=True):
        st.caption("PERSISTED HISTORICAL ANALYTICS (DATABASE)")
        db_events = get_recent_events(100)
        st.write(f"• Total Historical Database Events Logged: **{len(db_events)}**")
        st.write(f"• High-Risk Historical Events: **{sum(1 for e in db_events if e.get('severity') == 'HIGH')}**")


# Page: AI Reports
def render_ai_reports():
    page_title("AI Reports", "Executive Site Intelligence & Automated Summaries")

    if st.button("🤖 Generate AI Executive Report", type="primary"):
        _generate_report()

    st.markdown("<br/>", unsafe_allow_html=True)
    with st.container(border=True):
        st.caption("LATEST EXECUTIVE REPORT")
        st.write(st.session_state.ai_summary)


def _generate_report():
    with st.spinner("Analyzing site telematics and safety logs with Gemini..."):
        try:
            svc = st.session_state.gemini_svc
            if svc is None:
                api_key = _get_gemini_api_key()
                if api_key and len(api_key) > 10:
                    svc = GeminiService(api_key)
                    st.session_state.gemini_svc = svc

            m = st.session_state.metrics
            data = {
                "workers": m.worker_count,
                "active_violations": st.session_state.active_violations,
                "safety_events": len(get_recent_events(100)) + len(st.session_state.session_events),
                "asset_utilisation": m.avg_utilisation,
                "risk": m.risk_level,
                "safety_score": m.safety_score,
            }

            if svc:
                st.session_state.ai_summary = svc.generate_summary(data)
                st.session_state.last_summary_time = time.time()
                src = getattr(svc, "last_generation_source", "gemini")
                if src == "gemini":
                    st.success("Executive report generated via Gemini 2.0 Flash AI.")
                else:
                    st.info("Executive report generated via Deterministic Fallback Engine (Gemini API offline/unconfigured).")
            else:
                st.session_state.ai_summary = _manual_fallback(data)
                st.info("Executive report generated via Deterministic Fallback Engine.")
        except Exception as e:
            st.error(f"Report generation error: {e}")


def _manual_fallback(data: dict) -> str:
    parts = []
    if data["active_violations"] > 0:
        parts.append(f"CRITICAL: {data['active_violations']} worker(s) currently detected inside active Safety Zones. Immediate site supervisor attention required.")
    else:
        parts.append("Site safety conditions are currently clear with zero active zone violations.")

    parts.append(f"Equipment utilisation is recorded at {data['asset_utilisation']:.1f}%.")
    parts.append(f"Overall site safety score is {data['safety_score']}/100 ({data['risk']} risk).")
    parts.append("Recommended supervisor action: Enforce safety zone perimeter boundaries and monitor stationary equipment.")
    return " ".join(parts)


# Page: Settings
def render_settings():
    page_title("Settings", "System Configuration & Diagnostic Telemetry")

    col_a, col_b = st.columns(2)
    with col_a:
        with st.container(border=True):
            st.caption("DETECTION ENGINE TUNING")
            new_conf = st.slider("YOLO Confidence Threshold", 0.1, 0.9, float(st.session_state.confidence_threshold), 0.05)
            new_idle = st.slider("Asset Idle Threshold (seconds)", 1.0, 10.0, float(st.session_state.idle_threshold), 0.5)
            st.session_state.confidence_threshold = new_conf
            st.session_state.idle_threshold = new_idle

    with col_b:
        with st.container(border=True):
            st.caption("SYSTEM DIAGNOSTICS")

            cam_status = "Connected" if st.session_state.camera_running else "Disconnected"
            st.markdown(f"Camera Stream: {_status_badge(cam_status)}", unsafe_allow_html=True)
            st.markdown(f"YOLO Model: {_status_badge('YOLOv8n')}", unsafe_allow_html=True)

            api_key = _get_gemini_api_key()
            gemini_on = bool(api_key and len(api_key) > 10 and "your_gemini" not in api_key.lower())
            gemini_status = "Configured" if gemini_on else "Not configured"
            st.markdown(f"Gemini AI: {_status_badge(gemini_status)}", unsafe_allow_html=True)

            db_exists = os.path.exists("data/site.db")
            db_status = "Active" if db_exists else "Not initialized"
            st.markdown(f"SQLite DB: {_status_badge(db_status)}", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Shared Visual & UI Helpers
# ─────────────────────────────────────────────────────────────────────────────
def page_title(title: str, subtitle: str):
    st.markdown(f"""
    <div style="margin-bottom:12px;">
        <div style="font-size:1.3rem; font-weight:800; color:var(--text-primary); letter-spacing:0.04em;">{title}</div>
        <div style="font-size:0.75rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.06em;">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


def _kpi_card(label: str, value: str, sub: str, color: str = "blue"):
    colors = {
        "blue": "#3b82f6", "green": "#10b981", "amber": "#f59e0b",
        "red": "#ef4444", "purple": "#8b5cf6", "gray": "#64748b",
    }
    accent = colors.get(color, "#3b82f6")
    st.markdown(f"""
    <div class="metric-card" style="border-top: 3px solid {accent};">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


def _render_camera_feed():
    if st.session_state.get("camera_source") == "Browser Camera (WebRTC)":
        buf = st.session_state.frame_buffer
        if buf is not None:
            rgb = cv2.cvtColor(buf, cv2.COLOR_BGR2RGB)
            st.image(rgb, channels="RGB")
        else:
            st.info("📹 Camera initializing... Click **START** in persistent stream box above if camera permission is requested.")
    else:
        buf = st.session_state.frame_buffer
        if buf is not None:
            rgb = cv2.cvtColor(buf, cv2.COLOR_BGR2RGB)
            st.image(rgb, channels="RGB")
        else:
            _offline_prompt()

    st.markdown(
        '<div style="font-size:0.7rem; color:var(--text-muted); margin-top:8px;">'
        '🔒 <strong>Camera Privacy Notice</strong>: Camera frames are processed in real-time '
        'for computer vision inference. Raw camera video is not recorded or stored by the application.'
        '</div>',
        unsafe_allow_html=True,
    )


def _render_event_row(evt):
    sev = evt.get("severity", "LOW").lower()
    css_class = {"high": "high", "medium": "medium", "low": "low"}.get(sev, "low")
    ts = evt.get("timestamp", "")
    if ts and "T" in ts:
        ts = ts.split("T")[1][:8]
    elif ts:
        ts = str(ts)[:8]

    msg = evt.get("message", "—")
    tid = evt.get("track_id")
    if tid is not None:
        msg = f"#{tid:02d} — {msg}"

    st.markdown(f"""
    <div class="event-row event-row-{css_class}">
        <span style="font-size:0.75rem; color:var(--text-muted);">{ts}</span>
        <span style="float:right; font-size:0.7rem;">{_sev_badge(sev)}</span>
        <div style="font-size:0.85rem; color:var(--text-primary); margin-top:2px;">{msg}</div>
    </div>
    """, unsafe_allow_html=True)


def _sev_badge(sev: str) -> str:
    labels = {"high": "HIGH", "medium": "MEDIUM", "low": "LOW"}
    colors = {"high": "red", "medium": "amber", "low": "green"}
    c = colors.get(sev, "gray")
    return f'<span class="badge badge-{c}">{labels.get(sev, sev.upper())}</span>'


def _status_badge(text: str) -> str:
    colors = {
        "Connected": "green", "Active": "green", "Configured": "green", "Online": "green",
        "Disconnected": "gray", "Not configured": "amber", "Not initialized": "gray",
        "Violation": "red",
    }
    c = colors.get(text, "blue")
    return f'<span class="badge badge-{c}">{text}</span>'


def _render_safety_status():
    m = st.session_state.metrics
    st.metric("Score", f"{m.safety_score}/100", delta=m.risk_level)
    st.caption(f"Workers: {m.worker_count} | Violators: {st.session_state.active_violations}")


def _asset_table(items):
    rows = []
    for item in items:
        rows.append({
            "Track ID": f"#{item['id']:02d}",
            "Detected Class": item["class"],
            "Status": item["status"],
            "Active Time": f"{item['active_s']:.1f}s",
            "Idle Time": f"{item['idle_s']:.1f}s",
            "Utilisation": f"{item['util']:.1f}%",
        })
    st.dataframe(rows, hide_index=True)


def _get_tracked_from_session():
    trk = st.session_state.trk
    if trk is None:
        return []
    return list(trk.tracks.values())


def _render_active_idle_chart(tracked):
    try:
        import importlib
        px = importlib.import_module("plotly.express")
        asset_items = [o for o in tracked if _is_equipment(o.class_name)]
        if not asset_items:
            st.caption("No equipment tracked.")
            return
        labels = [f"{get_display_label(o.class_name)} #{o.track_id:02d}" for o in asset_items]
        active_vals = [o.active_time for o in asset_items]
        idle_vals = [o.idle_time for o in asset_items]
        fig = px.bar(
            x=labels,
            y=[active_vals, idle_vals],
            barmode="stack",
            labels={"x": "Asset", "y": "Seconds"},
            color_discrete_sequence=["#10b981", "#64748b"],
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#a0aec0",
            height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", y=-0.15),
        )
        fig.update_xaxes(gridcolor="#253040")
        fig.update_yaxes(gridcolor="#253040")
        st.plotly_chart(fig)
    except Exception:
        st.caption("Install plotly for charts.")


def _render_util_chart(tracked):
    try:
        import importlib
        px = importlib.import_module("plotly.express")
        asset_items = [o for o in tracked if _is_equipment(o.class_name)]
        if not asset_items:
            st.caption("No equipment tracked.")
            return
        labels = [f"#{o.track_id:02d}" for o in asset_items]
        utils = []
        for o in asset_items:
            total = o.active_time + o.idle_time
            u = (o.active_time / total * 100) if total > 0 else 0.0
            utils.append(round(u, 1))

        fig = px.bar(
            x=labels,
            y=utils,
            labels={"x": "Asset", "y": "Utilisation %"},
            color=utils,
            color_continuous_scale=["#ef4444", "#f59e0b", "#10b981"],
            range_color=[0, 100],
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#a0aec0",
            height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
        )
        fig.update_xaxes(gridcolor="#253040")
        fig.update_yaxes(gridcolor="#253040", range=[0, 100])
        st.plotly_chart(fig)
    except Exception:
        st.caption("Install plotly for charts.")


def _render_event_chart(events):
    try:
        import importlib
        px = importlib.import_module("plotly.express")
        if not events:
            st.caption("No events to chart.")
            return
        counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for e in events:
            s = e.get("severity", "LOW").upper()
            counts[s] = counts.get(s, 0) + 1

        labels = list(counts.keys())
        values = list(counts.values())
        colors_map = {"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#10b981"}

        fig = px.pie(
            names=labels,
            values=values,
            color=labels,
            color_discrete_map=colors_map,
            hole=0.55,
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#a0aec0",
            height=260,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=True,
            legend=dict(orientation="h", y=-0.1),
        )
        st.plotly_chart(fig)
    except ImportError:
        st.caption("Install plotly for charts.")


def _offline_prompt():
    st.markdown("""
    <div style="text-align:center; padding: 30px;">
        <div style="font-size:2.5rem; margin-bottom:8px;">📹</div>
        <div style="font-size:1.0rem; font-weight:600; margin-bottom:4px;">Camera Offline</div>
        <div style="color:var(--text-muted); font-size:0.85rem;">
            Click <strong>Start Camera</strong> in the sidebar to begin live monitoring.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Application Router
# ─────────────────────────────────────────────────────────────────────────────
process_frame()

page = st.session_state.page

if page == "Dashboard":
    render_dashboard()
elif page == "Live Monitor":
    render_live_monitor()
elif page == "Safety":
    render_safety()
elif page == "Safety Zones":
    render_safety_zones()
elif page == "Assets":
    render_assets()
elif page == "Events":
    render_events()
elif page == "Analytics":
    render_analytics()
elif page == "AI Reports":
    render_ai_reports()
elif page == "Settings":
    render_settings()
else:
    render_dashboard()

if st.session_state.camera_running:
    time.sleep(0.08)
    st.rerun()
