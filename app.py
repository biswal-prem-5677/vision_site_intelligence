"""
Site Intelligence Dashboard — Professional Industrial Control Center
Laptop webcam + YOLO + tracking + safety monitoring + asset utilisation
"""

import os
import time
import cv2
import numpy as np
import streamlit as st
from dotenv import load_dotenv

from app.config import (
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
)
from app.services.risk_engine import calculate_safety_score
from app.services.gemini_service import GeminiService

# ─────────────────────────────────────────────────────────────────────────────
# Page Setup
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Site Intelligence",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Professional Dark Theme CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    :root {
        --bg-primary: #0f1419;
        --bg-secondary: #1a1f2b;
        --bg-card: #1e2432;
        --bg-hover: #252d3a;
        --border: #2a3342;
        --border-light: #3a4558;
        --text-primary: #e8eaed;
        --text-secondary: #9aa0a8;
        --text-muted: #6b7280;
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

    /* Header */
    .site-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.6rem 0;
        border-bottom: 1px solid var(--border);
        margin-bottom: 1rem;
    }
    .site-header-left {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .site-title {
        font-size: 1.1rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        color: var(--text-primary);
        text-transform: uppercase;
    }
    .site-subtitle {
        font-size: 0.7rem;
        color: var(--text-muted);
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .live-indicator {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .live-online {
        background: rgba(16, 185, 129, 0.15);
        color: var(--accent-green);
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .live-offline {
        background: rgba(107, 114, 128, 0.15);
        color: var(--text-muted);
        border: 1px solid rgba(107, 114, 128, 0.3);
    }
    .live-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    .live-online .live-dot { background: var(--accent-green); }
    .live-offline .live-dot { background: var(--text-muted); animation: none; }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] .stButton > button {
        background: transparent;
        border: none;
        color: var(--text-secondary);
        font-size: 0.85rem;
        padding: 8px 12px;
        border-radius: 6px;
        text-align: left;
        width: 100%;
        justify-content: flex-start;
        transition: all 0.15s;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: var(--bg-hover);
        color: var(--text-primary);
    }

    /* Metric cards */
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 14px 16px;
        transition: border-color 0.15s;
    }
    .metric-card:hover {
        border-color: var(--border-light);
    }
    .metric-label {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 0.72rem;
        color: var(--text-secondary);
        margin-top: 2px;
    }

    /* Cards */
    .panel {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 16px;
    }
    .panel-title {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 12px;
    }

    /* Status colors */
    .status-green { color: var(--accent-green); }
    .status-amber { color: var(--accent-amber); }
    .status-red { color: var(--accent-red); }
    .status-blue { color: var(--accent-blue); }

    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }
    .badge-green { background: rgba(16,185,129,0.15); color: var(--accent-green); }
    .badge-amber { background: rgba(245,158,11,0.15); color: var(--accent-amber); }
    .badge-red { background: rgba(239,68,68,0.15); color: var(--accent-red); }
    .badge-blue { background: rgba(59,130,246,0.15); color: var(--accent-blue); }
    .badge-gray { background: rgba(107,114,128,0.15); color: var(--text-muted); }

    /* Event table */
    .event-row {
        padding: 10px 12px;
        border-radius: 6px;
        border: 1px solid var(--border);
        margin-bottom: 6px;
        background: var(--bg-secondary);
    }
    .event-row-high { border-left: 3px solid var(--accent-red); }
    .event-row-medium { border-left: 3px solid var(--accent-amber); }
    .event-row-low { border-left: 3px solid var(--accent-green); }

    /* Table */
    table.dataframe {
        font-size: 0.82rem !important;
    }
    table.dataframe th {
        background: var(--bg-secondary) !important;
        color: var(--text-muted) !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        font-size: 0.72rem !important;
        letter-spacing: 0.05em !important;
        border-bottom: 1px solid var(--border) !important;
        padding: 8px 12px !important;
    }
    table.dataframe td {
        padding: 8px 12px !important;
        border-bottom: 1px solid var(--border) !important;
    }

    /* H1/H2 */
    h1, h2, h3 {
        color: var(--text-primary) !important;
    }

    /* Slider */
    [data-testid="stSlider"] { color: var(--accent-blue); }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-primary); }
    ::-webkit-scrollbar-thumb { background: var(--border-light); border-radius: 3px; }

    /* Native Streamlit element overrides */
    div[data-testid="stMetricValue"] {
        font-size: 1.6rem;
        font-weight: 700;
        color: var(--text-primary);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.7rem;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 0.75rem;
    }
    [data-testid="stHorizontalBlock"] > div {
        gap: 0.6rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Init
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()
init_db()

def _get_gemini_api_key() -> str:
    key = ""
    try:
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    if not key:
        key = os.environ.get("GEMINI_API_KEY", "")
    return key


# Session state defaults
_defaults = {
    "camera_source": "Browser Camera (WebRTC)",
    "camera_running": False,
    "frame_buffer": None,
    "last_frame_time": 0.0,
    "session_events": [],
    "metrics": SiteMetrics(),
    "ai_summary": "Start camera and visit AI Reports to generate a site summary.",
    "last_summary_time": 0.0,
    "fps": 0.0,
    "inference_fps": 0.0,
    "page": "Dashboard",
    "active_violations": 0,
    "confidence_threshold": 0.5,
    "idle_threshold": 3.0,
    "danger_zone_enabled": True,
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
# Sidebar Navigation
# ─────────────────────────────────────────────────────────────────────────────
def nav_page(label: str, icon: str) -> bool:
    active = st.session_state.page == label
    if st.button(f"{icon} {label}", key=f"nav_{label}"):
        st.session_state.page = label
        st.rerun()
    return active


with st.sidebar:
    # Brand
    st.markdown(
        '<div class="site-title" style="padding:4px 12px 12px;">🏗️ SITE INTELLIGENCE</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style="padding:0 12px 16px; font-size:0.7rem; color:var(--text-muted); '
        'letter-spacing:0.06em; text-transform:uppercase;">Vision-Driven Monitoring</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style="padding:0 12px 8px; font-size:0.65rem; color:var(--text-muted); '
        'font-weight:600; letter-spacing:0.08em; text-transform:uppercase;">Navigation</div>',
        unsafe_allow_html=True,
    )

    nav_page("Dashboard", "📊")
    nav_page("Live Monitor", "📹")
    nav_page("Assets", "🚜")
    nav_page("Safety", "🛡️")
    nav_page("Events", "📋")
    nav_page("Analytics", "📈")
    nav_page("AI Reports", "🤖")
    nav_page("Settings", "⚙️")

    st.markdown("---")

    # Camera controls (always visible)
    st.markdown(
        '<div style="padding:0 12px 8px; font-size:0.65rem; font-weight:600; '
        'color:var(--text-muted); letter-spacing:0.08em; text-transform:uppercase;">Camera Source</div>',
        unsafe_allow_html=True,
    )
    st.session_state.camera_source = st.selectbox(
        "Source",
        ["Browser Camera (WebRTC)", "Local Webcam / Demo Video"],
        index=0 if st.session_state.camera_source == "Browser Camera (WebRTC)" else 1,
        label_visibility="collapsed",
    )

    if st.session_state.camera_source == "Local Webcam / Demo Video":
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Start", type="primary", use_container_width=True):
                _start_camera()
        with col_b:
            if st.button("Stop", use_container_width=True):
                _stop_camera()
    else:
        st.caption("WebRTC active. Click START in live camera panel.")

    # Danger zone toggle
    st.caption("Danger Zone")
    st.session_state.danger_zone_enabled = st.checkbox(
        "Enable zone monitoring",
        value=st.session_state.danger_zone_enabled,
    )

    st.markdown("---")
    if st.button("Clear All Data", use_container_width=True):
        _clear_data()


# ─────────────────────────────────────────────────────────────────────────────
# Camera Helpers
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
            st.session_state.ai_summary = "Camera active — visit AI Reports for site summary."
        else:
            st.error("Cannot open camera — check webcam connection")
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
    st.session_state.ai_summary = "Data cleared. Start camera to begin."
    st.session_state.trk = SimpleTracker()
    st.success("All data cleared")


# ─────────────────────────────────────────────────────────────────────────────
# Process One Frame (shared by all pages)
# ─────────────────────────────────────────────────────────────────────────────
def process_frame() -> bool:
    """
    Process one camera frame. Call this once per rerun when camera is running.
    Returns True if a frame was processed.
    """
    if not st.session_state.camera_running:
        return False

    if st.session_state.get("camera_source") == "Browser Camera (WebRTC)":
        return True  # WebRTC processor handles frame acquisition asynchronously

    cam = st.session_state.cam
    det = st.session_state.det
    trk = st.session_state.trk

    ret, frame = cam.read()
    if not ret:
        return False

    now = time.time()

    # ── Frame delta ──────────────────────────────────────────────────────────
    if st.session_state.last_frame_time > 0:
        dt = now - st.session_state.last_frame_time
    else:
        dt = 1.0 / PROCESS_FPS
    st.session_state.last_frame_time = now
    raw_fps = 1.0 / dt if dt > 0 else 0.0
    st.session_state.fps = round(raw_fps, 1)

    # ── Inference throttle ───────────────────────────────────────────────────
    elapsed = now - st.session_state._last_process_time
    min_interval = 1.0 / PROCESS_FPS
    if elapsed < min_interval:
        return True  # Skip inference but count as "processing"

    st.session_state._last_process_time = now
    st.session_state.inference_fps = round(1.0 / max(elapsed, 0.001), 1)

    h, w = frame.shape[:2]

    # ── Detection ────────────────────────────────────────────────────────────
    detections = det.detect(frame)

    # ── Tracking ─────────────────────────────────────────────────────────────
    tracked = trk.update(detections, w, h)

    # ── Activity Engine (equipment only) ─────────────────────────────────────
    update_activity(tracked, dt, w, h)

    # ── Safety Engine ────────────────────────────────────────────────────────
    active_violations = 0
    if st.session_state.danger_zone_enabled:
        safety_events, active_violations = check_safety(
            tracked, w, h, now
        )
        if safety_events:
            flush_events(safety_events)
            st.session_state.session_events.extend(safety_events)
            if len(st.session_state.session_events) > 200:
                st.session_state.session_events = st.session_state.session_events[-200:]
    st.session_state.active_violations = active_violations

    # ── Annotate Frame ─────────────────────────────────────────────��─────────
    ann = frame.copy()

    # Danger zone
    if st.session_state.danger_zone_enabled:
        pts = np.array(
            [[int(p[0] * w), int(p[1] * h)] for p in DANGER_ZONE_POLYGON],
            dtype=np.int32,
        )
        overlay = ann.copy()
        cv2.fillPoly(overlay, [pts], (0, 0, 120))
        cv2.addWeighted(overlay, 0.3, ann, 0.7, 0, ann)
        cv2.polylines(ann, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
        cv2.putText(
            ann, "RESTRICTED ZONE",
            (pts[0][0] + 5, pts[0][1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2,
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

    # FPS
    cv2.putText(
        ann,
        f"FPS: {st.session_state.fps:.1f}",
        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
    )

    # Detected count
    people = sum(1 for o in tracked if _is_person(o.class_name))
    equip = sum(1 for o in tracked if _is_equipment(o.class_name))
    cv2.putText(
        ann,
        f"People: {people} | Equipment: {equip}",
        (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2,
    )

    st.session_state.frame_buffer = ann

    # ── Metrics ──────────────────────────────────────────────────────────────
    asset_s = get_asset_summary(tracked)
    session_incidents = sum(1 for e in st.session_state.session_events if e.severity == "HIGH")
    score, risk = calculate_safety_score(
        active_violations=active_violations,
        session_incidents=session_incidents,
    )
    people_count = sum(1 for o in tracked if _is_person(o.class_name))

    st.session_state.metrics = SiteMetrics(
        worker_count=people_count,
        asset_count=asset_s["count"],
        avg_utilisation=asset_s["avg_util"],
        safety_score=score,
        risk_level=risk,
    )

    # ── AI Summary (throttled) ────────────────────────────────────────────────
    if (
        st.session_state.gemini_svc is not None
        and (now - st.session_state.last_summary_time) > 15
    ):
        try:
            data = {
                "workers": people_count,
                "safety_events": len(get_recent_events(100)),
                "active_violations": active_violations,
                "asset_utilisation": asset_s["avg_util"],
                "risk": risk,
                "safety_score": score,
            }
            st.session_state.ai_summary = st.session_state.gemini_svc.generate_summary(
                data
            )
            st.session_state.last_summary_time = now
        except Exception:
            pass

    return True


def _is_person(class_name: str) -> bool:
    return class_name.lower() == "person"


def _is_equipment(class_name: str) -> bool:
    return class_name.lower() in EQUIPMENT_CLASSES


# ─────────────────────────────────────────────────────────────────────────────
# Page: Dashboard
# ───────────────────────────────────────────────���─────────────────────────────
def render_dashboard():
    m = st.session_state.metrics
    page_title("Dashboard", "Site overview — current conditions")

    # KPI Cards
    _kpis = st.columns(5)
    with _kpis[0]:
        _kpi_card("Workers", str(m.worker_count), "detected", "blue")
    with _kpis[1]:
        _kpi_card("Assets", str(m.asset_count), "equipment tracked", "purple")
    with _kpis[2]:
        _kpi_card("Utilisation", f"{m.avg_utilisation:.1f}%", "equipment activity", "green")
    with _kpis[3]:
        _kpi_card("Safety", str(m.safety_score), "out of 100", "amber")
    with _kpis[4]:
        risk_color = {"LOW": "green", "MODERATE": "amber", "HIGH": "red"}.get(m.risk_level, "gray")
        _kpi_card("Risk", m.risk_level, "site risk level", risk_color)

    st.markdown("<br/>", unsafe_allow_html=True)

    if not st.session_state.camera_running:
        _offline_prompt()
        return

    # Camera preview + panels
    cam_col, right = st.columns([2, 1])
    with cam_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.caption("LIVE FEED")
        _render_camera_feed()
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        # Recent events
        st.markdown('<div class="panel" style="margin-bottom:10px;">', unsafe_allow_html=True)
        st.caption("RECENT EVENTS")
        events = get_recent_events(8)
        if events:
            for evt in events:
                _render_event_row(evt)
        else:
            st.caption("No events detected yet.")
        st.markdown("</div>", unsafe_allow_html=True)

        # Safety
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.caption("SAFETY STATUS")
        _render_safety_status()
        st.markdown("</div>", unsafe_allow_html=True)

    # Asset summary
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.caption("ASSET ACTIVITY")
    tracked = _get_tracked_from_session()
    if tracked:
        asset_sum = get_asset_summary(tracked)
        if asset_sum["items"]:
            for item in asset_sum["items"]:
                status_badge = _status_badge(item["status"])
                st.markdown(
                    f"{status_badge} **{item['class']} #{item['id']:02d}** — "
                    f"{item['status']} | Util: {item['util']:.1f}% | "
                    f"Active: {item['active_s']:.0f}s / Idle: {item['idle_s']:.0f}s",
                )
        else:
            st.caption("No equipment currently detected. Move equipment in front of camera.")
    else:
        st.caption("Camera not active. Start camera to track equipment.")
    st.markdown("</div>", unsafe_allow_html=True)

    # AI Summary
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.caption("AI SITE SUMMARY")
    st.write(st.session_state.ai_summary)
    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Page: Live Monitor
# ─────────────────────────────────────────────────────────────────────────────
def render_live_monitor():
    page_title("Live Monitor", "Real-time computer vision feed")

    if not st.session_state.camera_running:
        _offline_prompt()
        return

    # Controls row
    ctrl = st.columns(4)
    with ctrl[0]:
        st.metric("Camera", "Connected", delta="Live")
    with ctrl[1]:
        st.metric("FPS", f"{st.session_state.fps:.1f}")
    with ctrl[2]:
        st.metric("Inference", f"{st.session_state.inference_fps:.1f} FPS")
    with ctrl[3]:
        st.metric("Model", "YOLOv8n")

    st.markdown("<br/>", unsafe_allow_html=True)
    _render_camera_feed()

    st.markdown("<br/>", unsafe_allow_html=True)
    st.caption(
        "Green boxes = normal | Red boxes = restricted zone | "
        "Track IDs shown with class labels"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Page: Assets
# ─────────────────────────────────────────────────────────────────────────────
def render_assets():
    page_title("Assets", "Equipment utilisation & activity intelligence")

    if not st.session_state.camera_running:
        _offline_prompt()
        return

    # Session-level assets
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.caption("SESSION TRACKS (LIVE)")
    tracked = _get_tracked_from_session()
    asset_sum = get_asset_summary(tracked)
    if asset_sum["items"]:
        _asset_table(asset_sum["items"])
    else:
        st.caption("No equipment currently detected. Show equipment to the camera.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # DB-persisted assets
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.caption("PERSISTED ASSETS (DATABASE)")
    db_assets = get_all_asset_metrics()
    if db_assets:
        rows = []
        for a in db_assets:
            status = "ACTIVE" if a["active_seconds"] > a["idle_seconds"] else "IDLE"
            total = a["active_seconds"] + a["idle_seconds"]
            util = (a["active_seconds"] / total * 100) if total > 0 else 0.0
            rows.append({
                "Track ID": f"#{a['track_id']:02d}",
                "Active": f"{a['active_seconds']:.1f}s",
                "Idle": f"{a['idle_seconds']:.1f}s",
                "Utilisation": f"{util:.1f}%",
                "Status": status,
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.caption("No asset data in database yet. Start camera to begin tracking.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.caption("DETECTED CLASSES")
    if tracked:
        classes = {}
        for o in tracked:
            label = get_display_label(o.class_name)
            classes[label] = classes.get(label, 0) + 1
        for cls, count in classes.items():
            st.write(f"**{cls}**: {count} track(s)")
    else:
        st.caption("No active tracks.")
    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Page: Safety
# ─────────────────────────────────────────────────────────────────────────────
def render_safety():
    page_title("Safety", "Worker safety monitoring & zone management")

    m = st.session_state.metrics
    av = st.session_state.active_violations

    # Risk banner
    risk_info = {
        "LOW": ("Low risk — normal operations", "green"),
        "MODERATE": ("Moderate risk — monitor closely", "amber"),
        "HIGH": ("High risk — immediate attention required", "red"),
    }
    risk_text, risk_color = risk_info.get(m.risk_level, ("Unknown", "gray"))

    st.markdown(f"""
    <div class="panel" style="border-left: 3px solid var(--accent-{risk_color});">
        <div class="panel-title">RISK LEVEL: {m.risk_level}</div>
        <p style="color: var(--text-secondary); margin: 0;">{risk_text}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _kpi_card("Safety Score", str(m.safety_score), "0–100", "amber")
    with c2:
        _kpi_card("Workers", str(m.worker_count), "detected", "blue")
    with c3:
        zone_status = "VIOLATION" if av > 0 else "Clear"
        zone_color = "red" if av > 0 else "green"
        _kpi_card("Zone Status", zone_status, "restricted area", zone_color)
    with c4:
        _kpi_card("Zone Violators", str(av), "active", "red")

    st.markdown("<br/>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.caption("ZONE CONFIGURATION")
        if st.session_state.danger_zone_enabled:
            st.caption("Zone monitoring is ENABLED. Workers entering the red polygon trigger safety events.")
        else:
            st.caption("Zone monitoring is DISABLED. No zone-based events will be generated.")
        st.caption(
            "Adjust zone coordinates in Settings. "
            "Default zone covers center 40% width × 30% height."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.caption("SCORING MODEL")
        st.caption(
            "Score starts at 100. "
            "Danger zone entry: -20. "
            "PPE violations: -15 (helmet), -10 (vest). "
            "Repeated violations: -5 each. "
            "Score resets to 100 when no violations are active."
        )
        st.caption(f"Penalty reference: zone={20}, helmet={15}, vest={10}, repeat={5}")
        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.camera_running:
        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.caption("SAFETY EVENTS")
        events = get_recent_events(15)
        if events:
            for evt in events:
                _render_event_row(evt)
        else:
            st.caption("No safety events recorded.")
        st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Page: Events
# ────────────────────────────────────────────────────────���────────────────────
def render_events():
    page_title("Events", "Chronological safety & activity log")

    # Filters
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        severity_filter = st.selectbox(
            "Severity", ["All", "HIGH", "MEDIUM", "LOW"], index=0,
        )
    with f2:
        type_filter = st.selectbox(
            "Type",
            ["All", "danger_zone_entry", "danger_zone_sustained", "danger_zone_cleared", "other"],
            index=0,
        )
    with f3:
        search = st.text_input("Search", placeholder="Search messages...")
    with f4:
        limit = st.slider("Show", 10, 100, 30, 10)

    events = get_recent_events(limit)

    # Apply filters
    if severity_filter != "All":
        events = [e for e in events if e.get("severity") == severity_filter]
    if type_filter != "All":
        if type_filter == "other":
            events = [e for e in events if not e.get("event_type", "").startswith("danger_zone")]
        else:
            events = [e for e in events if e.get("event_type") == type_filter]
    if search:
        events = [e for e in events if search.lower() in (e.get("message", "") or "").lower()]

    # Session events (not yet DB-persisted)
    if st.session_state.session_events:
        db_ids = {e["id"] for e in events if "id" in e}
        for evt in st.session_state.session_events:
            events.insert(0, {
                "timestamp": "Current session",
                "event_type": evt.event_type,
                "severity": evt.severity,
                "track_id": evt.track_id,
                "message": evt.message,
            })

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    if events:
        for evt in events[:limit]:
            _render_event_row(evt)
    else:
        st.caption("No events match the current filters.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    st.caption(
        "Event lifecycle: ENTRY → (SUSTAINED, every 5s) → CLEARED. "
        "Identical entry events are deduplicated via per-track cooldown."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Page: Analytics
# ─────────────────────────────────────────────────────────────────────────────
def render_analytics():
    page_title("Analytics", "Operational intelligence & trends")

    m = st.session_state.metrics

    # Top metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _kpi_card("Avg Utilisation", f"{m.avg_utilisation:.1f}%", "all equipment", "green")
    with c2:
        events = get_recent_events(100)
        st.caption("")
        st.caption("")
        st.metric("Total Events", len(events))
    with c3:
        high_risk = sum(1 for e in events if e.get("severity") == "HIGH")
        st.caption("")
        st.caption("")
        st.metric("High-Risk Events", high_risk)
    with c4:
        st.caption("")
        st.caption("")
        st.metric("Workers", m.worker_count)

    st.markdown("<br/>", unsafe_allow_html=True)

    if not st.session_state.camera_running:
        st.caption("Start camera to generate session analytics.")
        return

    tracked = _get_tracked_from_session()

    # Activity breakdown
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.caption("ACTIVE vs IDLE")
        _render_active_idle_chart(tracked)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.caption("UTILISATION BY ASSET")
        _render_util_chart(tracked)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # Event distribution
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.caption("EVENT SEVERITY DISTRIBUTION")
    _render_event_chart(events)
    st.markdown("</div>", unsafe_allow_html=True)

    # Safety score gauge
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.caption("SAFETY SCORE")
    _render_safety_gauge(m.safety_score, m.risk_level)
    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Page: AI Reports
# ─────────────────────────────────────────────────────────────────────────────
def render_ai_reports():
    page_title("AI Reports", "Management-level site intelligence")

    if st.button("Generate Report", type="primary"):
        _generate_report()

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.caption("LATEST REPORT")
    st.write(st.session_state.ai_summary)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.caption("HOW IT WORKS")
    st.caption(
        "The CV system extracts structured metrics (worker count, active violations, "
        "equipment utilisation, event counts). These are sent to Gemini AI, which "
        "generates a management-level site summary. "
        "If Gemini is unavailable, a rule-based fallback is used automatically."
    )
    status = "Active" if st.session_state.gemini_svc else "Not configured"
    status_badge = _status_badge(status.lower())
    st.caption(f"Gemini status: {status_badge}")
    st.markdown("</div>", unsafe_allow_html=True)

    # Raw metrics reference
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.caption("RAW METRICS (SENT TO AI)")
    m = st.session_state.metrics
    st.write(
        f"- Workers: {m.worker_count}\n"
        f"- Assets: {m.asset_count}\n"
        f"- Utilisation: {m.avg_utilisation:.1f}%\n"
        f"- Safety Score: {m.safety_score}\n"
        f"- Risk Level: {m.risk_level}\n"
        f"- Active Violations: {st.session_state.active_violations}\n"
        f"- Session Events: {len(st.session_state.session_events)}"
    )
    st.markdown("</div>", unsafe_allow_html=True)


def _generate_report():
    with st.spinner("Generating report..."):
        try:
            svc = st.session_state.gemini_svc
            if svc is None:
                api_key = os.environ.get("GEMINI_API_KEY", "")
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
                st.success("Report generated")
            else:
                # Fallback
                st.session_state.ai_summary = _manual_fallback(data)
                st.success("Report generated (fallback)")
        except Exception as e:
            st.error(f"Report generation failed: {e}")


def _manual_fallback(data: dict) -> str:
    parts = []
    if data["active_violations"] > 0:
        parts.append(
            f"{data['active_violations']} worker(s) currently in the restricted zone. "
            "Immediate attention required."
        )
    else:
        parts.append("No active zone violations. Site is currently clear.")

    parts.append(f"Asset utilisation at {data['asset_utilisation']:.1f}%.")
    parts.append(f"Safety score: {data['safety_score']}/100 ({data['risk']} risk).")
    parts.append(
        "Recommended action: Continue monitoring zone boundaries and PPE compliance."
    )
    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Page: Settings
# ─────────────────────────────────────────────────────────────────────────────
def render_settings():
    page_title("Settings", "System configuration & status")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.caption("DETECTION")
        new_conf = st.slider(
            "Confidence Threshold", 0.1, 0.9,
            float(st.session_state.confidence_threshold), 0.05,
        )
        new_idle = st.slider(
            "Idle Threshold (seconds)", 1.0, 10.0,
            float(st.session_state.idle_threshold), 0.5,
        )
        st.session_state.confidence_threshold = new_conf
        st.session_state.idle_threshold = new_idle
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.caption("SYSTEM STATUS")

        # Camera
        cam_status = "Connected" if st.session_state.camera_running else "Disconnected"
        st.markdown(f"Camera: {_status_badge(cam_status)}")

        # Model
        st.markdown(f"Model: {_status_badge('YOLOv8n')}")

        # Gemini
        api_key = os.environ.get("GEMINI_API_KEY", "")
        gemini_on = bool(api_key and len(api_key) > 10 and "your_gemini" not in api_key.lower())
        gemini_status = "Configured" if gemini_on else "Not configured"
        st.markdown(f"Gemini AI: {_status_badge(gemini_status)}")

        # DB
        import os as _os
        db_exists = _os.path.exists("data/site.db")
        db_status = "Active" if db_exists else "Not initialized"
        st.markdown(f"Database: {_status_badge(db_status)}")

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.caption("DANGER ZONE")
    st.caption("Current polygon (normalized 0-1 coordinates):")
    for i, pt in enumerate(DANGER_ZONE_POLYGON):
        st.write(f"  Vertex {i+1}: ({pt[0]:.2f}, {pt[1]:.2f})")
    st.caption("Use the Danger Zone sliders in the sidebar to adjust.")

    st.caption("Equipment classes detected:")
    st.write(", ".join(c.title() for c in sorted(EQUIPMENT_CLASSES)))

    st.caption("Scoring penalties:")
    st.write(f"Danger zone: -{PENALTY_DANGER_ZONE} | Helmet: -{PENALTY_NO_HELMET} | "
             f"Vest: -{PENALTY_NO_VEST} | Repeated: -{PENALTY_REPEATED_VIOLATION}")
    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Shared UI Components
# ─────────────────────────────────────────────────────────────────────────────
def page_title(title: str, subtitle: str):
    st.markdown(f"""
    <div class="site-header">
        <div class="site-header-left">
            <div>
                <div class="site-title">{title}</div>
                <div class="site-subtitle">{subtitle}</div>
            </div>
        </div>
        <div>
            {"<span class='live-indicator live-online'><span class='live-dot'></span> LIVE</span>"
             if st.session_state.camera_running else
             "<span class='live-indicator live-offline'><span class='live-dot'></span> OFFLINE</span>"}
        </div>
    </div>
    """, unsafe_allow_html=True)


def _kpi_card(label: str, value: str, sub: str, color: str = "blue"):
    color_map = {
        "blue": "var(--accent-blue)",
        "green": "var(--accent-green)",
        "amber": "var(--accent-amber)",
        "red": "var(--accent-red)",
        "purple": "var(--accent-purple)",
        "gray": "var(--text-muted)",
    }
    accent = color_map.get(color, "var(--accent-blue)")
    st.markdown(f"""
    <div class="metric-card" style="border-top: 2px solid {accent};">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


def _render_camera_feed():
    if st.session_state.get("camera_source") == "Browser Camera (WebRTC)":
        try:
            from streamlit_webrtc import webrtc_streamer, WebRtcMode
            from app.services.webrtc_engine import WebRTCVideoProcessor, RTC_CONFIGURATION

            webrtc_ctx = webrtc_streamer(
                key="browser-camera-feed",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                video_processor_factory=WebRTCVideoProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )

            if webrtc_ctx.video_processor:
                proc = webrtc_ctx.video_processor
                proc.danger_zone_enabled = st.session_state.danger_zone_enabled
                with proc.lock:
                    st.session_state.camera_running = True
                    st.session_state.metrics = proc.metrics
                    st.session_state.active_violations = proc.active_violations
                    st.session_state.fps = proc.fps
                    st.session_state.inference_fps = proc.inference_fps
                    if proc.session_events:
                        st.session_state.session_events = list(proc.session_events)
            else:
                st.info("📹 Click **START** above to grant browser camera access.")
        except Exception as e:
            st.error(f"WebRTC Error: {e}")
            buf = st.session_state.frame_buffer
            if buf is not None:
                rgb = cv2.cvtColor(buf, cv2.COLOR_BGR2RGB)
                st.image(rgb, use_container_width=True, channels="RGB")
    else:
        buf = st.session_state.frame_buffer
        if buf is not None:
            rgb = cv2.cvtColor(buf, cv2.COLOR_BGR2RGB)
            st.image(rgb, use_container_width=True, channels="RGB")
        else:
            st.caption("Waiting for camera feed...")

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
            "ID": f"#{item['id']:02d}",
            "Class": item["class"],
            "Status": item["status"],
            "Active": f"{item['active_s']:.1f}s",
            "Idle": f"{item['idle_s']:.1f}s",
            "Utilisation": f"{item['util']:.1f}%",
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _get_tracked_from_session():
    trk = st.session_state.trk
    if trk is None:
        return []
    return list(trk.tracks.values())


def _render_active_idle_chart(tracked):
    try:
        import plotly.express as px
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
            color_discrete_sequence=["#10b981", "#6b7280"],
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#9aa0a8",
            height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", y=-0.15),
        )
        fig.update_xaxes(gridcolor="#2a3342")
        fig.update_yaxes(gridcolor="#2a3342")
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.caption("Install plotly for charts.")


def _render_util_chart(tracked):
    try:
        import plotly.express as px
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
            font_color="#9aa0a8",
            height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
        )
        fig.update_xaxes(gridcolor="#2a3342")
        fig.update_yaxes(gridcolor="#2a3342", range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.caption("Install plotly for charts.")


def _render_event_chart(events):
    try:
        import plotly.express as px
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
            font_color="#9aa0a8",
            height=260,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=True,
            legend=dict(orientation="h", y=-0.1),
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.caption("Install plotly for charts.")


def _render_safety_gauge(score: int, risk: str):
    try:
        import plotly.graph_objects as go
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": f"Risk: {risk}", "font": {"size": 14, "color": "#9aa0a8"}},
            number={"font": {"size": 48, "color": "#e8eaed"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#3a4558", "tickfont": {"color": "#9aa0a8"}},
                "bar": {"color": _gauge_color(score)},
                "bgcolor": "#1e2432",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 70], "color": "rgba(239,68,68,0.15)"},
                    {"range": [70, 90], "color": "rgba(245,158,11,0.15)"},
                    {"range": [90, 100], "color": "rgba(16,185,129,0.15)"},
                ],
            },
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=20, b=20),
            height=200,
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.caption("Install plotly for charts.")


def _gauge_color(score: int) -> str:
    if score >= 90:
        return "#10b981"
    if score >= 70:
        return "#f59e0b"
    return "#ef4444"


def _offline_prompt():
    st.markdown("""
    <div class="panel" style="text-align:center; padding: 40px;">
        <div style="font-size:3rem; margin-bottom:12px;">📹</div>
        <div style="font-size:1.1rem; font-weight:600; margin-bottom:8px;">Camera Offline</div>
        <div style="color:var(--text-muted); font-size:0.9rem;">
            Click <strong>Start</strong> in the sidebar to begin live monitoring.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────────────────────
# Process frame first so all pages see fresh data
process_frame()

page = st.session_state.page

if page == "Dashboard":
    render_dashboard()
elif page == "Live Monitor":
    render_live_monitor()
elif page == "Assets":
    render_assets()
elif page == "Safety":
    render_safety()
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

# Auto-refresh when camera is running
if st.session_state.camera_running:
    time.sleep(0.08)
    st.rerun()
