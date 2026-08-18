import time
import threading
import av
import cv2
import numpy as np
from streamlit_webrtc import VideoProcessorBase, RTCConfiguration

from app.config import (
    DANGER_ZONE_POLYGON,
    PROCESS_FPS,
    EQUIPMENT_CLASSES,
)
from app.schemas import SiteMetrics, SafetyEvent
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

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


class WebRTCVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.det = Detector()
        self.trk = SimpleTracker()
        self.session_events = []
        self.metrics = SiteMetrics()
        self.active_violations = 0
        self.fps = 0.0
        self.inference_fps = 0.0
        self._last_process_time = 0.0
        self.frame_buffer = None
        self.danger_zone_enabled = True

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        now = time.time()

        elapsed = now - self._last_process_time
        min_interval = 1.0 / PROCESS_FPS

        if elapsed >= min_interval or self._last_process_time == 0.0:
            self._last_process_time = now
            inf_fps = round(1.0 / max(elapsed, 0.001), 1)

            h, w = img.shape[:2]

            # Detections
            detections = self.det.detect(img)

            # Tracking
            tracked = self.trk.update(detections, w, h)

            # Activity Engine
            dt = max(elapsed, 0.05)
            update_activity(tracked, dt, w, h)

            # Safety Engine
            active_violations = 0
            if self.danger_zone_enabled:
                safety_events, active_violations = check_safety(tracked, w, h, now)
                if safety_events:
                    flush_events(safety_events)
                    with self.lock:
                        self.session_events.extend(safety_events)
                        if len(self.session_events) > 200:
                            self.session_events = self.session_events[-200:]

            # Annotations
            ann = img.copy()
            if self.danger_zone_enabled:
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
            high_incidents = sum(1 for e in self.session_events if e.severity == "HIGH")
            score, risk = calculate_safety_score(
                active_violations=active_violations,
                session_incidents=high_incidents,
            )
            people_count = sum(1 for o in tracked if _is_person(o.class_name))

            with self.lock:
                self.fps = inf_fps
                self.inference_fps = inf_fps
                self.active_violations = active_violations
                self.metrics = SiteMetrics(
                    worker_count=people_count,
                    asset_count=asset_s["count"],
                    avg_utilisation=asset_s["avg_util"],
                    safety_score=score,
                    risk_level=risk,
                )
                self.frame_buffer = ann

            return av.VideoFrame.from_ndarray(ann, format="bgr24")
        else:
            with self.lock:
                buf = self.frame_buffer
            if buf is not None:
                return av.VideoFrame.from_ndarray(buf, format="bgr24")
            return av.VideoFrame.from_ndarray(img, format="bgr24")
