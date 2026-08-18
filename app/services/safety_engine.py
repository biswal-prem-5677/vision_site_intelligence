"""
Safety Engine — Worker zone violations, PPE detection, event lifecycle.

Event lifecycle:
  ENTRY → (SUSTAINED, repeated) → EXIT/CLEARED

Per-track cooldown prevents frame-by-frame spam.
"""

from typing import List, Tuple
from datetime import datetime
from app.schemas import TrackedObject, SafetyEvent
from app.config import (
    DANGER_ZONE_POLYGON,
    PENALTY_DANGER_ZONE,
    PENALTY_NO_HELMET,
    PENALTY_NO_VEST,
    PENALTY_REPEATED_VIOLATION,
    EVENT_COOLDOWN_SECONDS,
    SUSTAINED_VIOLATION_INTERVAL,
)
from app.database import insert_event


def point_in_polygon(x: float, y: float, polygon) -> bool:
    """Ray casting point-in-polygon test. Coordinates are normalized 0-1."""
    n = len(polygon)
    inside = False
    px, py = x, y
    for i in range(n):
        j = (i + 1) % n
        yi, yj = polygon[i][1], polygon[j][1]
        xi, xj = polygon[i][0], polygon[j][0]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
    return inside


def check_safety(
    tracked: List[TrackedObject],
    frame_width: int,
    frame_height: int,
    now: float,
) -> Tuple[List[SafetyEvent], int]:
    """
    Check safety for all tracked objects.

    Returns:
        events: list of new safety events this frame
        active_violations: count of objects currently in the danger zone
                           (used for safety score)
    """
    events: List[SafetyEvent] = []
    active_violations = 0

    for obj in tracked:
        nx = obj.cx / frame_width
        ny = obj.cy / frame_height
        in_zone = point_in_polygon(nx, ny, DANGER_ZONE_POLYGON)
        obj.in_zone = in_zone

        # ── Zone lifecycle ─────────────────────────────────────────────────
        if in_zone:
            active_violations += 1
            if not obj.zone_alert_sent:
                # ENTRY event
                events.append(SafetyEvent(
                    event_type="danger_zone_entry",
                    severity="HIGH",
                    track_id=obj.track_id,
                    message=f"Worker #{obj.track_id:02d} entered restricted zone",
                ))
                obj.zone_alert_sent = True
                obj.zone_enter_time = now
                obj.zone_last_sustained = now
            else:
                # Already in zone — check for sustained violation
                if (now - obj.zone_last_sustained) >= SUSTAINED_VIOLATION_INTERVAL:
                    events.append(SafetyEvent(
                        event_type="danger_zone_sustained",
                        severity="HIGH",
                        track_id=obj.track_id,
                        message=f"Worker #{obj.track_id:02d} sustained restricted zone violation",
                    ))
                    obj.zone_last_sustained = now
        else:
            if obj.zone_alert_sent:
                # EXIT / CLEARED
                events.append(SafetyEvent(
                    event_type="danger_zone_cleared",
                    severity="LOW",
                    track_id=obj.track_id,
                    message=f"Worker #{obj.track_id:02d} cleared restricted zone",
                ))
            obj.zone_alert_sent = False
            obj.zone_enter_time = None
            obj.zone_last_sustained = None

        # ── PPE checks ─────────────────────────────────────────────────────
        # Note: Standard YOLOv8n COCO does NOT have helmet/vest classes.
        # PPE detection requires a construction-specific model.
        # Stub kept for future integration — no fake detections.
        if obj.confidence > 0.6:
            pass

    return events, active_violations


def flush_events(events: List[SafetyEvent]):
    """Persist events to database."""
    for evt in events:
        insert_event(
            timestamp=datetime.now().isoformat(),
            event_type=evt.event_type,
            severity=evt.severity,
            track_id=evt.track_id,
            message=evt.message,
        )
