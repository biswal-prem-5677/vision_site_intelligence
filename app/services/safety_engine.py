"""
Safety Engine — Worker zone violations, PPE detection, event lifecycle.

Event lifecycle:
  ENTRY → (SUSTAINED, repeated) → EXIT/CLEARED

Per-track cooldown prevents frame-by-frame spam.
"""

from typing import List, Optional, Tuple
from datetime import datetime
from app.schemas import TrackedObject, SafetyEvent
from app.config import (
    SAFETY_ZONES,
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
    zones: Optional[List[dict]] = None,
) -> Tuple[List[SafetyEvent], int, int]:
    """
    Check safety across all categorized Safety Zones for tracked objects.

    Returns:
        events: list of new safety events this frame
        active_violating_workers: count of unique workers currently in at least one active zone
        active_zone_violations: total count of active zone memberships
    """
    events: List[SafetyEvent] = []
    active_zone_violations = 0
    violating_worker_ids = set()

    eval_zones = zones if (zones is not None) else SAFETY_ZONES

    for obj in tracked:
        nx = obj.cx / frame_width
        ny = obj.cy / frame_height

        if obj.active_zone_states is None:
            obj.active_zone_states = {}

        in_any_zone = False

        # Evaluate against each enabled Safety Zone
        for zone in eval_zones:
            if not zone.get("enabled", True):
                continue

            zone_id = zone.get("id", "default_zone")
            zone_name = zone.get("name", "Restricted Area")
            zone_severity = zone.get("severity", "HIGH")
            zone_icon = zone.get("icon", "🔴")
            zone_poly = zone.get("polygon", DANGER_ZONE_POLYGON)

            in_this_zone = point_in_polygon(nx, ny, zone_poly)

            if in_this_zone:
                in_any_zone = True
                active_zone_violations += 1
                violating_worker_ids.add(obj.track_id)

                if zone_id not in obj.active_zone_states:
                    events.append(SafetyEvent(
                        event_type=f"zone_entry_{zone_id}",
                        severity=zone_severity,
                        track_id=obj.track_id,
                        message=f"Worker #{obj.track_id:02d} entered {zone_name} {zone_icon}",
                        zone_id=zone_id,
                        zone_name=zone_name,
                    ))
                    obj.active_zone_states[zone_id] = {
                        "alert_sent": True,
                        "enter_time": now,
                        "last_sustained": now,
                    }
                else:
                    st_info = obj.active_zone_states[zone_id]
                    if (now - st_info["last_sustained"]) >= SUSTAINED_VIOLATION_INTERVAL:
                        events.append(SafetyEvent(
                            event_type=f"zone_sustained_{zone_id}",
                            severity=zone_severity,
                            track_id=obj.track_id,
                            message=f"Worker #{obj.track_id:02d} sustained violation in {zone_name} {zone_icon}",
                            zone_id=zone_id,
                            zone_name=zone_name,
                        ))
                        st_info["last_sustained"] = now
            else:
                if zone_id in obj.active_zone_states:
                    events.append(SafetyEvent(
                        event_type=f"zone_cleared_{zone_id}",
                        severity="LOW",
                        track_id=obj.track_id,
                        message=f"Worker #{obj.track_id:02d} cleared {zone_name} {zone_icon}",
                        zone_id=zone_id,
                        zone_name=zone_name,
                    ))
                    del obj.active_zone_states[zone_id]

        obj.in_zone = in_any_zone

    active_violating_workers = len(violating_worker_ids)
    return events, active_violating_workers, active_zone_violations


def flush_events(events: List[SafetyEvent]):
    """Persist events to database."""
    for evt in events:
        insert_event(
            timestamp=datetime.now().isoformat(),
            event_type=evt.event_type,
            severity=evt.severity,
            track_id=evt.track_id,
            message=evt.message,
            zone_id=evt.zone_id,
            zone_name=evt.zone_name,
        )
