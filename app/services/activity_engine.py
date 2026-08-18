"""
Activity Engine — Equipment activity classification.

Logic:
  - People (person class) are tracked for safety only.
  - Equipment classes (car, truck, bus, motorcycle, bicycle) are tracked for activity.
  - Movement threshold determines ACTIVE vs IDLE.
  - Utilisation = active_time / (active_time + idle_time) * 100.
"""

from typing import List
from datetime import datetime
from app.schemas import TrackedObject
from app.config import (
    MOVEMENT_THRESHOLD,
    IDLE_TIME_THRESHOLD,
    EQUIPMENT_CLASSES,
)
from app.database import upsert_asset_metrics


def _is_equipment(class_name: str) -> bool:
    return class_name.lower() in EQUIPMENT_CLASSES


def _is_person(class_name: str) -> bool:
    return class_name.lower() == "person"


def update_activity(
    tracked: List[TrackedObject],
    dt: float,
    frame_width: int,
    frame_height: int,
):
    """
    Update active/idle state for all equipment tracks.
    Persist metrics for each unique equipment class.
    """
    for obj in tracked:
        if not _is_equipment(obj.class_name):
            continue  # People are not equipment

        if obj.movement >= MOVEMENT_THRESHOLD:
            obj.active_time += dt
            obj.idle_time = 0
            obj.is_idle = False
            obj.idle_alert_sent = False
        else:
            obj.idle_time += dt
            if obj.idle_time >= IDLE_TIME_THRESHOLD:
                obj.is_idle = True

        # Persist per equipment track
        total = obj.active_time + obj.idle_time
        util = (obj.active_time / total * 100) if total > 0 else 0.0
        upsert_asset_metrics(
            track_id=obj.track_id,
            active_seconds=round(obj.active_time, 1),
            idle_seconds=round(obj.idle_time, 1),
            utilisation_percent=round(util, 1),
        )


def get_asset_summary(tracked: List[TrackedObject]) -> dict:
    """Return summary dict keyed by equipment tracks only."""
    items = []
    for obj in tracked:
        if not _is_equipment(obj.class_name):
            continue
        total = obj.active_time + obj.idle_time
        util = round(obj.active_time / total * 100, 1) if total > 0 else 0.0
        items.append({
            "id": obj.track_id,
            "class": get_display_label(obj.class_name),
            "status": "IDLE" if obj.is_idle else "ACTIVE",
            "util": util,
            "active_s": round(obj.active_time, 1),
            "idle_s": round(obj.idle_time, 1),
            "confidence": round(obj.confidence, 2),
        })
    avg = sum(i["util"] for i in items) / len(items) if items else 0.0
    active_count = sum(1 for i in items if i["status"] == "ACTIVE")
    idle_count = sum(1 for i in items if i["status"] == "IDLE")
    return {
        "count": len(items),
        "active": active_count,
        "idle": idle_count,
        "avg_util": round(avg, 1),
        "items": items,
    }


def get_display_label(class_name: str) -> str:
    """Get human-readable label for detected classes."""
    labels = {
        "person": "Person",
        "car": "Vehicle",
        "truck": "Truck",
        "bus": "Bus",
        "motorcycle": "Motorcycle",
        "bicycle": "Bicycle",
    }
    return labels.get(class_name.lower(), class_name.title())
