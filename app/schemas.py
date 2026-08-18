from dataclasses import dataclass
from typing import Optional


@dataclass
class Detection:
    track_id: Optional[int]
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class TrackedObject:
    track_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    cx: float = 0.0
    cy: float = 0.0
    prev_cx: float = 0.0
    prev_cy: float = 0.0
    movement: float = 0.0
    active_time: float = 0.0
    idle_time: float = 0.0
    is_idle: bool = False
    idle_alert_sent: bool = False
    in_zone: bool = False
    zone_alert_sent: bool = False
    zone_enter_time: Optional[float] = None
    zone_last_sustained: Optional[float] = None
    active_zone_states: Optional[dict] = None  # zone_id -> {alert_sent: bool, enter_time: float, last_sustained: float}


@dataclass
class SafetyEvent:
    event_type: str
    severity: str
    track_id: Optional[int]
    message: str


@dataclass
class SiteMetrics:
    worker_count: int = 0
    asset_count: int = 0
    avg_utilisation: float = 0.0
    safety_score: int = 100
    risk_level: str = "LOW"
