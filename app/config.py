# YOLO model
YOLO_MODEL = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45

# Classes
PERSON_CLASS = 0  # COCO class for person

# Equipment proxy classes (COCO class names)
EQUIPMENT_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}
EQUIPMENT_CLASS_IDS = {2, 3, 5, 7, 1}  # COCO IDs

# Tracking
TRACKER_MAX_AGE = 30
TRACKER_INIT_HIT_STREAK = 3

# Camera
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS_TARGET = 30
PROCESS_FPS = 15  # Target inference FPS

# Danger zone (legacy fallback)
DANGER_ZONE_POLYGON = [
    (0.3, 0.2),
    (0.7, 0.2),
    (0.7, 0.5),
    (0.3, 0.5),
]

# Categorized Safety Zones
SAFETY_ZONES = [
    {
        "id": "crane_swing",
        "name": "Crane Swing Area",
        "color_bgr": (0, 0, 255),        # Red
        "color_hex": "#ef4444",
        "icon": "🔴",
        "severity": "HIGH",
        "enabled": True,
        "polygon": [(0.1, 0.1), (0.45, 0.1), (0.45, 0.45), (0.1, 0.45)],
    },
    {
        "id": "excavation",
        "name": "Excavation Zone",
        "color_bgr": (0, 140, 255),      # Orange
        "color_hex": "#f97316",
        "icon": "🟠",
        "severity": "HIGH",
        "enabled": True,
        "polygon": [(0.55, 0.1), (0.9, 0.1), (0.9, 0.45), (0.55, 0.45)],
    },
    {
        "id": "restricted_personnel",
        "name": "Restricted Personnel Area",
        "color_bgr": (0, 0, 200),        # Deep Red
        "color_hex": "#dc2626",
        "icon": "🔴",
        "severity": "HIGH",
        "enabled": True,
        "polygon": [(0.3, 0.55), (0.7, 0.55), (0.7, 0.9), (0.3, 0.9)],
    },
    {
        "id": "equipment_operating",
        "name": "Equipment Operating Area",
        "color_bgr": (0, 230, 255),      # Yellow
        "color_hex": "#eab308",
        "icon": "🟡",
        "severity": "MEDIUM",
        "enabled": True,
        "polygon": [(0.05, 0.55), (0.25, 0.55), (0.25, 0.9), (0.05, 0.9)],
    },
]

# Activity detection
MOVEMENT_THRESHOLD = 15.0  # pixels per frame
IDLE_TIME_THRESHOLD = 3.0  # seconds before marking idle

# Safety scoring (bounded 0-100, derived from current state)
SAFETY_BASE_SCORE = 100
PENALTY_DANGER_ZONE = 20
PENALTY_NO_HELMET = 15
PENALTY_NO_VEST = 10
PENALTY_REPEATED_VIOLATION = 5

# Risk levels
RISK_LOW_MIN = 90
RISK_MODERATE_MIN = 70

# Event lifecycle cooldowns
EVENT_COOLDOWN_SECONDS = 2.0  # Min seconds between similar events
SUSTAINED_VIOLATION_INTERVAL = 5.0  # Seconds between sustained violation alerts

# Database
DB_PATH = "data/site.db"

# Gemini
GEMINI_MODEL = "gemini-2.0-flash"
