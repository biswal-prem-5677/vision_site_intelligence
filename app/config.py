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
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS_TARGET = 15
PROCESS_FPS = 10  # Target inference FPS

# Danger zone (normalized 0-1, set as polygon vertices)
DANGER_ZONE_POLYGON = [
    (0.3, 0.2),
    (0.7, 0.2),
    (0.7, 0.5),
    (0.3, 0.5),
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
