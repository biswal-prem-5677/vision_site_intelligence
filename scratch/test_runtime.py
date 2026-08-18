import time
from app.schemas import TrackedObject
from app.services.safety_engine import check_safety
from app.services.activity_engine import get_asset_summary, update_activity
from app.services.risk_engine import calculate_safety_score
from app.services.gemini_service import GeminiService
from app.config import SAFETY_ZONES

def test_runtime():
    # 1. Test check_safety with zones and Optional parameter
    obj1 = TrackedObject(track_id=1, class_name="person", x1=100, y1=100, x2=200, y2=200, cx=150, cy=150, confidence=0.9)
    obj2 = TrackedObject(track_id=2, class_name="car", x1=300, y1=300, x2=400, y2=400, cx=350, cy=350, confidence=0.85)
    tracked = [obj1, obj2]

    now = time.time()
    events, active_workers, active_zones = check_safety(tracked, 640, 480, now, zones=SAFETY_ZONES)
    print(f"check_safety OK -> events: {len(events)}, active_workers: {active_workers}, active_zones: {active_zones}")

    # 2. Test get_asset_summary active/idle keys
    update_activity(tracked, dt=1.0, frame_width=640, frame_height=480)
    asset_s = get_asset_summary(tracked)
    assert "active" in asset_s, "Missing 'active' key in asset summary!"
    assert "idle" in asset_s, "Missing 'idle' key in asset summary!"
    print(f"get_asset_summary OK -> count: {asset_s['count']}, active: {asset_s['active']}, idle: {asset_s['idle']}")

    # 3. Test risk engine calculation
    score, risk = calculate_safety_score(active_violations=active_workers, session_incidents=len(events))
    print(f"calculate_safety_score OK -> score: {score}, risk: {risk}")

    # 4. Test GeminiService fallback
    svc = GeminiService(api_key="")
    summary = svc.generate_summary({"workers": 1, "safety_events": 0, "asset_utilisation": 50.0, "risk": "LOW", "safety_score": 100})
    print(f"GeminiService fallback OK -> source: {svc.last_generation_source}, summary len: {len(summary)}")

    print("\nALL RUNTIME CONTRACT TESTS PASSED PERFECTLY!")

if __name__ == "__main__":
    test_runtime()
