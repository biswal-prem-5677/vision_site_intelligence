"""
Safety Score — State-based 0-100 risk model.

The safety score reflects the CURRENT site state, not accumulated history.

Logic:
  1. Start at 100.
  2. Subtract penalties for CURRENT active violations:
     - Worker in danger zone: -20 per worker
     - No helmet (when model supports): -15
     - No vest (when model supports): -10
  3. Clamp to [0, 100].

Risk bands:
  90-100: LOW
  70-89:  MODERATE
  0-69:   HIGH
"""

from app.config import (
    SAFETY_BASE_SCORE,
    PENALTY_DANGER_ZONE,
    PENALTY_NO_HELMET,
    PENALTY_NO_VEST,
    PENALTY_REPEATED_VIOLATION,
    RISK_LOW_MIN,
    RISK_MODERATE_MIN,
)


def calculate_safety_score(
    active_violations: int = 0,
    session_incidents: int = 0,
    no_helmet_count: int = 0,
    no_vest_count: int = 0,
    worker_count: int = 0,
) -> tuple:
    """
    Calculate safety metrics separating Current Safety Status from Session Safety Score.

    Args:
        active_violations: Workers currently in the danger zone
        session_incidents: Total safety incidents recorded in the current session
        no_helmet_count: Workers missing helmets (stub, future)
        no_vest_count: Workers missing safety vests (stub, future)
        worker_count: Total workers detected

    Returns:
        (session_score, current_risk)
    """
    # Current status reflects real-time active violations
    if active_violations == 0:
        current_risk = "LOW"
    elif active_violations == 1:
        current_risk = "MODERATE"
    else:
        current_risk = "HIGH"

    # Session score retains recorded safety incidents over time
    score = SAFETY_BASE_SCORE - (session_incidents * 15)
    if active_violations > 0:
        score -= PENALTY_DANGER_ZONE

    # Clamp to [0, 100]
    score = max(0, min(100, score))

    return score, current_risk
