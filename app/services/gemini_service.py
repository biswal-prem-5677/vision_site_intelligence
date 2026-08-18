from typing import Optional
from app.config import GEMINI_MODEL
import google.generativeai as genai


class GeminiService:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(GEMINI_MODEL)
        self.enabled = bool(api_key)

    def generate_summary(self, metrics: dict) -> str:
        if not self.enabled:
            return self._fallback_summary(metrics)

        prompt = f"""You are a construction site safety analyst. Generate a concise site intelligence summary based on these metrics:

Workers detected: {metrics.get('workers', 0)}
Safety events: {metrics.get('safety_events', 0)}
Asset utilization: {metrics.get('asset_utilisation', 0)}%
Risk level: {metrics.get('risk', 'LOW')}
Safety score: {metrics.get('safety_score', 100)}%

Format your response as:
- A 2-3 sentence overall assessment
- Top priority action item
- Brief recommendation

Keep it under 120 words. Be direct and professional."""

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception:
            return self._fallback_summary(metrics)

    def _fallback_summary(self, metrics: dict) -> str:
        risk = metrics.get("risk", "LOW")
        events = metrics.get("safety_events", 0)
        util = metrics.get("asset_utilisation", 0)

        lines = []
        if events > 0:
            lines.append(f"{events} safety event(s) were observed this session.")
        else:
            lines.append("No safety events detected in this session.")

        if util >= 70:
            lines.append("Asset utilization is healthy.")
        elif util > 0:
            lines.append("Asset utilization is below optimal levels.")

        if risk == "HIGH":
            lines.append("Recommended action: Review operating boundaries and reinforce zone safety protocols.")
        elif risk == "MODERATE":
            lines.append("Recommended action: Monitor the restricted zone and address minor compliance gaps.")
        else:
            lines.append("Site conditions are within acceptable safety parameters.")

        return " ".join(lines)
