from typing import Optional
import warnings
from app.config import GEMINI_MODEL

# Suppress deprecation warning if present
warnings.filterwarnings("ignore", category=FutureWarning, module="google.generativeai")


class GeminiService:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.enabled = bool(api_key and len(api_key) > 10 and "your_gemini" not in api_key.lower())
        self.client = None
        self.use_new_sdk = False
        self.last_generation_source = "none"

        if self.enabled:
            try:
                from google import genai
                self.client = genai.Client(api_key=api_key)
                self.use_new_sdk = True
            except Exception:
                try:
                    import google.generativeai as legacy_genai
                    legacy_genai.configure(api_key=api_key)
                    self.client = legacy_genai.GenerativeModel(GEMINI_MODEL)
                    self.use_new_sdk = False
                except Exception:
                    self.enabled = False

    def generate_summary(self, metrics: dict) -> str:
        if not self.enabled or self.client is None:
            self.last_generation_source = "fallback"
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
            if self.use_new_sdk:
                response = self.client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                )
                summary = response.text.strip()
            else:
                response = self.client.generate_content(prompt)
                summary = response.text.strip()

            self.last_generation_source = "gemini"
            return summary
        except Exception:
            self.last_generation_source = "fallback"
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
