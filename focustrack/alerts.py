from __future__ import annotations

from dataclasses import dataclass

@dataclass
class AlertRule:
    rule_key: str
    label: str
    metric: str
    threshold: float
    window_seconds: float
    severity: str = "media"
    enabled: bool = True


DEFAULT_ALERT_RULES = [
    AlertRule("low_score", "Score bajo", "score_mean", 45.0, 60.0, "alta"),
    AlertRule("phone", "Celular recurrente", "phone_detected_pct", 20.0, 60.0, "alta"),
    AlertRule("absence", "Ausencia prolongada", "pct_ausente", 20.0, 60.0, "alta"),
    AlertRule("sleepy", "Somnolencia", "pct_somnoliento", 15.0, 60.0, "media"),
    AlertRule("distracting_apps", "Aplicaciones distractoras", "screen_distraccion_pct", 25.0, 60.0, "media"),
]