from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

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

def rules_from_frame(rules_frame: pd.DataFrame) -> list[AlertRule]:
    if rules_frame.empty:
        return DEFAULT_ALERT_RULES
    labels = {rule.rule_key: rule for rule in DEFAULT_ALERT_RULES}
    rules: list[AlertRule] = []
    for _, row in rules_frame.iterrows():
        base = labels.get(str(row["rule_key"]))
        if base is None:
            continue
        rules.append(
            AlertRule(
                rule_key=base.rule_key,
                label=base.label,
                metric=base.metric,
                threshold=float(row["threshold"]),
                window_seconds=float(row["window_seconds"]),
                severity=str(row["severity"]),
                enabled=bool(row["enabled"]),
            )
        )
    return rules or DEFAULT_ALERT_RULES

