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


def evaluate_alerts(windows: pd.DataFrame, rules: list[AlertRule] | None = None) -> pd.DataFrame:
    if windows.empty:
        return pd.DataFrame(columns=["window_start", "window_end", "alerta", "severidad", "valor", "umbral"])
    selected_rules = rules or DEFAULT_ALERT_RULES
    rows: list[dict[str, object]] = []
    for _, window in windows.iterrows():
        for rule in selected_rules:
            if not rule.enabled or rule.metric not in windows.columns:
                continue
            value = float(window.get(rule.metric, 0.0) or 0.0)
            triggered = value < rule.threshold if rule.metric == "score_mean" else value >= rule.threshold
            if triggered:
                rows.append(
                    {
                        "session_id": window.get("session_id", ""),
                        "window_start": window.get("window_start"),
                        "window_end": window.get("window_end"),
                        "alerta": rule.label,
                        "severidad": rule.severity,
                        "valor": round(value, 2),
                        "umbral": rule.threshold,
                    }
                )
    return pd.DataFrame(rows)
