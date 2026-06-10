from __future__ import annotations

import pandas as pd


EVENT_DEFINITIONS = {
    "mirada_desviada": ("attention_state", {"desviado"}),
    "somnolencia": ("attention_state", {"somnoliento"}),
    "ausencia": ("attention_state", {"ausente"}),
    "celular": ("phone_detected", {True}),
    "mano_en_rostro": ("hand_on_face", {True}),
    "postura_encorvada": ("posture_state", {"encorvada"}),
    "app_distractora": ("screen_category", {"distraccion"}),
}


def build_timeline_events(history: pd.DataFrame, min_duration_seconds: float = 2.0) -> pd.DataFrame:
    if history.empty or "timestamp" not in history:
        return pd.DataFrame(columns=["session_id", "evento", "inicio", "fin", "duracion_seg", "detalle"])

    frame = history.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values(["session_id", "timestamp"] if "session_id" in frame else ["timestamp"])
    events: list[dict[str, object]] = []

    for session_id, session in frame.groupby("session_id", dropna=False) if "session_id" in frame else [("", frame)]:
        for event_name, (column, positive_values) in EVENT_DEFINITIONS.items():
            if column not in session:
                continue
            active_start = None
            last_timestamp = None
            details = []
            for _, row in session.iterrows():
                value = row[column]
                active = _is_positive(value, positive_values)
                timestamp = row["timestamp"]
                if active and active_start is None:
                    active_start = timestamp
                    details = [str(value)]
                elif active:
                    details.append(str(value))
                elif active_start is not None:
                    events.append(_event_row(session_id, event_name, active_start, last_timestamp or timestamp, details))
                    active_start = None
                    details = []
                last_timestamp = timestamp
            if active_start is not None and last_timestamp is not None:
                events.append(_event_row(session_id, event_name, active_start, last_timestamp, details))

    result = pd.DataFrame(events)
    if result.empty:
        return result
    return result[result["duracion_seg"] >= min_duration_seconds].reset_index(drop=True)


def _is_positive(value: object, positive_values: set[object]) -> bool:
    if isinstance(value, str):
        return value in positive_values
    return bool(value) in positive_values


def _event_row(session_id: object, event_name: str, start, end, details: list[str]) -> dict[str, object]:
    duration = max(0.0, (end - start).total_seconds())
    return {
        "session_id": session_id,
        "evento": event_name,
        "inicio": start,
        "fin": end,
        "duracion_seg": round(duration, 2),
        "detalle": details[-1] if details else "",
    }
