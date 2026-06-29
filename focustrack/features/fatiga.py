from __future__ import annotations

import pandas as pd

TITLE = ":material/bedtime: Fatiga Visual"
ORDER = 10


def compute_fatigue(history: pd.DataFrame) -> dict:
    """Calcula nivel de fatiga a partir del historial de la sesión."""
    if history.empty:
        return {"nivel": "sin_datos", "tasa_parpadeo": 0.0, "pct_ojos_cerrados": 0.0, "ear_promedio": None}

    total = len(history)

    # tasa de parpadeo por minuto
    blink_count = 0
    if "blink_count" in history.columns:
        blink_series = pd.to_numeric(history["blink_count"], errors="coerce").dropna()
        if not blink_series.empty:
            blink_count = int(blink_series.iloc[-1] - blink_series.iloc[0])
            blink_count = max(0, blink_count)
    minutos = max(total / 60, 1 / 60)
    tasa_parpadeo = round(blink_count / minutos, 1)

    # % frames con ojos cerrados
    pct_ojos_cerrados = 0.0
    if "eyes_closed" in history.columns:
        cerrados = history["eyes_closed"].astype(bool).sum()
        pct_ojos_cerrados = round(cerrados / total * 100, 1)

    # EAR promedio
    ear_promedio = None
    if "avg_ear" in history.columns:
        ear_series = pd.to_numeric(history["avg_ear"], errors="coerce").dropna()
        if not ear_series.empty:
            ear_promedio = round(float(ear_series.mean()), 3)

    # fatigue_score promedio (0–1)
    fatigue_avg = 0.0
    if "fatigue_score" in history.columns:
        f_series = pd.to_numeric(history["fatigue_score"], errors="coerce").dropna()
        if not f_series.empty:
            fatigue_avg = float(f_series.mean())

    if fatigue_avg >= 0.5 or pct_ojos_cerrados >= 20:
        nivel = "alta"
    elif fatigue_avg >= 0.2 or pct_ojos_cerrados >= 8:
        nivel = "media"
    else:
        nivel = "baja"

    return {
        "nivel": nivel,
        "tasa_parpadeo": tasa_parpadeo,
        "pct_ojos_cerrados": pct_ojos_cerrados,
        "ear_promedio": ear_promedio,
        "fatigue_avg": round(fatigue_avg, 3),
    }


_NIVEL_COLOR = {"baja": "normal", "media": "inverse", "alta": "off", "sin_datos": "off"}
_NIVEL_ICON = {"baja": ":material/check_circle:", "media": ":material/warning:", "alta": ":material/dangerous:", "sin_datos": ":material/help:"}

