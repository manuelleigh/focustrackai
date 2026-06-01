from __future__ import annotations

from io import BytesIO

import pandas as pd


CRITICAL_STATES = {"Distraido"}


def available_sessions(history: pd.DataFrame) -> list[str]:
    if history.empty or "session_id" not in history.columns:
        return []

    sessions = history["session_id"].dropna().astype(str).unique().tolist()
    return sorted(sessions)


def filter_session(history: pd.DataFrame, session_id: str | None) -> pd.DataFrame:
    if history.empty:
        return history
    if not session_id or session_id == "Todas":
        return history.copy()

    return history[history["session_id"].astype(str) == str(session_id)].copy()


def build_session_summary(session_history: pd.DataFrame, sample_seconds: float = 1.0) -> dict[str, object]:
    if session_history.empty:
        return {
            "registros": 0,
            "score_promedio": 0.0,
            "score_minimo": 0.0,
            "score_maximo": 0.0,
            "clasificacion_dominante": "Sin datos",
            "tiempo_total_min": 0.0,
            "tiempo_distraido_min": 0.0,
            "porcentaje_distraido": 0.0,
            "eventos_criticos": 0,
            "app_principal": "Sin datos",
        }

    total_records = len(session_history)
    distracted_mask = session_history["productivity_label"].isin(CRITICAL_STATES)
    distracted_records = int(distracted_mask.sum())
    app_mode = session_history["active_app"].mode()
    label_mode = session_history["productivity_label"].mode()

    return {
        "registros": total_records,
        "score_promedio": round(float(session_history["productivity_score"].mean()), 2),
        "score_minimo": round(float(session_history["productivity_score"].min()), 2),
        "score_maximo": round(float(session_history["productivity_score"].max()), 2),
        "clasificacion_dominante": str(label_mode.iloc[0]) if not label_mode.empty else "Sin datos",
        "tiempo_total_min": round((total_records * sample_seconds) / 60.0, 2),
        "tiempo_distraido_min": round((distracted_records * sample_seconds) / 60.0, 2),
        "porcentaje_distraido": round((distracted_records / max(total_records, 1)) * 100.0, 2),
        "eventos_criticos": distracted_records,
        "app_principal": str(app_mode.iloc[0]) if not app_mode.empty else "Sin datos",
    }


def build_critical_events(session_history: pd.DataFrame, limit: int = 30) -> pd.DataFrame:
    if session_history.empty:
        return pd.DataFrame()

    event_columns = [
        "timestamp",
        "productivity_score",
        "productivity_label",
        "attention_state",
        "posture_state",
        "object_state",
        "active_app",
        "screen_category",
    ]
    available_columns = [column for column in event_columns if column in session_history.columns]
    events = session_history[session_history["productivity_label"].isin(CRITICAL_STATES)]
    return events[available_columns].tail(limit)


def build_recommendations(summary: dict[str, object], session_history: pd.DataFrame) -> list[str]:
    if session_history.empty:
        return ["No hay datos suficientes para generar recomendaciones."]

    recommendations: list[str] = []
    distracted_pct = float(summary["porcentaje_distraido"])
    avg_score = float(summary["score_promedio"])

    if avg_score >= 75:
        recommendations.append("Mantener las condiciones actuales: el promedio de productividad es saludable.")
    elif avg_score >= 45:
        recommendations.append("Revisar pausas activas y ergonomia: el rendimiento promedio es regular.")
    else:
        recommendations.append("Intervenir la sesion: el rendimiento promedio esta en zona de distraccion.")

    if distracted_pct >= 30:
        recommendations.append("Reducir distractores externos: mas del 30% de la sesion fue clasificada como distraida.")

    if "phone_detected" in session_history and session_history["phone_detected"].astype(bool).mean() >= 0.1:
        recommendations.append("Aplicar politica de celular: se detecto uso frecuente durante la jornada.")

    if "posture_state" in session_history and (session_history["posture_state"] == "encorvada").mean() >= 0.2:
        recommendations.append("Ajustar silla, monitor o postura: se detecta mala postura de forma repetida.")

    if "attention_state" in session_history and (session_history["attention_state"] == "somnoliento").mean() >= 0.1:
        recommendations.append("Evaluar fatiga visual o cansancio: hubo eventos de ojos cerrados/somnolencia.")

    if "screen_category" in session_history and (session_history["screen_category"] == "distraccion").mean() >= 0.15:
        recommendations.append("Revisar uso de aplicaciones no laborales durante el turno.")

    return recommendations


def build_report_tables(session_history: pd.DataFrame, sample_seconds: float = 1.0) -> dict[str, pd.DataFrame]:
    summary = build_session_summary(session_history, sample_seconds)
    recommendations = build_recommendations(summary, session_history)
    critical_events = build_critical_events(session_history)

    summary_table = pd.DataFrame(
        [{"metrica": key, "valor": value} for key, value in summary.items()]
    )
    recommendations_table = pd.DataFrame({"recomendacion": recommendations})

    if not session_history.empty:
        by_app = (
            session_history.groupby(["active_app", "screen_category"], dropna=False)
            .agg(
                registros=("active_app", "size"),
                score_promedio=("productivity_score", "mean"),
            )
            .reset_index()
        )
        by_app["minutos_estimados"] = (by_app["registros"] * sample_seconds / 60.0).round(2)
        by_app["score_promedio"] = by_app["score_promedio"].round(2)
    else:
        by_app = pd.DataFrame()

    return {
        "resumen": summary_table,
        "recomendaciones": recommendations_table,
        "eventos_criticos": critical_events,
        "apps": by_app,
        "detalle": session_history,
    }


def export_report_excel(session_history: pd.DataFrame, sample_seconds: float = 1.0) -> bytes:
    tables = build_report_tables(session_history, sample_seconds)
    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, table in tables.items():
            safe_name = sheet_name[:31]
            table.to_excel(writer, index=False, sheet_name=safe_name)

    return output.getvalue()


def export_report_html(session_history: pd.DataFrame, sample_seconds: float = 1.0) -> str:
    tables = build_report_tables(session_history, sample_seconds)
    summary = tables["resumen"].to_html(index=False)
    recommendations = tables["recomendaciones"].to_html(index=False)
    events = tables["eventos_criticos"].to_html(index=False)
    apps = tables["apps"].to_html(index=False)

    return f"""
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>Reporte FocusTrack AI</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #20242a; }}
    h1, h2 {{ color: #102a43; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; }}
    th, td {{ border: 1px solid #d8dee9; padding: 8px; text-align: left; }}
    th {{ background: #edf2f7; }}
  </style>
</head>
<body>
  <h1>Reporte FocusTrack AI</h1>
  <h2>Resumen</h2>
  {summary}
  <h2>Recomendaciones</h2>
  {recommendations}
  <h2>Eventos criticos</h2>
  {events}
  <h2>Uso por aplicacion</h2>
  {apps}
</body>
</html>
"""
