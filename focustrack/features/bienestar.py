from __future__ import annotations

import pandas as pd

TITLE = ":material/favorite: Bienestar"
ORDER = 50


def _summarize_session(analytics: dict) -> str:
    """Genera un párrafo en español resumiendo la sesión."""
    score = analytics.get("avg_productivity_score")
    snaps = analytics.get("snapshot_count", 0)
    label = analytics.get("dominant_productivity_label", "") or ""
    app = analytics.get("dominant_app", "") or ""
    breakdown = analytics.get("productivity_breakdown", {}) or {}

    if snaps == 0 or score is None:
        return "No hay suficientes datos para generar un resumen de esta sesión."

    minutos = round(snaps / 60)
    minutos_str = f"{minutos} minuto{'s' if minutos != 1 else ''}" if minutos > 0 else "menos de un minuto"

    distraido_pct = round(breakdown.get("Distraido", 0) / max(snaps, 1) * 100)
    productivo_pct = round(breakdown.get("Productivo", 0) / max(snaps, 1) * 100)

    partes = [f"Durante esta sesión de aproximadamente {minutos_str}, tu score promedio de productividad fue de {score:.1f} puntos."]

    if productivo_pct >= 60:
        partes.append(f"Estuviste productivo el {productivo_pct} % del tiempo.")
    elif distraido_pct >= 40:
        partes.append(f"Te mantuviste distraído el {distraido_pct} % del tiempo, lo que afectó tu productividad.")
    else:
        partes.append(f"Tu rendimiento fue variable durante la sesión ({label.lower() if label else 'sin clasificar'}).")

    if app:
        partes.append(f"La aplicación que más usaste fue {app}.")

    if score >= 75:
        partes.append("¡Buen trabajo! Mantén este ritmo.")
    elif score >= 45:
        partes.append("Hay oportunidad de mejorar; reduce las distracciones.")
    else:
        partes.append("Tu productividad fue baja en esta sesión. Considera revisar tus hábitos de trabajo.")

    return " ".join(partes)


def _recommendations(fatigue_avg: float, pct_ojos_cerrados: float, pct_distraido: float, bostezos: int) -> list[str]:
    recs = []
    if fatigue_avg >= 0.5 or pct_ojos_cerrados >= 20:
        recs.append(":material/bedtime: **Fatiga alta** — descansa la vista mirando a lo lejos por 20 segundos (regla 20-20-20).")
    if bostezos >= 3:
        recs.append(":material/coffee: **Varios bostezos detectados** — considera una pausa corta o levantarte a caminar.")
    if pct_distraido >= 30:
        recs.append(":material/do_not_disturb_on: **Alta distracción** — intenta silenciar notificaciones y usar modo enfocado.")
    if not recs:
        recs.append(":material/check_circle: Todo bien. Sigue con tu ritmo actual.")
    return recs


def render(st, storage, config) -> None:
    st.header(":material/favorite: Panel de Bienestar")
    st.caption("Resumen de tu sesión: fatiga, bostezos, recomendaciones y resumen automático.")

    history = storage.load_history(limit=400)
    summaries = storage.load_session_summaries(limit=10)

    # ── Métricas de bienestar ──────────────────────────────────────────────
    st.subheader("Estado de bienestar")

    fatigue_avg = 0.0
    pct_ojos = 0.0
    bostezos = 0

    if not history.empty:
        if "fatigue_score" in history.columns:
            f_s = pd.to_numeric(history["fatigue_score"], errors="coerce").dropna()
            fatigue_avg = float(f_s.mean()) if not f_s.empty else 0.0
        if "eyes_closed" in history.columns:
            cerrados = history["eyes_closed"].astype(bool).sum()
            pct_ojos = round(cerrados / len(history) * 100, 1)
        if "yawning" in history.columns:
            yawning = history["yawning"].astype(bool)
            transitions = (~yawning.shift(1, fill_value=False)) & yawning
            bostezos = int(transitions.sum())

    pct_distraido = 0.0
    if not history.empty and "productivity_label" in history.columns:
        pct_distraido = round((history["productivity_label"] == "Distraido").sum() / len(history) * 100, 1)

    # Nivel de fatiga
    if fatigue_avg >= 0.5 or pct_ojos >= 20:
        nivel_fatiga = "Alta"
    elif fatigue_avg >= 0.2 or pct_ojos >= 8:
        nivel_fatiga = "Media"
    elif history.empty:
        nivel_fatiga = "Sin datos"
    else:
        nivel_fatiga = "Baja"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Fatiga visual", nivel_fatiga)
    col2.metric("Bostezos", bostezos)
    col3.metric("% Distraído", f"{pct_distraido} %")
    col4.metric("% Ojos cerrados", f"{pct_ojos} %")

    st.divider()

    # ── Recomendaciones ───────────────────────────────────────────────────
    st.subheader(":material/tips_and_updates: Recomendaciones")
    for rec in _recommendations(fatigue_avg, pct_ojos, pct_distraido, bostezos):
        st.markdown(f"- {rec}")

    st.divider()

    # ── Resumen en lenguaje natural ───────────────────────────────────────
    st.subheader(":material/article: Resumen de la sesión")

    if summaries.empty:
        st.info("Sin sesiones registradas. Inicia el monitoreo para generar un resumen.")
        return

    session_labels = {}
    for _, row in summaries.iterrows():
        ts = str(row.get("started_at", ""))[:16]
        name = row.get("session_name", "") or ""
        score = row.get("avg_productivity_score")
        score_str = f" | Score: {score:.1f}" if score is not None else ""
        session_labels[f"{ts}{score_str}{' — ' + name if name else ''}"] = row["session_id"]

    sel = st.selectbox("Sesión a resumir", list(session_labels.keys()), key="bien_sel")
    session_id = session_labels[sel]
    analytics = storage.load_session_analytics(session_id)
    resumen = _summarize_session(analytics)

    st.info(resumen)
