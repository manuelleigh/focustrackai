from __future__ import annotations

import pandas as pd

TITLE = ":material/sentiment_very_dissatisfied: Bostezo"
ORDER = 20


def compute_yawns(history: pd.DataFrame) -> dict:
    """Cuenta bostezos detectados en el historial."""
    if history.empty or "yawning" not in history.columns:
        return {"total_bostezos": 0, "pct_bostezando": 0.0}

    yawning = history["yawning"].astype(bool)
    # Contar transiciones False→True como eventos de bostezo
    transitions = (~yawning.shift(1, fill_value=False)) & yawning
    total = int(transitions.sum())
    pct = round(yawning.sum() / max(len(history), 1) * 100, 1)
    return {"total_bostezos": total, "pct_bostezando": pct}


def render(st, storage, config) -> None:
    st.header(":material/sentiment_very_dissatisfied: Detección de Bostezo")
    st.caption("Detecta bostezos midiendo la apertura de la boca (MAR) durante el monitoreo.")

    history = storage.load_history(limit=400)
    result = compute_yawns(history)

    col1, col2 = st.columns(2)
    col1.metric("Bostezos detectados", result["total_bostezos"])
    col2.metric("% tiempo bostezando", f"{result['pct_bostezando']} %")

    st.divider()

    if "yawning" not in (history.columns if not history.empty else []):
        st.info("Sin datos de bostezo. El monitoreo debe ejecutarse con la versión v4 para capturar esta señal.")
        return

    if result["total_bostezos"] >= 5:
        st.error(":material/dangerous: Muchos bostezos detectados — considera tomar una pausa.")
    elif result["total_bostezos"] >= 2:
        st.warning(":material/warning: Algunos bostezos detectados — vigila tu nivel de energía.")
    else:
        st.success(":material/check_circle: Sin bostezos significativos.")

    if not history.empty and "mouth_aspect_ratio" in history.columns:
        st.subheader("Evolución del MAR (apertura de boca)")
        mar = history[["timestamp", "mouth_aspect_ratio"]].copy()
        mar["mouth_aspect_ratio"] = pd.to_numeric(mar["mouth_aspect_ratio"], errors="coerce")
        mar = mar.dropna(subset=["mouth_aspect_ratio"])
        if not mar.empty:
            st.line_chart(mar.set_index("timestamp")["mouth_aspect_ratio"])
            st.caption(f"Umbral de bostezo: MAR > {config.thresholds.mar_open}")
