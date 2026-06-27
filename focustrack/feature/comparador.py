from __future__ import annotations

import pandas as pd

TITLE = ":material/compare_arrows: Comparador de Sesiones"
ORDER = 40


def compare_sessions(storage, id_a: str, id_b: str) -> pd.DataFrame:
    """Devuelve DataFrame con métricas clave de dos sesiones para comparar."""

    def _stats(session_id: str) -> dict:
        analytics = storage.load_session_analytics(session_id)
        history = storage.load_history(limit=None, session_id=session_id)

        pct_distraido = 0.0
        if not history.empty and "productivity_label" in history.columns:
            total = len(history)
            distraido = (history["productivity_label"] == "Distraido").sum()
            pct_distraido = round(distraido / max(total, 1) * 100, 1)

        return {
            "Score promedio": round(analytics.get("avg_productivity_score") or 0, 1),
            "Snapshots": analytics.get("snapshot_count", 0),
            "% Distraído": pct_distraido,
            "App dominante": analytics.get("dominant_app", "—") or "—",
            "Estado dominante": analytics.get("dominant_productivity_label", "—") or "—",
        }

    stats_a = _stats(id_a)
    stats_b = _stats(id_b)

    df = pd.DataFrame(
        {
            "Métrica": list(stats_a.keys()),
            id_a[:20]: list(stats_a.values()),
            id_b[:20]: list(stats_b.values()),
        }
    )

    return df

def render(st, storage, config) -> None:
    st.header(":material/compare_arrows: Comparador de Sesiones")
    st.caption("Compara dos sesiones lado a lado para ver evolución o diferencias.")

    summaries = storage.load_session_summaries(limit=20)

    if summaries.empty or len(summaries) < 2:
        st.info(
            "Necesitas al menos 2 sesiones grabadas para comparar. "
            "Inicia y detén el monitoreo en momentos distintos."
        )
        return

    # Etiquetas legibles para los selectores
    def _label(row) -> str:
        name = row.get("session_name", "") or ""
        ts = str(row.get("started_at", ""))[:16]
        score = row.get("avg_productivity_score")
        score_str = f" | Score: {score:.1f}" if score is not None else ""
        return f"{ts}{score_str}{' — ' + name if name else ''}"

    options = {_label(r): r["session_id"] for _, r in summaries.iterrows()}
    labels = list(options.keys())

    col_a, col_b = st.columns(2)

    with col_a:
        sel_a = st.selectbox(
            "Sesión A",
            labels,
            index=0,
            key="cmp_a",
        )

    with col_b:
        sel_b = st.selectbox(
            "Sesión B",
            labels,
            index=min(1, len(labels) - 1),
            key="cmp_b",
        )

    if sel_a == sel_b:
        st.warning("Selecciona dos sesiones distintas.")
        return

    id_a = options[sel_a]
    id_b = options[sel_b]

    df = compare_sessions(storage, id_a, id_b)

    st.divider()
    st.dataframe(df, hide_index=True, use_container_width=True)

    # KPIs lado a lado
    st.divider()
    st.subheader("Score promedio")

    try:
        score_a = float(
            df.loc[
                df["Métrica"] == "Score promedio",
                df.columns[1],
            ].values[0]
        )

        score_b = float(
            df.loc[
                df["Métrica"] == "Score promedio",
                df.columns[2],
            ].values[0]
        )

        kpi_a, kpi_b = st.columns(2)

        kpi_a.metric(
            "Sesión A",
            f"{score_a:.1f}",
            delta=f"{score_a - score_b:+.1f}",
        )

        kpi_b.metric(
            "Sesión B",
            f"{score_b:.1f}",
        )

    except (IndexError, ValueError):
        pass