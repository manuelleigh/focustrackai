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
