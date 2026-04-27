from __future__ import annotations

import time

import cv2
import pandas as pd
import streamlit as st

from focustrack.config import FocusTrackConfig, OptionalModels, ProductivityWeights
from focustrack.monitoring.storage import StorageManager


st.set_page_config(
    page_title="FocusTrack AI",
    page_icon=":brain:",
    layout="wide",
)


def _normalize_weights(raw_weights: dict[str, float]) -> ProductivityWeights:
    total = sum(raw_weights.values()) or 1.0
    return ProductivityWeights(
        attention=raw_weights["attention"] / total,
        phone=raw_weights["phone"] / total,
        posture=raw_weights["posture"] / total,
        screen=raw_weights["screen"] / total,
    )


def _build_config() -> tuple[FocusTrackConfig, int, float]:
    st.sidebar.header("Configuracion")
    camera_index = int(st.sidebar.number_input("Indice de camara", min_value=0, max_value=5, value=0, step=1))
    refresh_seconds = float(st.sidebar.slider("Intervalo de muestreo (seg)", min_value=0.5, max_value=3.0, value=1.0, step=0.5))
    enable_yolo = st.sidebar.checkbox("Activar YOLO si esta disponible", value=False)
    enable_dlib = st.sidebar.checkbox("Activar respaldo con dlib si esta disponible", value=False)
    capture_screen = st.sidebar.checkbox("Guardar capturas de pantalla", value=False)

    st.sidebar.subheader("Pesos del score")
    raw_weights = {
        "attention": float(st.sidebar.slider("Atencion visual", min_value=5, max_value=70, value=40, step=5)),
        "phone": float(st.sidebar.slider("Celular / objetos", min_value=5, max_value=50, value=20, step=5)),
        "posture": float(st.sidebar.slider("Postura", min_value=5, max_value=40, value=15, step=5)),
        "screen": float(st.sidebar.slider("Actividad en PC", min_value=5, max_value=60, value=25, step=5)),
    }
    weights = _normalize_weights(raw_weights)

    config = FocusTrackConfig(
        weights=weights,
        models=OptionalModels(
            enable_yolo=enable_yolo,
            enable_dlib=enable_dlib,
        ),
        screen_capture_enabled=capture_screen,
    )
    return config, camera_index, refresh_seconds


def _render_kpis(history: pd.DataFrame) -> None:
    if history.empty:
        st.info("Aun no hay registros. Inicia el monitoreo para generar datos.")
        return

    last = history.iloc[-1]
    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("Score actual", f"{last['productivity_score']:.1f}")
    metric_2.metric("Clasificacion", str(last["productivity_label"]))
    metric_3.metric("Atencion", str(last["attention_state"]))
    metric_4.metric("App activa", str(last["active_app"]))


def _render_history(history: pd.DataFrame, refresh_seconds: float) -> None:
    if history.empty:
        return

    history = history.copy()
    history = history.dropna(subset=["timestamp"])
    if history.empty:
        return

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.subheader("Score en el tiempo")
        score_chart = history.set_index("timestamp")[["productivity_score"]].tail(120)
        st.line_chart(score_chart)

    with chart_right:
        st.subheader("Tiempo estimado por aplicacion")
        time_by_app = (
            history.tail(120)
            .groupby("active_app")
            .size()
            .sort_values(ascending=False)
            .head(8)
            .mul(refresh_seconds)
            .rename("segundos")
        )
        st.bar_chart(time_by_app)

    st.subheader("Ultimos eventos")
    cols = [
        "timestamp",
        "productivity_score",
        "productivity_label",
        "attention_state",
        "posture_state",
        "object_state",
        "active_app",
        "screen_category",
    ]
    st.dataframe(history.tail(20)[cols], use_container_width=True, hide_index=True)


def _frame_to_rgb(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


st.title("Sistema Inteligente de Monitoreo de Rendimiento y Distraccion Laboral")
st.caption("Vision por computadora + reglas de IA para estimar atencion, fatiga, postura, distracciones y actividad en PC.")

config, camera_index, refresh_seconds = _build_config()
storage = StorageManager(config.data_dir)

status_col, info_col = st.columns([1.2, 2])
with status_col:
    history = storage.load_history(limit=200)
    _render_kpis(history)
with info_col:
    st.markdown(
        """
        **Que detecta esta demo**

        - Rostro, ojos, EAR y mirada con `OpenCV + MediaPipe`, con soporte opcional para `dlib`.
        - Postura corporal usando `MediaPipe Pose`.
        - Celular y objetos si `YOLO` esta disponible; si no, usa heuristicas de manos y ausencia.
        - Aplicacion activa y clasificacion trabajo vs distraccion en el escritorio.
        """
    )

frame_placeholder = st.empty()
alert_placeholder = st.empty()

history = storage.load_history(limit=400)
_render_history(history, refresh_seconds)
