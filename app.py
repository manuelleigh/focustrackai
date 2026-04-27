from __future__ import annotations

import time

import cv2
import pandas as pd
import streamlit as st

from focustrack.config import FocusTrackConfig, OptionalModels, ProductivityWeights


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


st.title("Sistema Inteligente de Monitoreo de Rendimiento y Distraccion Laboral")
st.caption("Vision por computadora + reglas de IA para estimar atencion, fatiga, postura, distracciones y actividad en PC.")

config, camera_index, refresh_seconds = _build_config()
