from __future__ import annotations

import time

import cv2
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="FocusTrack AI",
    page_icon=":brain:",
    layout="wide",
)

st.title("Sistema Inteligente de Monitoreo de Rendimiento y Distraccion Laboral")
st.caption("Vision por computadora + reglas de IA para estimar atencion, fatiga, postura, distracciones y actividad en PC.")
