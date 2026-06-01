from __future__ import annotations

import time

import cv2
import pandas as pd
import streamlit as st

from focustrack.ai.classifier import model_path, predict_history, train_classifier
from focustrack.config import FocusTrackConfig, OptionalModels, ProductivityWeights
from focustrack.monitor import FocusTrackMonitor
from focustrack.monitoring.storage import StorageManager
from focustrack.reporting.session_report import (
    available_sessions,
    build_critical_events,
    build_recommendations,
    build_session_summary,
    export_report_excel,
    export_report_html,
    filter_session,
)


st.set_page_config(
    page_title="FocusTrack AI",
    page_icon="FT",
    layout="wide",
)


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #f8fafc 0%, #eef4ff 100%);
            color: #0f172a;
        }
        section[data-testid="stSidebar"] {
            background: #0f172a;
            border-right: 1px solid #1e293b;
        }
        section[data-testid="stSidebar"] * {
            color: #e5edf8 !important;
        }
        section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
            color: #cbd5e1 !important;
            font-weight: 650;
        }
        .block-container {
            max-width: 1320px;
            padding-top: 3.1rem;
            padding-bottom: 3rem;
        }
        h1 {
            color: #0f172a;
            font-size: 2.15rem !important;
            line-height: 1.15 !important;
            letter-spacing: 0 !important;
            font-weight: 760 !important;
            margin-bottom: 0.35rem !important;
        }
        h2, h3 {
            color: #0f172a;
            letter-spacing: 0 !important;
        }
        h2 {
            font-size: 1.35rem !important;
        }
        .ft-hero {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 22px 24px;
            box-shadow: 0 14px 36px rgba(15, 23, 42, 0.07);
            margin-bottom: 16px;
        }
        .ft-kicker {
            color: #1d4ed8;
            font-size: 0.78rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 8px;
        }
        .ft-subtitle {
            max-width: 900px;
            color: #64748b;
            font-size: 0.98rem;
            line-height: 1.55;
            margin-top: 8px;
        }
        .ft-notice {
            border: 1px solid #bfdbfe;
            background: #eff6ff;
            color: #1e3a8a;
            border-radius: 8px;
            padding: 13px 16px;
            margin: 14px 0 16px 0;
            font-size: 0.94rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            border-bottom: 1px solid #e2e8f0;
        }
        .stTabs [data-baseweb="tab"] {
            height: 42px;
            padding: 0 14px;
            border-radius: 6px 6px 0 0;
            font-weight: 650;
            color: #475569;
        }
        .stTabs [aria-selected="true"] {
            color: #1d4ed8 !important;
            background: #eff6ff;
        }
        div.stButton > button:first-child {
            border-radius: 7px;
            height: 42px;
            font-weight: 700;
            border: 1px solid #cbd5e1;
        }
        div.stButton > button[kind="primary"] {
            background: #1d4ed8;
            border-color: #1d4ed8;
        }
        div.stButton > button[kind="primary"]:hover {
            background: #1e3a8a;
            border-color: #1e3a8a;
        }
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 14px 16px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
        }
        [data-testid="stMetricLabel"] p {
            color: #64748b !important;
            font-weight: 650;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            overflow: hidden;
        }
        code {
            color: #0f766e !important;
            background: #ecfdf5 !important;
            border-radius: 4px;
            padding: 0.1rem 0.25rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


_inject_styles()


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


def _render_report_panel(history: pd.DataFrame, refresh_seconds: float, data_dir) -> None:
    st.subheader("Reporte exportable")
    if history.empty:
        st.info("Aun no hay datos para generar reportes.")
        return

    sessions = ["Todas"] + available_sessions(history)
    selected_session = st.selectbox("Sesion a reportar", sessions, index=len(sessions) - 1)
    session_history = filter_session(history, selected_session)

    if session_history.empty:
        st.warning("La sesion seleccionada no tiene datos.")
        return

    summary = build_session_summary(session_history, refresh_seconds)
    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("Promedio", f"{summary['score_promedio']:.1f}")
    metric_2.metric("Tiempo total", f"{summary['tiempo_total_min']:.2f} min")
    metric_3.metric("Tiempo distraido", f"{summary['tiempo_distraido_min']:.2f} min")
    metric_4.metric("Eventos criticos", str(summary["eventos_criticos"]))

    st.markdown("**Recomendaciones automaticas**")
    for recommendation in build_recommendations(summary, session_history):
        st.write(f"- {recommendation}")

    critical_events = build_critical_events(session_history)
    if not critical_events.empty:
        st.markdown("**Eventos criticos recientes**")
        st.dataframe(critical_events, use_container_width=True, hide_index=True)

    excel_bytes = export_report_excel(session_history, refresh_seconds)
    html_report = export_report_html(session_history, refresh_seconds).encode("utf-8")
    report_name = str(selected_session).replace(" ", "_").lower()

    download_left, download_right = st.columns(2)
    with download_left:
        st.download_button(
            "Descargar Excel",
            data=excel_bytes,
            file_name=f"focustrack_reporte_{report_name}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with download_right:
        st.download_button(
            "Descargar HTML",
            data=html_report,
            file_name=f"focustrack_reporte_{report_name}.html",
            mime="text/html",
            use_container_width=True,
        )


def _render_ai_panel(history: pd.DataFrame, data_dir) -> None:
    st.subheader("Modelo IA de clasificacion")
    st.caption("Entrena un Random Forest para clasificar cada registro como Productivo, Regular o Distraido.")

    model_file = model_path(data_dir)
    if model_file.exists():
        st.success(f"Modelo disponible: {model_file.name}")
    else:
        st.info("Aun no hay modelo entrenado. Puedes entrenarlo con historico real o con datos simulados para demo.")

    if st.button("Entrenar / actualizar modelo IA", type="primary", use_container_width=True):
        result = train_classifier(history, data_dir)
        if result.trained:
            st.success(result.message)
            st.write(f"Fuente: `{result.source}` | Filas usadas: `{result.rows}` | Clases: `{', '.join(result.labels)}`")
            if result.accuracy is not None:
                st.metric("Accuracy de validacion", f"{result.accuracy:.2%}")
            if result.report:
                st.text(result.report)
        else:
            st.error(result.message)

    predictions = predict_history(history.tail(200), data_dir)
    if predictions.empty:
        return

    st.markdown("**Predicciones recientes del modelo**")
    columns = [
        "timestamp",
        "productivity_label",
        "ai_prediction",
        "ai_confidence",
        "productivity_score",
        "attention_state",
        "object_state",
        "screen_category",
    ]
    available_columns = [column for column in columns if column in predictions.columns]
    st.dataframe(predictions.tail(30)[available_columns], use_container_width=True, hide_index=True)

    agreement = (predictions["productivity_label"].astype(str) == predictions["ai_prediction"].astype(str)).mean()
    st.metric("Coincidencia IA vs regla", f"{agreement:.2%}")


def _frame_to_rgb(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


st.markdown(
    """
    <div class="ft-hero">
        <div class="ft-kicker">Analitica laboral asistida por IA</div>
        <h1>FocusTrack AI</h1>
        <div class="ft-subtitle">
            Sistema local para analizar atencion visual, postura, distracciones y actividad digital mediante
            vision por computadora, reglas explicables y clasificacion supervisada.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="ft-notice">
        Uso responsable: el sistema requiere consentimiento informado. Las capturas estan desactivadas por defecto;
        el historial y los modelos se almacenan localmente.
    </div>
    """,
    unsafe_allow_html=True,
)

config, camera_index, refresh_seconds = _build_config()
storage = StorageManager(config.data_dir)

if "monitor" not in st.session_state:
    st.session_state.monitor = None
if "monitor_running" not in st.session_state:
    st.session_state.monitor_running = False
if "last_frame" not in st.session_state:
    st.session_state.last_frame = None

controls_left, controls_right = st.columns([1, 3])
with controls_left:
    start_clicked = st.button("Iniciar monitoreo", use_container_width=True, type="primary", disabled=st.session_state.monitor_running)
with controls_right:
    stop_clicked = st.button("Detener monitoreo", use_container_width=True, disabled=not st.session_state.monitor_running)

if start_clicked:
    try:
        if st.session_state.monitor is not None:
            st.session_state.monitor.stop()
        monitor = FocusTrackMonitor(config=config, camera_index=camera_index)
        monitor.start()
        st.session_state.monitor = monitor
        st.session_state.monitor_running = True
    except Exception as exc:
        st.error(f"No fue posible iniciar el monitoreo: {exc}")
        st.session_state.monitor = None
        st.session_state.monitor_running = False

if stop_clicked:
    if st.session_state.monitor is not None:
        st.session_state.monitor.stop()
    st.session_state.monitor = None
    st.session_state.monitor_running = False

status_col, info_col = st.columns([1.2, 2])
with status_col:
    history = storage.load_history(limit=200)
    _render_kpis(history)
with info_col:
    st.markdown(
        """
        **Que detecta esta demo**

        - Rostro, ojos, EAR y mirada con `MediaPipe` si la instalacion expone `solutions`; si no, usa fallback con `OpenCV`.
        - Postura corporal con `MediaPipe Pose` o una estimacion visual de respaldo con `OpenCV`.
        - Celular y objetos si `YOLO` esta disponible; si no, usa heuristicas de manos y ausencia.
        - Aplicacion activa y clasificacion trabajo vs distraccion en el escritorio.
        """
    )

frame_placeholder = st.empty()
alert_placeholder = st.empty()

if st.session_state.monitor_running and st.session_state.monitor is not None:
    try:
        snapshot, frame = st.session_state.monitor.process_next()
        st.session_state.last_frame = frame
        history = storage.load_history(limit=200)

        if snapshot.productivity_label == "Distraido":
            alert_placeholder.error("Nivel de distraccion alto. Conviene revisar la causa o activar alertas.")
        elif snapshot.productivity_label == "Regular":
            alert_placeholder.warning("Atencion irregular detectada. Revisa postura, mirada o actividad en pantalla.")
        else:
            alert_placeholder.success("Comportamiento dentro del rango productivo.")
    except Exception as exc:
        alert_placeholder.error(f"Error de monitoreo: {exc}")
        if st.session_state.monitor is not None:
            st.session_state.monitor.stop()
        st.session_state.monitor = None
        st.session_state.monitor_running = False

if st.session_state.last_frame is not None:
    frame_placeholder.image(_frame_to_rgb(st.session_state.last_frame), caption="Vista analizada", use_container_width=True)
else:
    frame_placeholder.info("Cuando inicies el monitoreo se mostrara aqui el frame anotado en tiempo real.")

history = storage.load_history(limit=400)
_render_history(history, refresh_seconds)

full_history = storage.load_history()
report_tab, ai_tab = st.tabs(["Reporte de sesion", "Clasificador IA"])
with report_tab:
    _render_report_panel(full_history, refresh_seconds, config.data_dir)
with ai_tab:
    _render_ai_panel(full_history, config.data_dir)

if st.session_state.monitor_running:
    time.sleep(refresh_seconds)
    st.rerun()
