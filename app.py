from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from focustrack.config import FocusTrackConfig, OptionalModels, ProductivityWeights
from focustrack.engine.evaluation import build_labeled_dataset, evaluate_label_predictions
from focustrack.monitor import FocusTrackMonitor
from focustrack.monitoring.storage import StorageManager


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
    camera_index = int(
        st.sidebar.number_input("Indice de camara", min_value=0, max_value=5, value=0, step=1)
    )
    refresh_seconds = float(
        st.sidebar.slider(
            "Intervalo de muestreo (seg)",
            min_value=0.5,
            max_value=3.0,
            value=1.0,
            step=0.5,
        )
    )
    enable_yolo = st.sidebar.checkbox("Activar YOLO si esta disponible", value=False)
    enable_dlib = st.sidebar.checkbox("Activar respaldo con dlib si esta disponible", value=False)
    capture_screen = st.sidebar.checkbox("Guardar capturas de pantalla", value=False)

    st.sidebar.subheader("Pesos del score")
    raw_weights = {
        "attention": float(
            st.sidebar.slider("Atencion visual", min_value=5, max_value=70, value=40, step=5)
        ),
        "phone": float(
            st.sidebar.slider("Celular / objetos", min_value=5, max_value=50, value=20, step=5)
        ),
        "posture": float(
            st.sidebar.slider("Postura", min_value=5, max_value=40, value=15, step=5)
        ),
        "screen": float(
            st.sidebar.slider("Actividad en PC", min_value=5, max_value=60, value=25, step=5)
        ),
    }
    weights = _normalize_weights(raw_weights)

    config = FocusTrackConfig(
        weights=weights,
        models=OptionalModels(enable_yolo=enable_yolo, enable_dlib=enable_dlib),
        screen_capture_enabled=capture_screen,
    )
    return config, camera_index, refresh_seconds


def _frame_to_rgb(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _ensure_session_state() -> None:
    defaults = {
        "monitor": None,
        "monitor_running": False,
        "last_frame": None,
        "last_snapshot": None,
        "active_session_id": "",
        "last_alert_signature": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _get_active_session_id() -> str:
    monitor = st.session_state.get("monitor")
    if monitor is not None:
        return monitor.session_id
    return str(st.session_state.get("active_session_id", ""))


def _parse_optional_iso_datetime(raw_value: str) -> str | None:
    value = raw_value.strip()
    if not value:
        return None
    return datetime.fromisoformat(value).isoformat()


def _save_session_note(
    storage: StorageManager,
    session_id: str,
    name: str,
    description: str,
    approved_for_training: bool,
    status: str,
) -> None:
    storage.upsert_session_note(
        session_id=session_id,
        name=name,
        description=description,
        approved_for_training=approved_for_training,
        status=status,
    )
    storage.append_audit_event(
        "session_note_updated",
        {
            "name": name,
            "status": status,
            "approved_for_training": approved_for_training,
        },
        session_id=session_id,
    )


def _register_human_label(
    storage: StorageManager,
    session_id: str,
    label: str,
    start_time: str | None,
    end_time: str | None,
    notes: str,
) -> None:
    storage.append_human_label(
        session_id=session_id,
        label=label,
        start_time=start_time,
        end_time=end_time,
        notes=notes,
    )
    storage.append_audit_event(
        "human_label_registered",
        {
            "label": label,
            "start_time": start_time,
            "end_time": end_time,
            "notes": notes,
        },
        session_id=session_id,
    )


def _save_alert_rules(storage: StorageManager, rules: list[dict[str, object]]) -> None:
    for rule in rules:
        threshold, window_seconds, severity = storage.validate_alert_rule(
            threshold=float(rule["threshold"]),
            window_seconds=float(rule["window_seconds"]),
            severity=str(rule["severity"]),
        )
        storage.upsert_alert_rule(
            rule_key=str(rule["rule_key"]),
            enabled=bool(rule["enabled"]),
            threshold=threshold,
            window_seconds=window_seconds,
            severity=severity,
        )


def _status_label(status: str) -> str:
    labels = {
        "registrada": "Registrada",
        "activa": "Activa",
        "en_revision": "En revision",
        "finalizada": "Finalizada",
    }
    return labels.get(status, status)


def _build_session_selector(storage: StorageManager) -> tuple[str, pd.DataFrame]:
    sessions = storage.load_session_summaries(limit=30)
    if sessions.empty:
        return _get_active_session_id(), sessions

    options: dict[str, str] = {}
    for row in sessions.to_dict(orient="records"):
        session_id = str(row["session_id"])
        session_name = str(row["session_name"] or f"Sesion {session_id}")
        status = _status_label(str(row["session_status"] or "registrada"))
        avg_score = row.get("avg_productivity_score")
        score_text = f"{float(avg_score):.1f}" if pd.notna(avg_score) else "s/d"
        options[f"{session_name} | {status} | score {score_text}"] = session_id

    selected_label = st.sidebar.selectbox(
        "Sesion para revisar",
        options=list(options.keys()),
        index=0,
    )
    return options[selected_label], sessions


def _render_kpis(history: pd.DataFrame) -> None:
    if history.empty:
        st.info("Aun no hay registros. Inicia el monitoreo para generar datos.")
        return

    last = history.iloc[-1]
    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("Score actual", f"{last['productivity_score']:.1f}")
    metric_2.metric("Clasificacion", str(last.get("productivity_label", "")))
    metric_3.metric("Atencion", str(last.get("attention_state", "")))
    metric_4.metric("App activa", str(last.get("active_app", "")))


def _render_history(history: pd.DataFrame, refresh_seconds: float, session_id: str) -> None:
    if history.empty:
        if session_id:
            st.info("La sesion seleccionada aun no tiene historial consolidado.")
        return

    history = history.copy().dropna(subset=["timestamp"])
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
    visible_columns = [column for column in cols if column in history.columns]
    st.subheader("Ultimos eventos")
    st.dataframe(history.tail(20)[visible_columns], use_container_width=True, hide_index=True)


def _render_storage_health(storage: StorageManager) -> None:
    st.subheader("Salud del storage")
    health = storage.storage_health()
    cols = st.columns(5)
    cols[0].metric("Snapshots", str(health["snapshots"]))
    cols[1].metric("Auditoria", str(health["audit_events"]))
    cols[2].metric("Etiquetas", str(health["human_labels"]))
    cols[3].metric("Notas", str(health["session_notes"]))
    cols[4].metric("Alertas", str(health["alert_rules"]))
    st.caption(f"SQLite: {health['sqlite_path']} | CSV: {health['csv_path']}")


def _build_export_path(storage: StorageManager, session_id: str) -> Path:
    export_name = f"historial_{session_id or 'global'}.csv"
    return storage.data_dir / "exports" / export_name


def _build_audit_export_path(storage: StorageManager, session_id: str) -> Path:
    export_name = f"auditoria_{session_id or 'global'}.csv"
    return storage.data_dir / "exports" / export_name


def _render_history_export(storage: StorageManager, session_id: str) -> None:
    st.subheader("Exportacion")
    export_limit = int(
        st.number_input(
            "Maximo de filas para exportar",
            min_value=1,
            max_value=5000,
            value=500,
            step=50,
        )
    )
    if st.button("Generar exportacion CSV", use_container_width=True):
        export_path = _build_export_path(storage, session_id)
        storage.export_history_csv(
            destination=export_path,
            session_id=session_id or None,
            limit=export_limit,
        )
        st.success(f"Exportacion generada en: {export_path}")

    if st.button("Generar exportacion de auditoria", use_container_width=True):
        export_path = _build_audit_export_path(storage, session_id)
        storage.export_audit_csv(
            destination=export_path,
            session_id=session_id or None,
            limit=export_limit,
        )
        st.success(f"Auditoria exportada en: {export_path}")


def _render_session_summary(selected_session_id: str, sessions: pd.DataFrame) -> None:
    st.subheader("Resumen de sesion")
    if not selected_session_id or sessions.empty:
        st.info("Aun no hay sesiones persistidas para resumir.")
        return

    current = sessions[sessions["session_id"] == selected_session_id]
    if current.empty:
        st.info("La sesion seleccionada aun no tiene snapshots consolidados.")
        return

    row = current.iloc[0]
    cols = st.columns(4)
    cols[0].metric("Sesion", str(row["session_id"]))
    cols[1].metric("Snapshots", str(int(row["snapshot_count"])))
    avg_score = row["avg_productivity_score"]
    cols[2].metric("Score promedio", f"{float(avg_score):.1f}" if pd.notna(avg_score) else "s/d")
    cols[3].metric("Estado", _status_label(str(row["session_status"] or "registrada")))
    st.caption(
        f"Inicio: {row['started_at']} | Ultima actividad: {row['last_seen_at']} | "
        f"Aprobada para entrenamiento: {'Si' if bool(row['approved_for_training']) else 'No'}"
    )


def _render_session_analytics(storage: StorageManager, session_id: str) -> None:
    st.subheader("Analitica de sesion")
    if not session_id:
        st.info("Selecciona o inicia una sesion para ver analitica derivada.")
        return

    analytics = storage.load_session_analytics(session_id)
    if int(analytics["snapshot_count"]) == 0:
        st.info("La sesion seleccionada aun no tiene snapshots para analisis.")
        return

    cols = st.columns(3)
    avg_score = analytics["avg_productivity_score"]
    cols[0].metric(
        "Score promedio consolidado",
        f"{float(avg_score):.1f}" if avg_score is not None else "s/d",
    )
    cols[1].metric(
        "Etiqueta dominante",
        str(analytics["dominant_productivity_label"] or "s/d"),
    )
    cols[2].metric(
        "Aplicacion principal",
        str(analytics["dominant_app"] or "s/d"),
    )

    left, right = st.columns(2)
    with left:
        st.caption("Distribucion de productividad")
        productivity_breakdown = analytics["productivity_breakdown"]
        st.dataframe(
            pd.DataFrame(
                [
                    {"etiqueta": label, "muestras": count}
                    for label, count in productivity_breakdown.items()
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    with right:
        st.caption("Distribucion de atencion")
        attention_breakdown = analytics["attention_breakdown"]
        st.dataframe(
            pd.DataFrame(
                [
                    {"estado": state, "muestras": count}
                    for state, count in attention_breakdown.items()
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )


def _rule_label(rule_key: str) -> str:
    labels = {
        "productivity_low": "Productividad baja",
        "productivity_medium": "Productividad en observacion",
        "success": "Sin alerta",
    }
    return labels.get(rule_key, rule_key)


def _evaluate_alert(snapshot, rules_map: dict[str, dict[str, object]]) -> dict[str, object]:
    if snapshot is None:
        return {
            "severity": "info",
            "message": "Aun no hay datos para evaluar alertas.",
            "rule_key": "success",
            "score": None,
        }

    low_rule = rules_map.get("productivity_low")
    medium_rule = rules_map.get("productivity_medium")
    score = float(snapshot.productivity_score)

    if low_rule and bool(low_rule.get("enabled")) and score < float(low_rule.get("threshold", 45.0)):
        return {
            "severity": str(low_rule.get("severity", "warning")),
            "message": f"Score por debajo del umbral critico ({score:.1f}).",
            "rule_key": "productivity_low",
            "score": score,
        }

    if medium_rule and bool(medium_rule.get("enabled")) and score < float(medium_rule.get("threshold", 75.0)):
        return {
            "severity": str(medium_rule.get("severity", "info")),
            "message": f"Score en zona de observacion ({score:.1f}).",
            "rule_key": "productivity_medium",
            "score": score,
        }

    return {
        "severity": "success",
        "message": f"Score dentro del rango esperado ({score:.1f}).",
        "rule_key": "success",
        "score": score,
    }


def _build_alert_signature(alert_result: dict[str, object], session_id: str) -> str:
    return (
        f"{session_id}|{alert_result['rule_key']}|"
        f"{alert_result['severity']}|{alert_result['score']}"
    )


def _register_alert_event_if_needed(
    storage: StorageManager,
    session_id: str,
    alert_result: dict[str, object],
) -> None:
    if not session_id or alert_result["rule_key"] == "success":
        return

    signature = _build_alert_signature(alert_result, session_id)
    if st.session_state.get("last_alert_signature") == signature:
        return

    storage.append_audit_event(
        "alert_triggered",
        {
            "rule_key": alert_result["rule_key"],
            "rule_label": _rule_label(str(alert_result["rule_key"])),
            "severity": alert_result["severity"],
            "score": alert_result["score"],
            "message": alert_result["message"],
        },
        session_id=session_id,
    )
    st.session_state["last_alert_signature"] = signature


def _render_alert_status(alert_result: dict[str, object]) -> None:
    st.subheader("Estado de alerta")
    cols = st.columns(3)
    cols[0].metric("Regla activa", _rule_label(str(alert_result["rule_key"])))
    cols[1].metric("Severidad", str(alert_result["severity"]).capitalize())
    score_value = alert_result["score"]
    cols[2].metric("Score evaluado", f"{float(score_value):.1f}" if score_value is not None else "s/d")


def _render_alert_rules(storage: StorageManager) -> None:
    st.subheader("Reglas de alerta")
    rules = storage.load_alert_rules()
    if rules.empty:
        st.info("No hay reglas cargadas.")
        return

    severity_labels = {
        "info": "Informativa",
        "warning": "Advertencia",
        "error": "Critica",
    }
    severity_reverse = {label: key for key, label in severity_labels.items()}

    with st.form("alert_rules_form"):
        for row in rules.to_dict(orient="records"):
            rule_key = str(row["rule_key"])
            st.markdown(f"**{rule_key}**")
            enabled = st.checkbox(
                f"Habilitada: {rule_key}",
                value=bool(row["enabled"]),
                key=f"enabled_{rule_key}",
            )
            threshold = st.number_input(
                f"Umbral: {rule_key}",
                value=float(row["threshold"]),
                step=1.0,
                key=f"threshold_{rule_key}",
            )
            window_seconds = st.number_input(
                f"Ventana (seg): {rule_key}",
                value=float(row["window_seconds"]),
                step=1.0,
                min_value=0.0,
                key=f"window_{rule_key}",
            )
            severity = st.selectbox(
                f"Severidad: {rule_key}",
                options=list(severity_labels.values()),
                index=list(severity_labels.keys()).index(str(row["severity"])),
                key=f"severity_{rule_key}",
            )
            st.divider()

        submitted = st.form_submit_button("Guardar reglas")

    if submitted:
        payload = []
        for row in rules.to_dict(orient="records"):
            rule_key = str(row["rule_key"])
            payload.append(
                {
                    "rule_key": rule_key,
                    "enabled": bool(st.session_state[f"enabled_{rule_key}"]),
                    "threshold": float(st.session_state[f"threshold_{rule_key}"]),
                    "window_seconds": float(st.session_state[f"window_{rule_key}"]),
                    "severity": str(severity_reverse[st.session_state[f"severity_{rule_key}"]]),
                }
            )
        try:
            _save_alert_rules(storage, payload)
        except ValueError as exc:
            st.error(str(exc))
            return
        st.success("Reglas de alerta actualizadas.")


def _render_session_notes(storage: StorageManager, session_id: str) -> None:
    st.subheader("Notas de sesion")
    notes = storage.load_session_notes(session_id=session_id) if session_id else pd.DataFrame()
    current = notes.iloc[0].to_dict() if not notes.empty else {}

    with st.form("session_note_form"):
        name = st.text_input(
            "Nombre de la sesion",
            value=str(current.get("name", f"Sesion {session_id}" if session_id else "")),
        )
        description = st.text_area(
            "Descripcion",
            value=str(current.get("description", "")),
            height=100,
        )
        approved = st.checkbox(
            "Aprobada para entrenamiento",
            value=bool(current.get("approved_for_training", False)),
        )
        status_options = ["registrada", "activa", "en_revision", "finalizada"]
        status_labels = {status: _status_label(status) for status in status_options}
        current_status = str(current.get("status", "registrada"))
        status = st.selectbox(
            "Estado",
            options=status_options,
            format_func=lambda option: status_labels[option],
            index=status_options.index(current_status) if current_status in status_options else 0,
        )
        submitted = st.form_submit_button("Guardar nota")

    if submitted:
        if not session_id:
            st.error("No hay una sesion activa o seleccionada para guardar la nota.")
            return
        _save_session_note(
            storage=storage,
            session_id=session_id,
            name=name,
            description=description,
            approved_for_training=approved,
            status=status,
        )
        st.success("Nota de sesion actualizada.")

    if not notes.empty:
        st.dataframe(notes, use_container_width=True, hide_index=True)


def _render_human_labels(storage: StorageManager, session_id: str) -> None:
    st.subheader("Etiquetas humanas")
    labels = storage.load_human_labels(session_id=session_id) if session_id else pd.DataFrame()

    with st.form("human_labels_form"):
        label = st.selectbox(
            "Etiqueta",
            options=["Productivo", "Regular", "Distraido", "Revisar"],
        )
        start_value = st.text_input("Inicio (ISO opcional)", value=datetime.now().isoformat(timespec="seconds"))
        end_value = st.text_input("Fin (ISO opcional)", value="")
        notes = st.text_area("Notas", value="", height=80)
        submitted = st.form_submit_button("Registrar etiqueta")

    if submitted:
        if not session_id:
            st.error("No hay una sesion activa o seleccionada para etiquetar.")
            return
        try:
            parsed_start = _parse_optional_iso_datetime(start_value)
            parsed_end = _parse_optional_iso_datetime(end_value)
        except ValueError:
            st.error("Las fechas deben estar en formato ISO valido, por ejemplo 2026-06-01T10:30:00.")
            return
        _register_human_label(
            storage=storage,
            session_id=session_id,
            label=label,
            start_time=parsed_start,
            end_time=parsed_end,
            notes=notes,
        )
        st.success("Etiqueta humana registrada.")

    if not labels.empty:
        st.dataframe(labels, use_container_width=True, hide_index=True)


def _render_audit_events(storage: StorageManager, session_id: str) -> None:
    st.subheader("Auditoria reciente")
    events = storage.load_audit_events(limit=50, session_id=session_id or None)
    if events.empty:
        if session_id:
            st.info("La sesion seleccionada aun no tiene eventos de auditoria.")
        else:
            st.info("Aun no hay eventos de auditoria.")
        return
    st.dataframe(events, use_container_width=True, hide_index=True)


def _render_evaluation_panel(storage: StorageManager, session_id: str) -> None:
    st.subheader("Evaluacion experimental")
    if not session_id:
        st.info("Selecciona una sesion con etiquetas humanas para evaluar el modelo.")
        return

    history = storage.load_history(limit=None, session_id=session_id)
    human_labels = storage.load_human_labels(session_id=session_id)
    dataset = build_labeled_dataset(history=history, human_labels=human_labels)
    report = evaluate_label_predictions(dataset)

    if int(report["total_samples"]) == 0:
        st.info("Aun no hay muestras evaluables. Registra etiquetas humanas con rango de tiempo.")
        return

    cols = st.columns(3)
    accuracy = report["accuracy"]
    macro_f1 = report["macro_f1"]
    cols[0].metric("Muestras evaluadas", str(report["total_samples"]))
    cols[1].metric("Accuracy", f"{float(accuracy):.3f}" if accuracy is not None else "s/d")
    cols[2].metric("Macro F1", f"{float(macro_f1):.3f}" if macro_f1 is not None else "s/d")

    st.caption("Comparacion entre `productivity_label` predicho y `human_label` registrado manualmente.")

    per_label = report["per_label"]
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "etiqueta": label,
                    "soporte": metrics["support"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "tp": metrics["tp"],
                    "fp": metrics["fp"],
                    "fn": metrics["fn"],
                }
                for label, metrics in per_label.items()
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    confusion_matrix = report["confusion_matrix"]
    if confusion_matrix:
        st.caption("Matriz de confusion")
        confusion_df = pd.DataFrame(confusion_matrix).T
        confusion_df.index.name = "human_label"
        st.dataframe(confusion_df, use_container_width=True)

    st.caption("Dataset evaluado")
    st.dataframe(dataset, use_container_width=True, hide_index=True)


def _handle_monitor_start(config: FocusTrackConfig, camera_index: int) -> None:
    try:
        if st.session_state.monitor is not None:
            st.session_state.monitor.stop()
        monitor = FocusTrackMonitor(config=config, camera_index=camera_index)
        monitor.start()
        st.session_state.monitor = monitor
        st.session_state.monitor_running = True
        st.session_state.active_session_id = monitor.session_id
    except Exception as exc:
        st.error(f"No fue posible iniciar el monitoreo: {exc}")
        st.session_state.monitor = None
        st.session_state.monitor_running = False


def _handle_monitor_stop() -> None:
    if st.session_state.monitor is not None:
        st.session_state.monitor.stop()
        st.session_state.active_session_id = st.session_state.monitor.session_id
    st.session_state.monitor = None
    st.session_state.monitor_running = False


def main() -> None:
    st.set_page_config(page_title="FocusTrack AI", page_icon=":brain:", layout="wide")
    st.title("Sistema Inteligente de Monitoreo de Rendimiento y Distraccion Laboral")
    st.caption(
        "Vision por computadora + reglas persistidas para estimar atencion, fatiga, postura, "
        "distracciones y actividad en PC."
    )

    config, camera_index, refresh_seconds = _build_config()
    storage = StorageManager(config.data_dir)
    _ensure_session_state()
    selected_session_id, sessions = _build_session_selector(storage)

    controls_left, controls_right = st.columns([1, 3])
    with controls_left:
        start_clicked = st.button(
            "Iniciar monitoreo",
            use_container_width=True,
            type="primary",
            disabled=st.session_state.monitor_running,
        )
    with controls_right:
        stop_clicked = st.button(
            "Detener monitoreo",
            use_container_width=True,
            disabled=not st.session_state.monitor_running,
        )

    if start_clicked:
        _handle_monitor_start(config, camera_index)
    if stop_clicked:
        _handle_monitor_stop()

    history = storage.load_history(limit=400, session_id=selected_session_id or None)
    _render_kpis(history)

    rules_map = storage.get_alert_rules_map()
    frame_placeholder = st.empty()
    alert_placeholder = st.empty()

    if st.session_state.monitor_running and st.session_state.monitor is not None:
        try:
            snapshot, frame = st.session_state.monitor.process_next()
            st.session_state.last_frame = frame
            st.session_state.last_snapshot = snapshot
            st.session_state.active_session_id = st.session_state.monitor.session_id
            history = storage.load_history(
                limit=400,
                session_id=st.session_state.monitor.session_id,
            )
        except Exception as exc:
            alert_placeholder.error(f"Error de monitoreo: {exc}")
            _handle_monitor_stop()

    active_session_id = _get_active_session_id() or selected_session_id
    alert_result = _evaluate_alert(st.session_state.last_snapshot, rules_map)
    _register_alert_event_if_needed(storage, active_session_id, alert_result)
    if alert_result["severity"] == "error":
        alert_placeholder.error(str(alert_result["message"]))
    elif alert_result["severity"] == "warning":
        alert_placeholder.warning(str(alert_result["message"]))
    elif alert_result["severity"] == "info":
        alert_placeholder.info(str(alert_result["message"]))
    else:
        alert_placeholder.success(str(alert_result["message"]))

    if st.session_state.last_frame is not None:
        frame_placeholder.image(
            _frame_to_rgb(st.session_state.last_frame),
            caption="Vista analizada",
            use_container_width=True,
        )
    else:
        frame_placeholder.info("Cuando inicies el monitoreo se mostrara aqui el frame anotado en tiempo real.")

    _render_history(history, refresh_seconds, selected_session_id)
    _render_storage_health(storage)
    _render_session_summary(selected_session_id or _get_active_session_id(), sessions)
    _render_session_analytics(storage, selected_session_id or _get_active_session_id())
    _render_alert_status(alert_result)

    session_id = active_session_id
    ops_tab, rules_tab, notes_tab, labels_tab, audit_tab, eval_tab = st.tabs(
        ["Operacion", "Alertas", "Notas", "Etiquetas", "Auditoria", "Evaluacion"]
    )
    with ops_tab:
        st.write(f"Sesion activa: `{session_id or 'sin_sesion'}`")
        st.write("El historial y las alertas se leen desde SQLite; el CSV se mantiene como respaldo.")
        _render_history_export(storage, session_id)
    with rules_tab:
        _render_alert_rules(storage)
    with notes_tab:
        _render_session_notes(storage, session_id)
    with labels_tab:
        _render_human_labels(storage, session_id)
    with audit_tab:
        _render_audit_events(storage, session_id)
    with eval_tab:
        _render_evaluation_panel(storage, session_id)

    if st.session_state.monitor_running:
        time.sleep(refresh_seconds)
        st.rerun()


if __name__ == "__main__":
    main()
