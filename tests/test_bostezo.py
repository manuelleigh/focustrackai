from focustrack.features.bostezo import BostezoTracker
from focustrack.models import AttentionMetrics


def _metrics(mar: float, yawning: bool) -> AttentionMetrics:
    return AttentionMetrics(
        face_detected=True,
        eyes_detected=True,
        eyes_closed=False,
        attention_state="atento",
        gaze_direction="centro",
        mouth_aspect_ratio=mar,
        yawning=yawning,
    )


def test_contador_incrementa_una_vez_por_bostezo():
    tracker = BostezoTracker()
    tracker.actualizar(_metrics(0.75, True))
    tracker.actualizar(_metrics(0.72, True))  # mismo bostezo, no debe duplicar
    tracker.actualizar(_metrics(0.20, False))
    assert tracker.contador == 1


def test_dos_bostezos_separados_cuentan_dos():
    tracker = BostezoTracker()
    tracker.actualizar(_metrics(0.75, True))
    tracker.actualizar(_metrics(0.20, False))
    tracker.actualizar(_metrics(0.70, True))
    assert tracker.contador == 2


def test_historial_mar_se_guarda():
    tracker = BostezoTracker()
    tracker.actualizar(_metrics(0.4, False))
    tracker.actualizar(_metrics(0.8, True))
    assert tracker.resumen()["historial_mar"] == [0.4, 0.8]


def test_ignora_frame_sin_deteccion():
    tracker = BostezoTracker()
    tracker.actualizar(_metrics(None, False))  # sin cara detectada
    assert tracker.historial_mar == []