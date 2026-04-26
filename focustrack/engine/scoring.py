from __future__ import annotations

from focustrack.config import ProductivityWeights
from focustrack.models import AttentionMetrics, ObjectMetrics, PostureMetrics, ProductivitySnapshot, ScreenMetrics


def _attention_component(metrics: AttentionMetrics) -> float:
    state_to_score = {
        "atento": 100.0,
        "desviado": 35.0,
        "somnoliento": 15.0,
        "ausente": 0.0,
    }
    base_score = state_to_score.get(metrics.attention_state, 50.0)
    fatigue_penalty = metrics.fatigue_score * 20.0
    return max(0.0, min(100.0, base_score - fatigue_penalty))


def _object_component(metrics: ObjectMetrics) -> float:
    if not metrics.person_present:
        return 0.0

    score = 100.0
    if metrics.phone_detected:
        score -= 70.0
    if metrics.hand_on_face:
        score -= 25.0

    return max(0.0, min(100.0, score))


def _posture_component(metrics: PostureMetrics) -> float:
    return max(0.0, min(100.0, metrics.posture_score))


def _screen_component(metrics: ScreenMetrics) -> float:
    return max(0.0, min(100.0, metrics.productivity_score))


def _label_for_score(score: float) -> str:
    if score >= 75.0:
        return "Productivo"
    if score >= 45.0:
        return "Regular"
    return "Distraido"


def evaluate_productivity(
    session_id: str,
    attention: AttentionMetrics,
    posture: PostureMetrics,
    objects: ObjectMetrics,
    screen: ScreenMetrics,
    weights: ProductivityWeights,
) -> ProductivitySnapshot:
    attention_component = _attention_component(attention)
    object_component = _object_component(objects)
    posture_component = _posture_component(posture)
    screen_component = _screen_component(screen)

    score = (
        attention_component * weights.attention
        + object_component * weights.phone
        + posture_component * weights.posture
        + screen_component * weights.screen
    )

    return ProductivitySnapshot(
        session_id=session_id,
        attention=attention,
        posture=posture,
        objects=objects,
        screen=screen,
        productivity_score=round(score, 2),
        productivity_label=_label_for_score(score),
        attention_component=attention_component,
        object_component=object_component,
        posture_component=posture_component,
        screen_component=screen_component,
    )
