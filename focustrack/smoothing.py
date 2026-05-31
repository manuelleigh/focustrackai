from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field

from focustrack.models import AttentionMetrics, ObjectMetrics, PostureMetrics


@dataclass
class TemporalSmoother:
    window_size: int = 5
    min_absent_votes: int = 3
    min_phone_votes: int = 2
    min_sleepy_votes: int = 3
    attention_states: deque[str] = field(default_factory=deque)
    posture_states: deque[str] = field(default_factory=deque)
    phone_votes: deque[bool] = field(default_factory=deque)
    absent_votes: deque[bool] = field(default_factory=deque)
    sleepy_votes: deque[bool] = field(default_factory=deque)
    score_values: deque[float] = field(default_factory=deque)

    def smooth_attention(self, metrics: AttentionMetrics) -> AttentionMetrics:
        self._append(self.attention_states, metrics.attention_state)
        self._append(self.absent_votes, metrics.attention_state == "ausente")
        self._append(self.sleepy_votes, metrics.attention_state == "somnoliento")

        smoothed = self._copy_attention(metrics)
        if metrics.attention_state == "ausente" and sum(self.absent_votes) < self.min_absent_votes:
            smoothed.attention_state = self._mode_without(self.attention_states, "ausente", default="desviado")
        elif metrics.attention_state == "somnoliento" and sum(self.sleepy_votes) < self.min_sleepy_votes:
            smoothed.attention_state = "atento" if metrics.gaze_direction == "centro" else "desviado"
        else:
            smoothed.attention_state = self._mode(self.attention_states, metrics.attention_state)

        smoothed.confidence = self._state_confidence(self.attention_states, smoothed.attention_state)
        return smoothed

    def smooth_posture(self, metrics: PostureMetrics) -> PostureMetrics:
        self._append(self.posture_states, metrics.posture_state)
        smoothed = PostureMetrics(
            posture_state=self._mode(self.posture_states, metrics.posture_state),
            posture_score=metrics.posture_score,
            shoulder_tilt=metrics.shoulder_tilt,
            torso_lean=metrics.torso_lean,
            head_offset=metrics.head_offset,
            confidence=self._state_confidence(self.posture_states, metrics.posture_state),
        )
        return smoothed

    def smooth_objects(self, metrics: ObjectMetrics) -> ObjectMetrics:
        self._append(self.phone_votes, metrics.phone_detected)
        phone_detected = metrics.phone_detected and sum(self.phone_votes) >= self.min_phone_votes
        object_state = metrics.object_state
        if metrics.object_state == "celular_detectado" and not phone_detected:
            object_state = "mano_en_rostro" if metrics.hand_on_face else "sin_objetos"

        confidence = max(sum(self.phone_votes) / max(1, len(self.phone_votes)), 0.65 if metrics.person_present else 0.35)
        return ObjectMetrics(
            phone_detected=phone_detected,
            hand_on_face=metrics.hand_on_face,
            person_present=metrics.person_present,
            object_state=object_state,
            backend=metrics.backend,
            confidence=round(min(1.0, confidence), 4),
        )

    def smooth_score(self, score: float) -> float:
        self._append(self.score_values, float(score))
        return round(sum(self.score_values) / max(1, len(self.score_values)), 2)

    def _append(self, values: deque, value: object) -> None:
        values.append(value)
        while len(values) > self.window_size:
            values.popleft()

    def _mode(self, values: deque[str], default: str) -> str:
        if not values:
            return default
        return Counter(values).most_common(1)[0][0]

    def _mode_without(self, values: deque[str], excluded: str, default: str) -> str:
        filtered = [value for value in values if value != excluded]
        if not filtered:
            return default
        return Counter(filtered).most_common(1)[0][0]

    def _state_confidence(self, values: deque[str], state: str) -> float:
        if not values:
            return 0.0
        return round(sum(value == state for value in values) / len(values), 4)

    def _copy_attention(self, metrics: AttentionMetrics) -> AttentionMetrics:
        return AttentionMetrics(
            face_detected=metrics.face_detected,
            eyes_detected=metrics.eyes_detected,
            eyes_closed=metrics.eyes_closed,
            attention_state=metrics.attention_state,
            gaze_direction=metrics.gaze_direction,
            left_ear=metrics.left_ear,
            right_ear=metrics.right_ear,
            avg_ear=metrics.avg_ear,
            gaze_ratio=metrics.gaze_ratio,
            fatigue_score=metrics.fatigue_score,
            blink_count=metrics.blink_count,
            backend=metrics.backend,
            confidence=metrics.confidence,
        )
