from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class AttentionMetrics:
    face_detected: bool = False
    eyes_detected: bool = False
    eyes_closed: bool = False
    attention_state: str = "ausente"
    gaze_direction: str = "desconocida"
    left_ear: float | None = None
    right_ear: float | None = None
    avg_ear: float | None = None
    gaze_ratio: float | None = None
    fatigue_score: float = 0.0
    blink_count: int = 0
    backend: str = "mediapipe"


@dataclass
class PostureMetrics:
    posture_state: str = "sin_datos"
    posture_score: float = 50.0
    shoulder_tilt: float | None = None
    torso_lean: float | None = None
    head_offset: float | None = None


@dataclass
class ObjectMetrics:
    phone_detected: bool = False
    hand_on_face: bool = False
    person_present: bool = False
    object_state: str = "sin_objetos"
    backend: str = "heuristico"


@dataclass
class ScreenMetrics:
    active_app: str = "Desconocida"
    window_title: str = ""
    category: str = "neutral"
    productivity_score: float = 60.0
    screenshot_path: str | None = None


@dataclass
class ProductivitySnapshot:
    session_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    attention: AttentionMetrics = field(default_factory=AttentionMetrics)
    posture: PostureMetrics = field(default_factory=PostureMetrics)
    objects: ObjectMetrics = field(default_factory=ObjectMetrics)
    screen: ScreenMetrics = field(default_factory=ScreenMetrics)
    productivity_score: float = 0.0
    productivity_label: str = "Regular"
    attention_component: float = 0.0
    object_component: float = 0.0
    posture_component: float = 0.0
    screen_component: float = 0.0

    def to_row(self) -> dict[str, object]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
            "productivity_score": round(self.productivity_score, 2),
            "productivity_label": self.productivity_label,
            "attention_component": round(self.attention_component, 2),
            "object_component": round(self.object_component, 2),
            "posture_component": round(self.posture_component, 2),
            "screen_component": round(self.screen_component, 2),
            "attention_state": self.attention.attention_state,
            "gaze_direction": self.attention.gaze_direction,
            "avg_ear": round(self.attention.avg_ear, 4) if self.attention.avg_ear is not None else None,
            "fatigue_score": round(self.attention.fatigue_score, 4),
            "blink_count": self.attention.blink_count,
            "face_detected": self.attention.face_detected,
            "eyes_closed": self.attention.eyes_closed,
            "posture_state": self.posture.posture_state,
            "posture_score": round(self.posture.posture_score, 2),
            "shoulder_tilt": round(self.posture.shoulder_tilt, 4) if self.posture.shoulder_tilt is not None else None,
            "torso_lean": round(self.posture.torso_lean, 4) if self.posture.torso_lean is not None else None,
            "head_offset": round(self.posture.head_offset, 4) if self.posture.head_offset is not None else None,
            "phone_detected": self.objects.phone_detected,
            "hand_on_face": self.objects.hand_on_face,
            "person_present": self.objects.person_present,
            "object_state": self.objects.object_state,
            "active_app": self.screen.active_app,
            "window_title": self.screen.window_title,
            "screen_category": self.screen.category,
            "screen_productivity_score": round(self.screen.productivity_score, 2),
            "screenshot_path": self.screen.screenshot_path,
        }

