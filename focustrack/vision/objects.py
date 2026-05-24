from __future__ import annotations

import cv2

from focustrack.config import DetectionThresholds, OptionalModels
from focustrack.models import ObjectMetrics
from focustrack.vision.mp_compat import HAS_MEDIAPIPE_SOLUTIONS, MP_SOLUTIONS

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - optional dependency
    YOLO = None


class ObjectAnalyzer:
    def __init__(self, thresholds: DetectionThresholds, models: OptionalModels):
        self.thresholds = thresholds
        self.models = models
        self.hands = None
        self.yolo_model = None
        self.last_boxes: list[tuple[tuple[int, int, int, int], str, float]] = []

        if HAS_MEDIAPIPE_SOLUTIONS:
            self.hands = MP_SOLUTIONS.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

        if models.enable_yolo and YOLO is not None:
            try:
                self.yolo_model = YOLO(models.yolo_weights)
            except Exception:
                self.yolo_model = None

    def close(self) -> None:
        if self.hands is not None:
            self.hands.close()