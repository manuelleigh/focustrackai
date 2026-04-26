from __future__ import annotations

import cv2
import mediapipe as mp

from focustrack.config import DetectionThresholds, OptionalModels
from focustrack.models import ObjectMetrics

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class ObjectAnalyzer:
    def __init__(self, thresholds: DetectionThresholds, models: OptionalModels):
        self.thresholds = thresholds
        self.models = models

        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.yolo_model = None
        self.last_boxes: list[tuple[tuple[int, int, int, int], str, float]] = []

        if models.enable_yolo and YOLO is not None:
            try:
                self.yolo_model = YOLO(models.yolo_weights)
            except Exception:
                self.yolo_model = None
    def analyze(
        self,
        frame,
        face_bbox: tuple[int, int, int, int] | None = None,
        frame_number: int = 0,
    ) -> tuple[ObjectMetrics, dict[str, object]]:
