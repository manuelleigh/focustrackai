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
