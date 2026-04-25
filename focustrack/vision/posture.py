from __future__ import annotations

import cv2
import mediapipe as mp

from focustrack.config import DetectionThresholds
from focustrack.models import PostureMetrics

class PostureAnalyzer:
    def __init__(self, thresholds: DetectionThresholds):
        self.thresholds = thresholds