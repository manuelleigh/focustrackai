from __future__ import annotations

import cv2
import mediapipe as mp

from focustrack.config import DetectionThresholds, OptionalModels
from focustrack.models import ObjectMetrics

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None