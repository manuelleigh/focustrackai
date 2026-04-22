from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from focustrack.config import DetectionThresholds
from focustrack.models import AttentionMetrics

try:
    import dlib
except ImportError:  # pragma: no cover - optional dependency
    dlib = None


LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
FACE_BBOX_REFERENCE = [10, 152, 234, 454]
