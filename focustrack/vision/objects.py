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
            
    def analyze(
        self,
        frame,
        face_bbox: tuple[int, int, int, int] | None = None,
        frame_number: int = 0,
    ) -> tuple[ObjectMetrics, dict[str, object]]:
        
        hands_result = None

        if self.hands is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hands_result = self.hands.process(rgb_frame)

        phone_detected = False
        person_present = face_bbox is not None
        backend = "hands" if self.hands is not None else "heuristico"

        if self.yolo_model is not None and frame_number % max(1, self.models.yolo_frame_stride) == 0:
            backend = "yolo+hands"
            self.last_boxes = self._run_yolo(frame)

        for _, label, _ in self.last_boxes:
            if label == "cell phone":
                phone_detected = True

            if label == "person":
                person_present = True

        hand_landmarks = getattr(hands_result, "multi_hand_landmarks", None)

        hand_on_face = self._hand_on_face(
            hands_result,
            face_bbox,
            frame.shape[1],
            frame.shape[0]
        )

        if hand_landmarks and not person_present:
            person_present = True

        if not person_present:
            object_state = "usuario_ausente"

        elif phone_detected:
            object_state = "celular_detectado"

        elif hand_on_face:
            object_state = "mano_en_rostro"

        else:
            object_state = "sin_objetos"

        metrics = ObjectMetrics(
            phone_detected=phone_detected,
            hand_on_face=hand_on_face,
            person_present=person_present,
            object_state=object_state,
            backend=backend,
            confidence=self._confidence(
                phone_detected,
                hand_on_face,
                person_present,
                backend,
            ),
        )

        debug = {
            "hand_landmarks": hand_landmarks,
            "yolo_boxes": self.last_boxes,
        }

        return metrics, debug
