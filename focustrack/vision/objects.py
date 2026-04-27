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
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands_result = self.hands.process(rgb_frame)
        phone_detected = False
        person_present = face_bbox is not None
        backend = "hands"

        if self.yolo_model is not None and frame_number % max(1, self.models.yolo_frame_stride) == 0:
            backend = "yolo+hands"
            self.last_boxes = self._run_yolo(frame)

        for _, label, _ in self.last_boxes:
            if label == "cell phone":
                phone_detected = True
            if label == "person":
                person_present = True

        hand_on_face = self._hand_on_face(hands_result, face_bbox, frame.shape[1], frame.shape[0])
        if hands_result.multi_hand_landmarks and not person_present:
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
        )
        debug = {
            "hand_landmarks": hands_result.multi_hand_landmarks if hands_result else None,
            "yolo_boxes": self.last_boxes,
        }
        return metrics, debug

    def close(self) -> None:
        self.hands.close()

    def _hand_on_face(self, hands_result, face_bbox: tuple[int, int, int, int] | None, frame_width: int, frame_height: int) -> bool:
        if face_bbox is None or not hands_result or not hands_result.multi_hand_landmarks:
            return False

        x1, y1, x2, y2 = face_bbox
        face_width = max(1, x2 - x1)
        face_height = max(1, y2 - y1)
        expand_x = int(face_width * self.thresholds.hand_face_distance)
        expand_y = int(face_height * self.thresholds.hand_face_distance)
        zone = (x1 - expand_x, y1 - expand_y, x2 + expand_x, y2 + expand_y)

        for hand_landmarks in hands_result.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                point_x = int(landmark.x * frame_width)
                point_y = int(landmark.y * frame_height)
                if zone[0] <= point_x <= zone[2] and zone[1] <= point_y <= zone[3]:
                    return True
        return False

def _run_yolo(self, frame) -> list[tuple[tuple[int, int, int, int], str, float]]:
        if self.yolo_model is None:
            return []

        try:
            result = self.yolo_model.predict(frame, conf=0.35, verbose=False)[0]
        except Exception:
            return self.last_boxes

        boxes: list[tuple[tuple[int, int, int, int], str, float]] = []
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            label = result.names.get(cls_id, str(cls_id))
            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
            boxes.append(((x1, y1, x2, y2), label, confidence))
        return boxes
