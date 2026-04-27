from __future__ import annotations

import unittest

from focustrack.config import ProductivityWeights
from focustrack.engine.scoring import evaluate_productivity
from focustrack.models import (
    AttentionMetrics,
    ObjectMetrics,
    PostureMetrics,
    ScreenMetrics,
)


class ScoringTests(unittest.TestCase):
    def test_productive_case_scores_high(self) -> None:
        snapshot = evaluate_productivity(
            session_id="demo",
            attention=AttentionMetrics(
                attention_state="atento", face_detected=True, eyes_detected=True
            ),
            posture=PostureMetrics(posture_state="correcta", posture_score=90.0),
            objects=ObjectMetrics(person_present=True),
            screen=ScreenMetrics(
                active_app="Code", category="trabajo", productivity_score=100.0
            ),
            weights=ProductivityWeights(),
        )
        self.assertGreaterEqual(snapshot.productivity_score, 85.0)
        self.assertEqual(snapshot.productivity_label, "Productivo")

    def test_distracted_case_scores_low(self) -> None:
        snapshot = evaluate_productivity(
            session_id="demo",
            attention=AttentionMetrics(
                attention_state="somnoliento", fatigue_score=1.0
            ),
            posture=PostureMetrics(posture_state="encorvada", posture_score=20.0),
            objects=ObjectMetrics(
                person_present=True, phone_detected=True, hand_on_face=True
            ),
            screen=ScreenMetrics(
                active_app="YouTube", category="distraccion", productivity_score=15.0
            ),
            weights=ProductivityWeights(),
        )
        self.assertLess(snapshot.productivity_score, 40.0)
        self.assertEqual(snapshot.productivity_label, "Distraido")


if __name__ == "__main__":
    unittest.main()
