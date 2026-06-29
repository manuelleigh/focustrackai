from __future__ import annotations

import types
import unittest

from focustrack.config import DetectionThresholds, OptionalModels
from focustrack.vision.attention import AttentionAnalyzer
from focustrack.vision.objects import ObjectAnalyzer
from focustrack.vision.posture import PostureAnalyzer
from focustrack.vision.temporal import TemporalConsensus


class VisionFilterTests(unittest.TestCase):
    def test_temporal_consensus_uses_majority_vote_in_window(self) -> None:
        consensus = TemporalConsensus[str](window_size=4, min_votes=2)
        self.assertEqual(consensus.update("atento"), "atento")
        self.assertEqual(consensus.update("desviado"), "atento")
        self.assertEqual(consensus.update("desviado"), "desviado")
        self.assertEqual(consensus.update("desviado"), "desviado")

    def test_attention_rejects_tiny_face_bbox(self) -> None:
        analyzer = AttentionAnalyzer(DetectionThresholds())
        self.assertFalse(analyzer._valid_face_bbox((10, 10, 30, 30), 960, 540))
        self.assertTrue(analyzer._valid_face_bbox((100, 50, 260, 220), 960, 540))
        analyzer.close()

    def test_attention_requires_consensus_to_mark_face_present(self) -> None:
        thresholds = DetectionThresholds(face_presence_consensus_frames=2)
        analyzer = AttentionAnalyzer(thresholds)
        self.assertFalse(analyzer._stable_face_detected(True))
        self.assertTrue(analyzer._stable_face_detected(True))
        analyzer.close()

    def test_object_binary_stabilization_requires_multiple_frames(self) -> None:
        analyzer = ObjectAnalyzer(DetectionThresholds(object_presence_consensus_frames=2), OptionalModels())
        self.assertFalse(analyzer._stable_binary(True, "phone_frames", 2))
        self.assertTrue(analyzer._stable_binary(True, "phone_frames", 2))
        analyzer.close()

    def test_posture_visibility_filter_rejects_low_visibility_landmarks(self) -> None:
        analyzer = PostureAnalyzer(DetectionThresholds(posture_visibility_min=0.6))
        low_visibility_landmarks = [types.SimpleNamespace(visibility=0.4) for _ in range(5)]
        high_visibility_landmarks = [types.SimpleNamespace(visibility=0.9) for _ in range(5)]
        self.assertFalse(analyzer._has_required_visibility(low_visibility_landmarks))
        self.assertTrue(analyzer._has_required_visibility(high_visibility_landmarks))
        analyzer.close()


if __name__ == "__main__":
    unittest.main()
