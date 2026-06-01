from __future__ import annotations

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover - optional dependency
    mp = None

HAS_MEDIAPIPE_SOLUTIONS = bool(mp and hasattr(mp, "solutions"))
MP_SOLUTIONS = mp.solutions if HAS_MEDIAPIPE_SOLUTIONS else None
