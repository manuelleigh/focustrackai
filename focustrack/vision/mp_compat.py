from __future__ import annotations

from types import SimpleNamespace

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover - optional dependency
    mp = None


def load_mediapipe_solutions():
    if mp is None:
        return None

    solutions = getattr(mp, "solutions", None)
    if solutions is not None:
        return solutions

    try:  # pragma: no cover - only applies to some package layouts
        from mediapipe.python import solutions as legacy_solutions

        return legacy_solutions
    except Exception:
        return None


MP_SOLUTIONS = load_mediapipe_solutions()
HAS_MEDIAPIPE_SOLUTIONS = MP_SOLUTIONS is not None


def drawing_namespace():
    if not HAS_MEDIAPIPE_SOLUTIONS:
        return None

    return SimpleNamespace(
        drawing_utils=MP_SOLUTIONS.drawing_utils,
        face_mesh=MP_SOLUTIONS.face_mesh,
        pose=MP_SOLUTIONS.pose,
        hands=MP_SOLUTIONS.hands,
    )
