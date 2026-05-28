from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass
class ProductivityWeights:
    attention: float = 0.40
    phone: float = 0.20
    posture: float = 0.15
    screen: float = 0.25


@dataclass
class DetectionThresholds:
    ear_closed: float = 0.20
    gaze_center_min: float = 0.35
    gaze_center_max: float = 0.65
    fatigue_frame_window: int = 8
    shoulder_tilt_max: float = 0.08
    torso_lean_max: float = 0.12
    head_offset_max: float = 0.14
    hand_face_distance: float = 0.12


@dataclass
class OptionalModels:
    enable_dlib: bool = False
    dlib_shape_predictor: Path | None = None
    enable_yolo: bool = False
    yolo_weights: str = "yolov8n.pt"
    yolo_frame_stride: int = 5


@dataclass
class FocusTrackConfig:
    weights: ProductivityWeights = field(default_factory=ProductivityWeights)
    thresholds: DetectionThresholds = field(default_factory=DetectionThresholds)
    models: OptionalModels = field(default_factory=OptionalModels)
    productive_keywords: tuple[str, ...] = (
        "excel",
        "word",
        "powerpoint",
        "power bi",
        "outlook",
        "teams",
        "slack",
        "notion",
        "trello",
        "jira",
        "github",
        "gitlab",
        "vscode",
        "visual studio",
        "pycharm",
        "terminal",
        "cmd",
        "powershell",
        "python",
        "erp",
        "crm",
        "sap",
        "zendesk",
        "salesforce",
        "call center",
    )
    distracting_keywords: tuple[str, ...] = (
        "youtube",
        "facebook",
        "instagram",
        "tiktok",
        "netflix",
        "twitch",
        "x.com",
        "twitter",
        "discord",
        "spotify",
        "whatsapp",
        "telegram",
        "steam",
        "game",
        "shorts",
        "reels",
    )
    data_dir: Path = field(default_factory=lambda: BASE_DIR / "data")
    screenshot_dir: Path = field(
        default_factory=lambda: BASE_DIR / "data" / "screenshots"
    )
    camera_width: int = 960
    camera_height: int = 540
    screen_capture_enabled: bool = False
