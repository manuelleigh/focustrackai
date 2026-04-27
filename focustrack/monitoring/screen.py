from __future__ import annotations

import ctypes
from ctypes import wintypes
from datetime import datetime
import platform

import psutil

from focustrack.config import FocusTrackConfig
from focustrack.models import ScreenMetrics


class ScreenActivityMonitor:
    def __init__(self, config: FocusTrackConfig):
        self.config = config
        self.screenshot_dir = config.screenshot_dir
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

    def sample(self) -> ScreenMetrics:
        app_name = "Desconocida"
        title = ""

        if platform.system().lower() == "windows":
            title, pid = self._get_windows_foreground_window()
            if pid is not None:
                try:
                    app_name = psutil.Process(pid).name()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    app_name = "Desconocida"
        else:
            app_name = self._fallback_active_process()

        category, score = self._classify_activity(app_name, title)
        screenshot_path = self._capture_screenshot() if self.config.screen_capture_enabled else None

        return ScreenMetrics(
            active_app=app_name,
            window_title=title,
            category=category,
            productivity_score=score,
            screenshot_path=screenshot_path,
        )