import logging
from plyer import notification
import time

class OSNotifier:
    _last_alert_time = 0
    _cooldown_seconds = 10

    @classmethod
    def send_notification(cls, title: str, message: str, severity: str = "warning"):
        current_time = time.time()
        if current_time - cls._last_alert_time < cls._cooldown_seconds:
            return

        cls._last_alert_time = current_time

        try:
            notification.notify(
                title=title,
                message=message,
                app_name="FocusTrack AI",
                timeout=5,
            )
        except Exception as e:
            logging.warning(f"No se pudo enviar notificacion nativa: {e}")
