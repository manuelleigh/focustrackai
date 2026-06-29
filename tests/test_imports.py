from __future__ import annotations

import importlib
import unittest


class ImportTests(unittest.TestCase):
    def test_critical_modules_import(self) -> None:
        modules = [
            "app",
            "focustrack.monitor",
            "focustrack.monitoring.storage",
            "focustrack.vision.attention",
            "focustrack.vision.objects",
            "focustrack.vision.posture",
        ]
        for module_name in modules:
            with self.subTest(module=module_name):
                importlib.import_module(module_name)


if __name__ == "__main__":
    unittest.main()
