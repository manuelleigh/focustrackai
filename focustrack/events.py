from __future__ import annotations

import pandas as pd


EVENT_DEFINITIONS = {
    "mirada_desviada": ("attention_state", {"desviado"}),
    "somnolencia": ("attention_state", {"somnoliento"}),
    "ausencia": ("attention_state", {"ausente"}),
    "celular": ("phone_detected", {True}),
    "mano_en_rostro": ("hand_on_face", {True}),
    "postura_encorvada": ("posture_state", {"encorvada"}),
    "app_distractora": ("screen_category", {"distraccion"}),
}