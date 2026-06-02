import cv2
import numpy as np


class AnalizadorIluminacion:

    @staticmethod
    def analizar(frame):

        gris = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        brillo = np.mean(gris)

        if brillo < 60:

            nivel = "Deficiente"

        elif brillo < 120:

            nivel = "Aceptable"

        elif brillo < 200:

            nivel = "Buena"

        else:

            nivel = "Excesiva"

        return {
            "brillo_promedio":
                round(float(brillo), 2),
            "nivel":
                nivel
        }
      resultado = (
    AnalizadorIluminacion
    .analizar(frame)
)

print(resultado)
