import time


class DetectorAusencia:

    def __init__(self, umbral_segundos=10):
        self.umbral_segundos = umbral_segundos
        self.inicio_ausencia = None

    def actualizar(self, rostro_detectado):

        if rostro_detectado:

            self.inicio_ausencia = None

            return {
                "presente": True,
                "tiempo_ausencia": 0
            }

        if self.inicio_ausencia is None:
            self.inicio_ausencia = time.time()

        tiempo_ausencia = (
            time.time() - self.inicio_ausencia
        )

        return {
            "presente": False,
            "tiempo_ausencia": round(
                tiempo_ausencia,
                2
            ),
            "ausencia_prolongada":
                tiempo_ausencia >= self.umbral_segundos
        }

  detector = DetectorAusencia()

resultado = detector.actualizar(
    rostro_detectado=False
)

print(resultado)
