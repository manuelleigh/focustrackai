import cv2


class DetectorMultiplesPersonas:

    def __init__(self):

        self.face_cascade = (
            cv2.CascadeClassifier(
                cv2.data.haarcascades +
                "haarcascade_frontalface_default.xml"
            )
        )

    def detectar(self, frame):

        gris = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        rostros = self.face_cascade.detectMultiScale(
            gris,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        cantidad = len(rostros)

        return {
            "cantidad_rostros": cantidad,
            "multiples_personas":
                cantidad > 1
        }
      detector = (
        
    DetectorMultiplesPersonas()
)

resultado = detector.detectar(frame)

print(resultado)
