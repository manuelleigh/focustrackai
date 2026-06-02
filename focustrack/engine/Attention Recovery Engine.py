from enum import Enum

class NivelRecuperacion(Enum):
    RAPIDA = "Recuperación rápida"
    NORMAL = "Recuperación normal"
    LENTA = "Recuperación lenta"
    CRITICA = "Recuperación crítica"
  
def calcular_recuperacion_concentracion(tiempos_recuperacion):

    if not tiempos_recuperacion:
        return "Sin datos"

    promedio = sum(tiempos_recuperacion) / len(tiempos_recuperacion)

    if promedio <= 3:
        return "Recuperación rápida"

    elif promedio <= 8:
        return "Recuperación normal"

    elif promedio <= 15:
        return "Recuperación lenta"

    return "Recuperación crítica"

