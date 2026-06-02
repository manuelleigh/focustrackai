from dataclasses import dataclass

@dataclass
class ProductividadAcademica:
    indice: float
    nivel: str
    observacion: str

PESOS_PRODUCTIVIDAD = {
    "concentracion": 0.40,
    "consistencia": 0.25,
    "participacion": 0.20,
    "eficiencia": 0.15
}

def calcular_productividad(
    concentracion,
    consistencia,
    participacion,
    eficiencia
):

    indice = (
        concentracion * 0.40 +
        consistencia * 0.25 +
        participacion * 0.20 +
        eficiencia * 0.15
    )

    return round(indice, 2)
