from enum import Enum

class ContextoEstudio(Enum):
    LECTURA = "Lectura"
    PRACTICA = "Práctica"
    EXAMEN = "Examen"
    INVESTIGACION = "Investigación"

PESOS_CONTEXTO = {
    "Lectura": {
        "atencion": 0.4,
        "postura": 0.3,
        "consistencia": 0.3
    },
    "Práctica": {
        "atencion": 0.6,
        "postura": 0.2,
        "consistencia": 0.2
    },
    "Examen": {
        "atencion": 0.7,
        "postura": 0.15,
        "consistencia": 0.15
    }
}

def calcular_score_contextual(
    contexto,
    atencion,
    postura,
    consistencia
):

    pesos = PESOS_CONTEXTO[contexto]

    score = (
        atencion * pesos["atencion"] +
        postura * pesos["postura"] +
        consistencia * pesos["consistencia"]
    )

    return round(score, 2)
