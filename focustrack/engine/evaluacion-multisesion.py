from dataclasses import dataclass
from datetime import datetime

@dataclass
class ResumenSesion:
    fecha: datetime
    score_promedio: float
    tiempo_concentrado: int
    interrupciones: int
  
  from enum import Enum

class EstadoProgreso(Enum):
    MEJORANDO = "Mejorando"
    ESTABLE = "Estable"
    EN_DESCENSO = "En descenso"
    CRITICO = "Crítico"
  
  VENTANA_ANALISIS = 10

UMBRAL_MEJORA = 10

UMBRAL_DESCENSO = -10

UMBRAL_CRITICO = -20

def calcular_tendencia(sesiones):

    if len(sesiones) < 2:
        return "Datos insuficientes"

    primer_score = sesiones[0].score_promedio
    ultimo_score = sesiones[-1].score_promedio

    diferencia = ultimo_score - primer_score

  def clasificar_progreso(diferencia):

    if diferencia >= 10:
        return "Mejorando"

    if diferencia <= -20:
        return "Crítico"

    if diferencia <= -10:
        return "En descenso"

    return "Estable"

    return diferencia

def calcular_indice_evolucion(sesiones):

    promedio_actual = sesiones[-1].score_promedio

    promedio_historico = (
        sum(s.score_promedio for s in sesiones)
        / len(sesiones)
    )

    return round(
        promedio_actual - promedio_historico,
        2
    )
