from dataclasses import dataclass
from datetime import datetime

@dataclass
class RegistroMetrica:
    timestamp: datetime
    score: float
    atencion: float
    postura: float
def registrar_metrica(historial, score, atencion, postura):

    historial.append(
        RegistroMetrica(
            timestamp=datetime.now(),
            score=score,
            atencion=atencion,
            postura=postura
        )
    )

    return historial
  
  def obtener_promedio_historico(historial):

    if not historial:
        return 0

    return sum(r.score for r in historial) / len(historial)
