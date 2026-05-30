# Funcionalidades Avanzadas

FocusTrack AI incorpora un panel operativo modular. La aplicacion ya no se limita al analisis visual en tiempo real; organiza el flujo completo de monitoreo, gestion, IA, eventos, alertas, reportes y diagnostico.

## Modulos funcionales

### Monitoreo

- Analisis visual en tiempo real.
- Score explicable por componentes.
- Historial reciente.
- Eventos consolidados por duracion.

### Sesiones

- Vista consolidada de sesiones.
- Nombre y descripcion por sesion.
- Estado de sesion: registrada, validada o descartada.
- Marcado de sesiones aptas para entrenamiento.

### Eventos

- Centro de eventos con inicio, fin y duracion.
- Filtros por tipo de evento.
- Metricas de duracion total y evento mas prolongado.

### Alertas

- Reglas configurables desde interfaz.
- Umbrales para score bajo, celular, ausencia, somnolencia y apps distractoras.
- Severidad por regla.
- Evaluacion sobre ventanas de tiempo.

### IA

- Entrenamiento por ventanas temporales.
- Comparacion regla vs IA.
- Confianza de prediccion.
- Deteccion de ventanas anomalas.
- Importancia de variables.
- Matriz de confusion y metricas formales.

### Dataset

- Etiquetado humano de sesiones.
- Calibracion de usuario.
- Simulador de sesiones sinteticas por escenario.

### Reportes

- Reporte individual Excel/HTML.
- Comparacion entre sesiones.
- Recomendaciones avanzadas basadas en patrones, eventos y alertas.

### Sistema

- Auditoria.
- Estado de SQLite, logs, capturas, evidencias y modelo IA.
- Ultimos eventos tecnicos.

## Escenarios sinteticos

El simulador puede generar sesiones de:

- Productividad estable.
- Sesion mixta.
- Fatiga.
- Celular recurrente.
- Ausencia prolongada.
- Aplicaciones distractoras.

Estos escenarios permiten probar reportes, IA, eventos y alertas sin depender siempre de la camara.
