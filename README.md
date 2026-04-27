# FocusTrack AI

Prototipo de sistema inteligente para monitorear atencion, distraccion y productividad laboral usando vision por computadora, reglas de IA y un dashboard en Streamlit.

## Modulos implementados

1. Deteccion de rostro y ojos
   - `OpenCV + MediaPipe Face Mesh` para detectar rostro, ojos y landmarks faciales.
   - Calculo de `EAR (Eye Aspect Ratio)` para estimar fatiga o somnolencia.
   - Soporte opcional para `dlib` como respaldo de deteccion facial.

2. Analisis de atencion
   - Estimacion basica de mirada con la posicion relativa del iris.
   - Estados disponibles: `atento`, `desviado`, `somnoliento`, `ausente`.

3. Deteccion de distracciones
   - Integracion opcional con `YOLO` para detectar `cell phone` y `person`.
   - Deteccion heuristica de mano en el rostro con `MediaPipe Hands`.
   - Identificacion de ausencia de usuario.

4. Postura corporal
   - `MediaPipe Pose` para evaluar inclinacion de hombros, torso y cabeza.
   - Clasificacion de postura: `correcta`, `mejorable`, `encorvada`.

5. Analisis de pantalla
   - Lectura de aplicacion/ventana activa.
   - Clasificacion simple entre `trabajo`, `neutral` y `distraccion`.
   - Captura opcional de pantalla para auditoria visual.

6. Sistema de puntuacion
   - Score de 0 a 100 basado en atencion, objetos, postura y actividad en PC.
   - Etiquetas: `Productivo`, `Regular`, `Distraido`.

7. Dashboard
   - Visualizacion en tiempo real del frame analizado.
   - KPIs, graficas de score, tiempo estimado por aplicacion e historial de eventos.

## Estructura

```text
focustrack/
  config.py
  models.py
  monitor.py
  engine/scoring.py
  monitoring/
    screen.py
    storage.py
  vision/
    attention.py
    posture.py
    objects.py
app.py
tests/test_scoring.py
```

## Instalacion

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

Opcional para funcionalidades avanzadas:

```bash
pip install -r requirements-optional.txt
```

Notas:

- `dlib` puede requerir Visual Studio Build Tools en Windows.
- `YOLO` usa por defecto `yolov8n.pt`; si no esta disponible, el sistema sigue funcionando con heuristicas.

## Ejecucion

```bash
streamlit run app.py
```

## Datos generados

- Historial CSV: `data/productivity_history.csv`
- Capturas de pantalla opcionales: `data/screenshots/`

## Ideas para seguir subiendo la nota

- Entrenar un clasificador supervisado con historicos reales.
- Agregar alertas sonoras y notificaciones.
- Soportar multiples empleados con identificacion por usuario o camara.
- Crear reportes diarios y semanales por persona.
- Incorporar OCR o clasificacion visual del contenido de pantalla.
