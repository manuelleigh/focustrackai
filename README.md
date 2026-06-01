# FocusTrack AI

Sistema de monitoreo de atencion, distraccion y productividad laboral usando vision por computadora, reglas persistidas y un dashboard en Streamlit.

## Capacidades actuales

1. Vision y fallbacks
   - `MediaPipe Face Mesh` para atencion visual y `MediaPipe Pose` para postura cuando la dependencia esta disponible.
   - Fallback con `OpenCV Haar Cascades` para rostro y postura cuando `MediaPipe` no esta presente.
   - Soporte opcional para `dlib` y `YOLO`.

2. Monitoreo y scoring
   - Estados de atencion: `atento`, `desviado`, `somnoliento`, `ausente`.
   - Estados de postura: `correcta`, `mejorable`, `encorvada`, `sin_datos`.
   - Score de 0 a 100 con etiquetas `Productivo`, `Regular`, `Distraido`.

3. Persistencia operativa
   - Escritura dual en `SQLite + CSV`.
   - Historial consultado desde SQLite con `payload_json` como respaldo diagnostico.
   - Auditoria de eventos del monitor.
   - Notas de sesion, etiquetas humanas y reglas de alerta persistidas.

4. Dashboard
   - Frame anotado en tiempo real.
   - KPIs, graficas de score e historial por aplicacion.
   - Paneles para salud del storage, auditoria, reglas, notas y etiquetas humanas.
   - Exportacion de historial filtrado por sesion a CSV de analisis.
   - Exportacion de auditoria filtrada por sesion a CSV para revision operativa.

## Instalacion

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

Dependencias opcionales:

```bash
pip install -r requirements-optional.txt
```

Notas:

- `mediapipe` sigue siendo la mejor ruta para analisis fino, pero la app ahora puede cargar con fallback heuristico si falta.
- `dlib` puede requerir Visual Studio Build Tools en Windows.
- `YOLO` usa por defecto `yolov8n.pt`; si no esta disponible, el sistema sigue funcionando sin ese backend.

## Ejecucion

```bash
streamlit run app.py
```

## Pruebas

```bash
C:\laragon\bin\python\python-3.10\python.exe -m unittest discover -s tests -v
```

## Datos generados

- Base SQLite: `data/focustrack.db`
- Historial CSV de respaldo: `data/productivity_history.csv`
- Capturas de pantalla opcionales: `data/screenshots/`

## Estructura relevante

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
    mp_compat.py
    posture.py
    objects.py
app.py
tests/
  test_imports.py
  test_scoring.py
  test_storage.py
```
