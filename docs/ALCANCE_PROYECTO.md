# Alcance del Proyecto

## Objetivo

FocusTrack AI es una aplicacion local para monitoreo individual en un entorno laboral simulado. Su proposito academico es demostrar integracion de vision por computadora, analisis de actividad en PC, reglas de productividad, reportes e IA supervisada.

## Alcance funcional

- Detectar atencion visual, mirada, ojos cerrados y ausencia del usuario.
- Estimar postura corporal.
- Detectar celular u objetos distractores cuando YOLO esta disponible.
- Clasificar la aplicacion activa como trabajo, neutral o distraccion.
- Calcular un score de productividad explicable de 0 a 100.
- Guardar historial local por sesion.
- Exportar reportes Excel y HTML.
- Entrenar un clasificador `RandomForestClassifier` con historico real o datos simulados academicos.
- Mostrar prediccion IA, confianza y coincidencia contra la regla.

## Fuera de alcance

- Monitoreo empresarial multiusuario.
- Sanciones automaticas a trabajadores.
- Reconocimiento de identidad facial.
- Almacenamiento en nube o servidor externo.
- Validacion estadistica formal para decisiones laborales reales.

## Criterios de exito

- La app inicia con `streamlit run app.py`.
- El monitoreo se puede iniciar y detener sin errores.
- Se genera historial en `data/productivity_history.csv`.
- Se puede descargar reporte Excel o HTML.
- Se puede entrenar el modelo IA desde el dashboard.
- Existe matriz de validacion con al menos 20 casos.
- El equipo puede explicar reglas, IA, privacidad, limitaciones y mejoras futuras.

## Riesgos

- Camara no detectada o bloqueada por permisos del sistema.
- MediaPipe puede tener limitaciones en Windows segun version instalada.
- YOLO puede ser pesado para equipos de baja potencia.
- `dlib` es opcional porque puede fallar al compilar en Windows.
- Los datos reales pueden ser insuficientes para una IA generalizable.
