# Manual de Usuario

## Inicio

1. Abrir una terminal en la carpeta del proyecto.
2. Ejecutar `streamlit run app.py`.
3. Configurar camara, intervalo de muestreo y opciones en la barra lateral.
4. Presionar `Iniciar monitoreo`.

## Monitoreo

La pantalla principal muestra el frame analizado, el score actual, la clasificacion, el estado de atencion y la aplicacion activa. Los estados principales son:

- `Productivo`: score alto y condiciones favorables.
- `Regular`: atencion o actividad irregular.
- `Distraido`: ausencia, celular, somnolencia o actividad distractora.

## Reportes

En `Reporte de sesion` se puede elegir una sesion, ver el resumen y descargar:

- Excel con resumen, recomendaciones, eventos criticos, uso por aplicacion y detalle.
- HTML como respaldo visual para presentacion.

## Clasificador IA

En `Clasificador IA`, presionar `Entrenar / actualizar modelo IA`. Si hay historico suficiente, se entrena con datos reales; si no, usa datos simulados academicos para demostrar el flujo.

## Validacion

La pestana `Validacion` contiene una matriz base de 20 casos. Durante las pruebas, completar resultado obtenido, correcto y observaciones.
