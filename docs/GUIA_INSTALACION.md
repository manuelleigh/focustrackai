# Guia de Instalacion

## Requisitos

- Windows 10/11 recomendado.
- Python 3.10 o superior.
- Camara web funcional.
- Acceso local a la carpeta del proyecto.

## Instalacion base

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Dependencias opcionales

```bash
pip install -r requirements-optional.txt
```

Para probar `dlib`:

```bash
pip install -r requirements-dlib.txt
```

`dlib` puede requerir Visual Studio Build Tools. Si falla, continuar sin `dlib`; el sistema mantiene deteccion con MediaPipe/OpenCV.

## Ejecucion

```bash
streamlit run app.py
```

## Checklist de demo

- Verificar que la camara no este usada por otra aplicacion.
- Iniciar Streamlit desde la carpeta raiz del proyecto.
- Mantener YOLO desactivado si el equipo esta lento.
- Mantener capturas de pantalla desactivadas salvo que se necesite evidencia visual.
- Ejecutar una sesion corta.
- Revisar historial, reporte, clasificador IA y matriz de validacion.
