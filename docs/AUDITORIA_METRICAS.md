# Auditoria, Logs y Metricas

## Almacenamiento robusto

El sistema guarda cada muestra en dos formatos:

- SQLite: `data/focustrack.db`, usado como almacenamiento principal.
- CSV: `data/productivity_history.csv`, mantenido como respaldo simple y exportable.

SQLite usa tablas separadas para:

- `snapshots`: registros de monitoreo con payload completo.
- `audit_events`: eventos relevantes del sistema.

## Auditoria

La bitacora registra eventos como:

- Inicio de monitoreo.
- Fallo al abrir camara.
- Detencion de monitoreo.
- Fallo de captura de frame.
- Distraccion critica.
- Entrenamiento exitoso o fallido del modelo IA.

Desde el dashboard se puede revisar y descargar la auditoria en CSV.

## Logs tecnicos

Los logs se escriben en:

```text
data/logs/focustrack.log
```

El logger usa rotacion de archivos para evitar que el log crezca indefinidamente.

## Metricas reales

El dashboard calcula metricas desde los datos almacenados:

- Sesiones registradas.
- Registros almacenados.
- Score promedio.
- Porcentaje Productivo, Regular y Distraido.
- Eventos auditados.

## Metricas de vision por computadora

Tambien se calculan metricas CV a partir del historial:

- Frames analizados.
- Porcentaje con rostro detectado.
- Porcentaje con persona presente.
- Porcentaje con ojos cerrados.
- Porcentaje con celular detectado.
- Porcentaje con postura encorvada.

## Evidencias visuales

Las evidencias de eventos criticos estan desactivadas por defecto por privacidad. Si se activan en la barra lateral, se guardan frames anotados de distraccion en:

```text
data/evidence/
```

Cada evidencia queda referenciada en la auditoria.
