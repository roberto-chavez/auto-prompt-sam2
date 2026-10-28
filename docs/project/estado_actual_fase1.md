# Estado actual de la fase 1

**Última actualización:** 28 de octubre de 2026.  
**Estado general:** No iniciada.  
**Etapas concluidas:** 0 de 9.

Este documento registra el avance de las etapas definidas en la [hoja de ruta de la fase 1](hoja_de_ruta_fase1.md). Actualmente no hay progreso de implementación ni ejecución experimental en ninguna etapa. La documentación de planificación y este registro constituyen la preparación documental, sin acreditar el cumplimiento de los criterios técnicos de las etapas.

## Estado por etapa

| Etapa | Estado | Progreso realizado | Entregables y validaciones | Próximo paso |
| --- | --- | --- | --- | --- |
| 1. Entorno de Colab, acceso y almacenamiento persistente | No iniciada | Ninguno. | Sin entregables técnicos; validaciones no ejecutadas. | Preparar el entorno reconstruible, verificar accesos y configurar la persistencia. |
| 2. Datos, particiones y preprocesamiento | No iniciada | Ninguno. | Sin datos preparados, manifiestos ni estadísticas; validaciones no ejecutadas. | Tras cerrar la etapa 1, preparar los datos y aislar los regímenes `official` y `leave_paraguay_out`. |
| 3. Línea base asistida y evaluador común | No iniciada | Ninguno. | Sin baseline, predicciones ni evaluador implementado; validaciones no ejecutadas. | Tras cerrar las etapas 1–2, implementar SAM congelado con puntos simulados y el evaluador común. |
| 4. Disección instrumentada de SAM 2.1 y DINOv3 | No iniciada | Ninguno. | Sin mediciones de parámetros, FLOPs, features, campo receptivo, latencia o memoria. | Tras cerrar las etapas 1–3, instrumentar ambos modelos sobre los datos permitidos. |
| 5. Probe de agua y generación automática de prompts | No iniciada | Ninguno. | Sin features extraídas, probe entrenado ni generador automático; validaciones no ejecutadas. | Tras cerrar las etapas 1–4, entrenar el probe y preparar los prompts por régimen. |
| 6. Entrenamiento exclusivo del mask decoder | No iniciada | Ninguno. | Sin entrenamiento, checkpoints ni curvas; validaciones no ejecutadas. | Tras cerrar las etapas 1–5, fijar la configuración y entrenar únicamente el decoder por régimen. |
| 7. Ablaciones controladas, eficiencia y selección | No iniciada | Ninguno. | Sin ablaciones ejecutadas, selección ni cierres experimentales. | Tras cerrar las etapas 1–6 en ambos regímenes, contrastar A1–A3 y sellar las configuraciones finales. |
| 8. Evaluación final y generalización geográfica | No iniciada | Ninguno. | Sin campañas finales, predicciones ni métricas de conjuntos reservados. | Tras cerrar la etapa 7 en ambos regímenes, evaluar las configuraciones prefijadas según el protocolo. |
| 9. Análisis de resultados, fallos y cierre de artefactos | No iniciada | Ninguno. | Sin análisis experimental ni paquete final; reconstrucción en sesión limpia no comprobada. | Tras cerrar la etapa 8, analizar los resultados y verificar el paquete reproducible de la fase 1. |

## Reglas de actualización

Después de concluir cada etapa, actualizar su fila y este resumen general con la fecha, el progreso real, los entregables y sus rutas o identificadores, las validaciones ejecutadas y su resultado, y los pendientes o bloqueos. Añadir una entrada al historial con el detalle necesario para continuar el trabajo. Cuando corresponda, distinguir el avance de `official` y `leave_paraguay_out`.

Usar los estados **No iniciada**, **En curso**, **Bloqueada** o **Concluida**. Marcar una etapa como concluida únicamente si cumple sus criterios verificables en la hoja de ruta; si el trabajo queda parcial o interrumpido, registrar el avance y mantener el estado que corresponda. No presentar instrucciones pendientes como comprobaciones ejecutadas ni la existencia de un notebook como evidencia suficiente de finalización.

El detalle de las hipótesis, decisiones técnicas y asistencia de IA corresponde al cuaderno de decisiones previsto en la hoja de ruta. Este registro debe enlazar esa evidencia cuando exista y facilitar conocer qué se hizo y qué falta.

## Historial de actualizaciones

| Fecha | Etapa o alcance | Actualización | Evidencia y pendientes |
| --- | --- | --- | --- |
| 2026-10-28 | Inicialización del seguimiento | Se registran las nueve etapas como no iniciadas, con 0 etapas concluidas. | Hoja de ruta disponible; implementación y experimentación aún pendientes. |
