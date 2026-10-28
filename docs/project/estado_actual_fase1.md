# Estado actual de la fase 1

**Última actualización:** 28 de octubre de 2026.  
**Estado general:** En curso. Etapa 1 concluida por indicación expresa del usuario, con salvedad de licencia de Sen1Floods11 pendiente.
**Etapas concluidas:** 1 de 9 (etapa 1 cerrada con salvedad documentada).

Este documento registra el avance de las etapas definidas en la [hoja de ruta de la fase 1](hoja_de_ruta_fase1.md). La etapa 1 tiene evidencia local y un notebook ejecutado por el usuario en Colab con `recover` y retorno 0. Se conservó el adjunto original y se actualizó la plantilla local. Los informes de acceso aportados confirman HEAD autenticado autorizado (302) para DINOv3 y SAM, con las revisiones fijadas e identidad coincidente con el run. El ZIP del run permitió verificar hardware/recursos, CUDA e integridad. El usuario dispuso cerrar la etapa 1 tras aportar el correo de consulta enviado; la licencia de Sen1Floods11 continúa sin aclarar. Se reabrirá y revisará el cierre al recibir una respuesta del publicador. Este cierre de seguimiento no acredita autorización de uso. No hay ejecución experimental ni avance de etapas 2–9.

## Estado por etapa

| Etapa | Estado | Progreso realizado | Entregables y validaciones | Próximo paso |
| --- | --- | --- | --- | --- |
| 1. Entorno de Colab, acceso y almacenamiento persistente | **Concluida con salvedad** | Notebook, bootstrap aislado, lock y persistencia; ejecución Colab de recover con retorno 0, pruebas/imports/pines e integridad inferidos del código ejecutado. Bootstrap actualizado para venv sin pip. | [Entregables y validaciones](validacion_etapa1.md), [accesos](accesos_y_licencias.md), [cuaderno](cuaderno_decisiones.md), [evidencias](evidence/stage1/). No hay datos ni pesos descargados. | Consulta de licencia enviada y archivada; revisar/reabrir esta etapa cuando llegue respuesta. Permiso del dataset aún no acreditado. Entorno, recuperación, CUDA y modelos validados. |
| 2. Datos, particiones y preprocesamiento | No iniciada | Ninguno. | Sin datos preparados, manifiestos ni estadísticas; validaciones no ejecutadas. | Etapa 1 cerrada con salvedad; antes de adquirir datos, revisar la base de uso documentada y preparar los regímenes `official` y `leave_paraguay_out`. |
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
| 2026-10-28 | Etapa 1, implementación | Se preparan `notebooks/01_entorno_colab.ipynb`, `requirements-colab.in/.txt`, `configs/environment.yaml`, utilidades runtime/artifacts/access, scripts y pruebas. Semilla 42, identidad por hashes, generaciones completas y recuperación compatible. | Validación local y accesos documentados en [informe](validacion_etapa1.md). El cierre sigue bloqueado: Colab no ejecutado por decisión del usuario, DINOv3 HEAD anónimo 401 y Sen1Floods11 proprietary sin permiso aclarado. 0 de 9 etapas concluidas. No se inicia etapa 2. |
| 2026-10-28 | Revisión de ejecución Colab | Se conserva el notebook del usuario con recover retorno 0, run `20261028T221055Z_1d0f6a92f4_aa07b10e`. Se incorporan diagnósticos visibles y bootstrap para entornos sin ensurepip, más exportador de informes. | [Revisión del adjunto](evidence/stage1/colab_notebook_review.json) y [paso de exportación](validacion_etapa1.md#exportar-los-informes-del-run-ya-ejecutado). Metadatos declaran T4; los valores medidos y HEAD no aparecen en el adjunto. Licencia del dataset pendiente; etapa 1 sigue bloqueada, etapas 2–9 sin iniciar. |
| 2026-10-28 | Verificación de ZIP del run Colab | Se verifican 10 generaciones por hash/tamaño/identidad y 15 archivos de fuente. Tesla T4, Python 3.12.13, CUDA 12.6, 50 pines; recover y semillas correctos, resource_blockers vacío. | [Auditoría](evidence/stage1/colab_run_20261028T221055Z/archive_review.json). Falta carpeta environment/access, situada fuera del run exportado, y aclaración de licencia del dataset. Etapa 1 sigue bloqueada. |
| 2026-10-28 | Auditoría de access.zip | Dos informes de acceso con hashes e identidad coincidentes con el run; HEAD 302 autorizado de SAM 2.1 tiny y DINOv3 SAT-493M en las revisiones fijadas. No se descargaron cuerpos de pesos/datos. | [Acceso verificado](evidence/stage1/colab_run_20261028T221055Z/access.json) y [auditoría](evidence/stage1/colab_run_20261028T221055Z/access_archive_review.json). Único bloqueo restante: licencia de Sen1Floods11. |
| 2026-10-28 | Cierre de etapa 1 por instrucción del usuario | Se revisa el EML de consulta enviado a Floodbase a las 18:05:23 -05:00. El usuario solicita cerrar la etapa 1 y reabrirla al recibir respuesta. Se registra 1 de 9 etapas concluidas con salvedad. | [Consulta enviada](evidence/stage1/licensing_request_sent.md). Validaciones técnicas completas; licencia sin aclarar y sin respuesta/permiso acreditado. No se adquieren datos ni se inicia etapa 2. |
