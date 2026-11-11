# Estado actual y contexto de la fase 1

**Última actualización:** 3 de noviembre de 2026 (America/Lima).

**Estado general: CONCLUIDA con alcance simplificado en dos partes.**

**Continuación (2026-11-11):** el cierre y las cifras de este documento se conservan. La demo de fase 2 ya tiene implementación y verificaciones locales sin modelos; la integración de pesos, inferencia remota y publicación siguen pendientes. Su estado vigente está en [fase2_demo.md](fase2_demo.md); las menciones a demo pendiente que siguen describen el cierre del 3 de noviembre.

**Decisión:** por instrucción del usuario, se adoptan los resultados del equipo ya incorporados al repositorio y los notebooks `E2_01` y `E2_02` como cierre de la fase 1. No se requieren nuevas ejecuciones para este cierre de seguimiento.

Este documento es el contexto vigente para continuar el proyecto. Sustituye el estado anterior de «4 de 9 etapas concluidas». Las nueve etapas eran el plan inicial, no el criterio de cierre actual; se conservan la [hoja de ruta original](hoja_de_ruta_fase1_original.md) y el [historial](historial_fase1.md). El alcance realizado y su correspondencia con aquel plan figuran en la [hoja de ruta adaptada](hoja_de_ruta_fase1.md).

## Cierre por parte

| Parte | Estado | Entregables y evidencia |
| --- | --- | --- |
| 1. Datos, baseline y teardown | **Concluida** | [E2_01](../../notebooks/E2_01_datos_baseline_teardown.ipynb): entorno Colab, Sen1Floods11, particiones y preprocesamiento, referencia SAM 2.1 tiny y disección instrumentada. Reutiliza módulos y antecedentes de las antiguas etapas 1–4. [Notebook ejecutado](evidence/e2/colab_run_20261030T013250Z/notebooks/E2_01_datos_baseline_teardown_ejecutado.ipynb). |
| 2. Intervención del decoder | **Concluida** | [E2_02](../../notebooks/E2_02_intervencion_decoder.ipynb): hipótesis previa, congelado vs ajustado, entrenamiento solo del decoder, métricas y análisis. [Notebook ejecutado](evidence/e2/colab_run_20261030T013250Z/notebooks/E2_02_intervencion_decoder_ejecutado.ipynb), [report.json](evidence/e2/colab_run_20261030T013250Z/run/report.json), [cuaderno](cuaderno_decisiones.md). |

Las dos partes de la fase 1 son notebooks experimentales. La **fase 2** del proyecto es la demo, pendiente; no debe confundirse con la segunda parte (`E2_02`) ni con el entregable académico **E2**.

## Resultados aceptados

Run `20261030T013250Z_2a795a89d5_f9fbd8b0`, código `f399047464b89a3ed18972f8dc7f1b1364d83a44`, Colab T4, `official`, semilla 42, 30 chips de train / 40 de validación.

| Medición | Resultado |
| --- | --- |
| IoU agua, decoder congelado | 0.17037849358838072 |
| IoU agua, mejor decoder ajustado (época 1) | 0.3080519029370676 |
| IoU agua, última época (4) | 0.14992778300691617 |
| Precisión, congelado → mejor ajustado | 0.171475 → 0.398793 |
| Recall, congelado → mejor ajustado | 0.963819 → 0.575162 |
| Dice, congelado → mejor ajustado | 0.291151 → 0.471009 |
| Parámetros configurados como entrenables | 4 215 109, exclusivamente en el mask decoder |
| Recuperación E2 | Ambos brazos reproducidos sin diferencias de IoU ni de máscaras, en un proceso nuevo de la misma sesión de Colab |

La mejora de la mejor época y la degradación posterior se conservan. La [síntesis de resultados](resultados_fase1.md), la [validación](validacion_e2.md), el [índice de evidencia](evidence/e2/colab_run_20261030T013250Z/README.md) y la [auditoría](auditoria_E2.md) permiten revisar las cifras y sus límites.

## Límites y trabajo excluido del cierre

- Los puntos se simulan desde etiquetas; no hay inferencia automática sin etiquetas ni probe DINOv3 entrenado. DINOv3 se instrumentó en el antecedente de disección, pero no se usó en la intervención E2.
- No se ejecutaron A1–A3 completas ni evaluación final en test, Bolivia o Paraguay reservado. E2 aísla el ajuste del decoder con puntos asistidos idénticos.
- Una semilla y un subconjunto pequeño; la mejor época se elige y se reporta con la misma validación. Los chips sin agua reciben máscara vacía por construcción.
- FLOPs y ERF son parciales por método. `recover` de E2 no acredita reinicio completo de Colab.
- Los checkpoints `best` y `last` están documentados en Drive, no incluidos en Git. Su disponibilidad para la demo deberá verificarse al integrar esa fase.

Estos límites no son tareas activas de fase 1. Retomar el probe automático, las ablaciones completas o la evaluación geográfica supondría ampliar nuevamente el alcance y registrar una decisión posterior.

## Pendientes separados de la fase 1

| Asunto | Situación |
| --- | --- |
| Demo y despliegue (fase 2) | Pendientes. Definir prompts disponibles en la interfaz; el flujo actual usa etiquetas. |
| Informe final, póster actualizado y defensa | Entregas académicas posteriores. El cierre técnico no certifica calificación ni comprensión individual. |
| Licencia de Sen1Floods11 | Consulta enviada; sin respuesta documentada. Mantener el [seguimiento](accesos_y_licencias.md). |
| Generador automático y evaluación geográfica | Fuera del alcance cerrado; posibles ampliaciones. |

## Registro de cierre y reglas para continuar

**2026-11-03:** se reemplaza la planificación activa de nueve etapas por dos partes ya ejecutadas; se cierra la fase 1 por instrucción del usuario y se preserva la evidencia existente. La versión terminada del compañero se toma como base de cierre del equipo. No se modifica código experimental, configuraciones, salidas, predicciones ni métricas. La publicación previa comunicada por el usuario sustituye el antiguo pendiente de subir el bundle; los cambios documentales de esta revisión permanecen en el árbol local hasta su publicación.

Las entradas anteriores permanecen en el [historial](historial_fase1.md) y el [cuaderno](cuaderno_decisiones.md). Sus conteos y órdenes de detenerse describen decisiones de aquellas fechas. Para continuar, usar este estado y la hoja de ruta adaptada; no iniciar automáticamente las antiguas etapas 5–9 ni repetir entrenamiento. Registrar cualquier nuevo alcance por separado y mantener todo entrenamiento y trabajo con GPU en Google Colab.
