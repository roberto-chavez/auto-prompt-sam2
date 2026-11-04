# Resultados y cierre de la fase 1

**Cierre:** 3 de noviembre de 2026 (America/Lima), por instrucción del usuario. Se aceptan los resultados existentes de las dos partes `E2_01` y `E2_02`, sin reentrenamiento ni sustitución de evidencia. Véanse el [estado vigente](estado_actual_fase1.md) y la [adaptación del plan original](hoja_de_ruta_fase1.md).

## Configuración y procedencia

| Elemento | Valor conservado |
| --- | --- |
| Run | `20261030T013250Z_2a795a89d5_f9fbd8b0` |
| Código experimental | `f399047464b89a3ed18972f8dc7f1b1364d83a44` |
| Entorno | Google Colab, Tesla T4, Python 3.12.13, torch 2.7.1, CUDA 12.6, float32 |
| Modelo | SAM 2.1 Hiera tiny, revisión `de431c4043854a71d8101e17995dfe596bf101a5` |
| Régimen y semilla | `official`, semilla 42 |
| Subconjuntos | 30 chips de entrenamiento y 40 de validación; 3/4 por evento, seleccionados por ID |
| Prompts | Simulados desde etiquetas: 1 positivo y 1 negativo por componente, política de la referencia asistida |
| Intervención | Solo mask decoder, 4 215 109 parámetros configurados como entrenables de 38 962 498 totales |
| Optimización | BCE+Dice en píxeles válidos, AdamW, lr 5e-5, acumulación de 4 chips, máximo 5 épocas, paciencia 3 |

Fuentes originales: [report.json](evidence/e2/colab_run_20261030T013250Z/run/report.json), [configuración](../../configs/e2_intervention.yaml), [notebooks ejecutados e índice](evidence/e2/colab_run_20261030T013250Z/README.md). La hipótesis, registrada antes de entrenar, fue que ajustar solo el decoder aumentaría la IoU de agua frente al congelado con los mismos puntos, chips y evaluador.

## Comparación medida

| Brazo en 40 chips de validación | IoU agua agrupado | Precisión | Recall | Dice |
| --- | ---: | ---: | ---: | ---: |
| Congelado (época 0) | 0.17037849358838072 | 0.171475 | 0.963819 | 0.291151 |
| Ajustado, mejor época (1) | 0.3080519029370676 | 0.398793 | 0.575162 | 0.471009 |
| Ajustado, última época (4) | 0.14992778300691617 | — | — | — |

La diferencia mejor ajustado − congelado es **+0.13767340934868688**, aproximadamente 13.77 puntos porcentuales. De los 33 chips con IoU individual definida en ambos brazos, 28 mejoran y 5 empeoran; los 7 chips sin agua reciben predicción vacía por construcción. El criterio previo de refutación no se cumplió en este subconjunto, sin que eso demuestre generalización.

| Época ajustada | Pérdida de entrenamiento | IoU agua de validación |
| --- | ---: | ---: |
| 1 | 1.862985 | 0.308052 |
| 2 | 0.977451 | 0.112774 |
| 3 | 0.935046 | 0.095690 |
| 4 | 0.876392 | 0.149928 |

Fuente: [history.csv](evidence/e2/colab_run_20261030T013250Z/run/history.csv). El entrenamiento terminó por paciencia 3 tras la época 4; se conserva tanto la mejor época como la última. La caída de la pérdida de entrenamiento junto con el deterioro de validación es compatible con sobreajuste o desalineación de la selección por IoU predicho, pero no se hizo una ablación que identifique la causa.

El ajuste reduce las falsas detecciones y pierde agua real: el área agregada de falsos positivos pasa de 401.1 a 72.6 millones de m², mientras el recall baja de 0.9638 a 0.5752. La mejora de IoU no significa que se detecte más agua. La [interpretación del cuaderno](cuaderno_decisiones.md) conserva ejemplos que mejoran y empeoran y las hipótesis alternativas.

## Línea base y disección de la parte 1

La validación completa del antecedente de baseline obtuvo IoU de agua **0.158044** en 89 chips `official` y **0.168434** en 75 chips `leave_paraguay_out`. Esas cifras no deben usarse como brazo congelado de la comparación de 40 chips. [Validación de baseline](validacion_etapa3.md).

La disección de SAM registró 38 962 498 parámetros y **296.05 GFLOPs parciales**. Las salidas de `E2_01` muestran 152.6/154.9 ms de latencia (`official`/excluido) y pico de memoria asignada de 684 419 584 B, batch 1, float32, T4, 5 calentamientos y 20 repeticiones. Las features y el ERF describen sensibilidad local de dos chips; no prueban semántica de agua ni generalización. [Teardown y cobertura](teardown_sam2.md).

La disección anterior de DINOv3 se conserva en [validacion_etapa4.md](validacion_etapa4.md). No se usó DINOv3 en la intervención del decoder ni se entrenó un probe para generar puntos.

## Artefactos conservados

| Artefacto | Ubicación y alcance |
| --- | --- |
| Notebooks ejecutados, métricas, curvas y ejemplos | [Índice del run](evidence/e2/colab_run_20261030T013250Z/README.md) |
| Predicciones, puntos y métricas por chip | [predictions_val.npz](evidence/e2/colab_run_20261030T013250Z/run/predictions_val.npz), [points_val.json](evidence/e2/colab_run_20261030T013250Z/run/points_val.json), [per_chip_val.csv](evidence/e2/colab_run_20261030T013250Z/run/per_chip_val.csv) |
| Checkpoints `best` y `last` | Drive: `e2/runs/20261030T013250Z_2a795a89d5_f9fbd8b0/checkpoints/`; generaciones y SHA-256 en `report.json` e índice del run, sin pesos en Git |
| Auditoría y recálculo previo | [auditoria_E2.md](auditoria_E2.md), [audit_recomputation.json](evidence/e2/audit_recomputation.json) |
| Recuperación | [Salida literal](evidence/e2/colab_run_20261030T013250Z/derived_from_notebooks/recover_output_literal.txt): diferencia de IoU 0.0, confusiones idénticas y 0 píxeles de desacuerdo en ambos brazos |

`recover` corrió en un proceso nuevo de la misma sesión de Colab, sin evidencia de reinicio completo del runtime. Las salidas ejecutadas y los archivos del run permanecen intactos; los textos «NO EJECUTADO» o «PENDIENTE» que aún contienen pertenecen a las plantillas históricas originales. El estado actual se registra en la documentación externa a esos artefactos.

## Alcance de las conclusiones

La fase 1 queda concluida como experimento asistido de disección e intervención del decoder. Una semilla, validación pequeña y selección/reporte en la misma validación limitan la inferencia. No se ejecutó evaluación final en test, Bolivia o Paraguay reservado; no hay flujo automático DINOv3→SAM ni matriz A1–A3 completa. Los resultados no cuantifican todavía el costo de reemplazar puntos asistidos por automáticos.

La consulta de licencia del dataset sigue pendiente según [accesos_y_licencias.md](accesos_y_licencias.md). La calificación, defensa oral, informe, póster actualizado y demo son asuntos separados del cierre de seguimiento. La demo deberá definir una fuente de prompts compatible con los componentes disponibles.
