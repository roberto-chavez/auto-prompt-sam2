# E3 — estado verificado (2026-11-11, actualizado tras la etapa 7 ejecutada)

## Qué exige el PDF (Tarea_Final_DeepLearning_PREGRADO_2026-2.pdf)
Dos filas E3, fecha 11-nov-2026: «Entrega de trabajo» (2 pts) e «Informe final (4–8 pág., IEEE/LNCS) revisado con la retroalimentación del workshop» (4 pts). El workshop (18–20 nov) es posterior al plazo: su retroalimentación está **pendiente de confirmación** (no existe aún; no se inventó). §8 cita una «plantilla de WAIMLAP» que no existe en el repositorio. Los §§2–3 también exigen evaluar bajo cambio de dominio, analizar fallos y entregar una demo funcional y reproducible. Las etapas 7–9 y la elección de Bolivia o Paraguay como conjuntos reservados forman parte del protocolo del equipo. La evaluación geográfica reservada sigue pendiente. El backend manual de la demo ya tiene evidencia de inferencia real.

## Resultados (solo validación)
Todas las cifras se regeneran con `python scripts/e3_collect_results.py` → [`evidence/e3/consolidated_results.md`](evidence/e3/consolidated_results.md). Fuente de la etapa 7: [`07_ablaciones_y_seleccion_ejecutado.ipynb`](../../notebooks/07_ablaciones_y_seleccion_ejecutado.ipynb) (commit `ebde34e`), métricas extraídas a [`evidence/e3/stage7_metrics_from_notebook.json`](evidence/e3/stage7_metrics_from_notebook.json).

| IoU agua, validación | A1 asistido (3+/3− nominal) | A1 automático (≤3+/3−) | A3 decoder ajustado |
| --- | ---: | ---: | ---: |
| official (89 chips) | 0.1540 | 0.2986 | 0.4841 |
| leave_paraguay_out (75 chips) | 0.1515 | 0.3168 | 0.5058 |

Controles del usuario verificados contra la salida del notebook (coinciden a 4 decimales). La etapa 7 reproduce exactamente las etapas 3 (histórica 1+/1−: 0.158044/0.168434) y 5 (puntos, malla).

- **A1:** automático > asistido igualado (+14.5/+16.5 pp); malla .179/.209; caja completa .098/.100. Asistido con 3+/3− no mejora al histórico 1+/1−.
- **A2:** k+ ∈{1,3,5,10}: .270/.299/.310/.315 (official), .293/.317/.327/.334 (excluido); k− ∈{0,1,3,5}: .283/.307/.299/.313 y .283/.312/.317/.320. Más k− sube precisión y baja recall.
- **A3:** +18.5 / +18.9 pp (reutiliza el mejor checkpoint de la etapa 6 elegido en esta misma validación).
- **Hipótesis H1:** primera parte (puntos < asistida) **refutada en validación**; segunda (puntos > malla) se cumple. H2 parcial. H3 no evaluable sin etapa 8.

## Presupuesto de puntos (no afirmar igualdad de totales)
El presupuesto es nominal por componente. Recuento real verificado el 18 de noviembre a partir de los JSON guardados en Drive, sin modelos ni acceso a reservados.

| Régimen | Asistido positivos / negativos | Automático positivos / negativos |
| --- | ---: | ---: |
| official, 89 chips | 9 879 / 26 322 | 1 204 / 2 055 |
| leave_paraguay_out, 75 chips | 8 311 / 22 047 | 966 / 1 593 |

Los componentes son 8 774 vs. 685 en official y 7 349 vs. 531 en el régimen excluido. La comparación comparte el tope nominal por componente y difiere en los componentes y el esfuerzo real. [Conteos y hashes originales](evidence/e3/e3_point_budget.json).


## Recuperación y auditoría de etapa 7 verificadas el 18 de noviembre

Se ejecutó recover en otro runtime Colab T4 con Python 3.12 aislado y el paquete original de fuentes SHA-256 e9556df886da8bd6585f8c3a29558fdd0c2fea052a2ab96a74e60025dbe94638. Las 13 filas de cada régimen conservaron métricas, selección y locks. La primera imagen de cada fila se reconstruyó sin features en caché con igualdad bit a bit.

La [auditoría final](evidence/e3/revision_stage7_20261118.json) verifica distinto boot_id entre validación y recuperación, cero fallos, 1 157 máscaras official y 975 del régimen excluido auditadas contra etiquetas de validación. Ambos offline_recovery_present son true. Su SHA-256 coincide con el archivo original en Drive, b35b1066f50e29c9b692ed96941c2499676ce41ccb7412b0993780f1685f5d41.

[Notebook de recuperación y auditoría ejecutado](evidence/e3/recovery_20261118.ipynb). Las lecturas paralelas precargaron antecedentes de validación en caché del sistema de archivos; los checks de hash y la reconstrucción sin caché de features permanecieron activos. Los ZIP finales en Drive tienen hashes df51ddc5… (official) y 89c24a9a… (excluido). La auditoría anterior del 11 de noviembre se conserva como antecedente.

La revisión humana del equipo sigue pendiente, stage8_authorized permanece false y reserved_access permanece vacío.

## Fase C según el PDF
§2: «Se evalúa el modelo en un dominio de distribución distinta a la de su preentrenamiento… reportar explícitamente lo que no funcionó». El cambio natural→Sentinel-2 ya está medido (etapas 3–7) y sus fallos reportados. La transferencia geográfica con reservados (Bolivia/Paraguay) es del **protocolo del equipo**, no una exigencia literal del PDF.
**Para consultar con el profesor:** §6 pide informe «IEEE/LNCS» (fila E3), §8 pide «resumen extendido (4–8 pág.) según la plantilla de WAIMLAP»; no hay archivo de plantilla WAIMLAP disponible. El informe actual usa IEEEtran.

## Etapa 8 (transferencia geográfica)
Actualización E3: `src/auto_prompt_sam2/evaluation.py` (compuerta, protocolo congelable, métricas por conjunto/evento, bootstrap pareado, contraste visto vs reservado) y `configs/stage8_protocol.json` (borrador; hash `52d6310c…cff3d`, congelar registrándolo ANTES de abrir). Probado con datos sintéticos (`tests/test_evaluation.py`, 4 pruebas). Falta conectar los predictores de la etapa 7 a los chips reservados (GPU). Compuerta real: CERRADA (`python scripts/run_stage8_local_test.py <revision_stage7_*.json>`).
Condiciones que faltan (`scripts/check_stage8_gate.py`, devuelve «BLOQUEADA»): revisión externa/humana, `stage8_authorized=true`. La recuperación offline ya está verificada. `reserved_access` sigue vacío. `notebooks/08_evaluacion_final_geografica.ipynb` solo aplica la compuerta; el evaluador agregado está implementado y probado con datos sintéticos; falta conectarlo al predictor GPU sobre los reservados.

## Demo
Roberto ejecutó el backend manual en Colab T4 el 11 de noviembre. La auditoría del 18 de noviembre verificó el ZIP y el notebook ejecutado sin errores: Ghana_1033830, máscara binaria de 512 × 512 con 23 252 positivos y hash de chip coincidente. [Evidencia y alcance](evidence/e3/demo_20261111_roberto/README.md). Este run usa SAM congelado y puntos manuales. La interfaz Gradio completa, el decoder ajustado y los puntos automáticos en la demo siguen pendientes.

## Limitaciones
Una semilla, sin IC; selección de época/umbral en la misma validación; sin reservados; Sen1Floods11 sin licencia verificada.
