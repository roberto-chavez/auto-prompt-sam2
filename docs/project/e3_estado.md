# E3 — estado verificado (actualizado 2026-11-18 tras la etapa 8)

## Qué exige el PDF (Tarea_Final_DeepLearning_PREGRADO_2026-2.pdf)
Dos filas E3, fecha 11-nov-2026: «Entrega de trabajo» (2 pts) e «Informe final (4–8 pág., IEEE/LNCS) revisado con la retroalimentación del workshop» (4 pts). El workshop (18–20 nov) es posterior al plazo: su retroalimentación está **pendiente de incorporar**. §8 cita una «plantilla de WAIMLAP» que no existe en el repositorio. La inferencia geográfica de la etapa 8 se ejecutó el 18 de noviembre con el protocolo congelado. Siguen pendientes la prueba interactiva completa de Gradio, la licencia de Sen1Floods11 y la retroalimentación del workshop. El backend manual de la demo sí tiene evidencia de inferencia real.

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

Al 18 de noviembre, la revisión humana del equipo está confirmada y la autorización de etapa 8 quedó registrada en un preflight derivado. El snapshot original conserva `stage8_authorized=false`; el preflight derivado registra autorización y supera la compuerta antes del acceso reservado.

## Fase C según el PDF
§2: «Se evalúa el modelo en un dominio de distribución distinta a la de su preentrenamiento… reportar explícitamente lo que no funcionó». El cambio natural→Sentinel-2 ya está medido (etapas 3–7) y sus fallos reportados. La transferencia geográfica con reservados (Bolivia/Paraguay) es del **protocolo del equipo**, no una exigencia literal del PDF.
**Para consultar con el profesor:** §6 pide informe «IEEE/LNCS» (fila E3), §8 pide «resumen extendido (4–8 pág.) según la plantilla de WAIMLAP»; no hay archivo de plantilla WAIMLAP disponible. El informe actual usa IEEEtran.

## Etapa 8 (transferencia geográfica)
La revisión humana de etapa 7 fue confirmada por el usuario el 2026-11-18; la autorización de continuar quedó registrada en `evidence/e3/human_review_stage7_user_20261118.json`. El registro de auditoría original permanece intacto. El preflight derivado pasa la compuerta y conserva `reserved_access=[]` hasta que comience la ejecución.

El protocolo se corrigió y congeló **antes de abrir datos reservados**: hash canónico `0ccacde9dbf259d1d4403085e331ef34b93fc2381291638d69cd14a0c3cbc681` ([registro](evidence/e3/stage8_protocol_freeze_20261118.json)). Se añadió `a3_frozen` como control de ruta: el efecto del decoder se contrasta `a3_tuned` vs. `a3_frozen` con la misma ruta directa; `a3_frozen` vs. `a1_points` cuantifica la diferencia entre la API predictor y la ruta directa.

La evaluación terminó en Colab T4 el 18 de noviembre. El reporte verificable está en Drive (`stage8/c3ddafe753c41e94f8c69de6ed6947aa9f68ec5b8c5f9086c3ae99909bbe0c5c/stage8_report.json`), con SHA-256 `aa95eee0fb110e4e94d39728d5d86ac1fcba06881fa8ea6c940e2c4e4e771f76`; el resumen reproducible sin identificadores por chip ni máscaras está en [`stage8_results_summary.json`](evidence/e3/stage8_results_summary.json). La identidad del run fija el código `0cd7b39c0564df2c31322eaac06fa4a866f4de37`, el protocolo `0ccacde9dbf259d1d4403085e331ef34b93fc2381291638d69cd14a0c3cbc681` y el par de etapa 7 `996b35878997ebcf5e4662a6e1800056964d73f6dc6ee85225d6302bb00eddcf`.

| Régimen / conjunto | n chips | A1 asistido | A1 puntos automáticos congelado | A3 decoder ajustado |
| --- | ---: | ---: | ---: | ---: |
| official / test_seen | 90 | 0.1823 | 0.2838 | 0.3974 |
| official / Bolivia reservada | 15 | 0.1961 | 0.4994 | 0.5248 |
| leave_paraguay_out / test_seen | 76 | 0.1961 | 0.3111 | 0.3319 |
| leave_paraguay_out / Paraguay reservado | 67 | 0.1796 | 0.1346 | 0.2540 |

Métrica: pooled water IoU; un chip sin agua no define IoU de agua y se excluye de ese agregado. Las filas `a3_frozen` coinciden exactamente con `a1_points` (diferencia de ruta 0). En los reservados, la diferencia automática menos asistida fue +0.3033 en Bolivia (IC bootstrap pareado 95% [0.1785, 0.3821]) y −0.0451 en Paraguay (IC [−0.1015, 0.0043]); el resultado cambia por país. El ajuste del decoder frente a la ruta congelada fue +0.0254 en Bolivia (IC [−0.0472, 0.1430]) y +0.1194 en Paraguay (IC [0.0640, 0.1782]). En el conjunto visto, los cambios del decoder fueron +0.1136 official (IC [−0.0037, 0.2275]) y +0.0209 leave-Paraguay-out (IC [−0.1338, 0.1666]). IC calculados con 2 000 bootstrap pareados por chip, semilla 42; una sola semilla experimental.

Estos datos respaldan **transferencia geográfica medida**, pero no una ventaja universal del prompt automático ni un efecto concluyente del decoder en todos los conjuntos. La elección del checkpoint ajustado procede de validación; los conjuntos reservados tienen 15 y 67 chips. La licencia del dataset sigue pendiente; las máscaras y predicciones permanecen solo en Drive privado y no se publican. La prueba interactiva completa de Gradio y la integración automática en la demo también siguen pendientes.

## Demo
Roberto ejecutó el backend manual en Colab T4 el 11 de noviembre. La auditoría del 18 de noviembre verificó el ZIP y el notebook ejecutado sin errores: Ghana_1033830, máscara binaria de 512 × 512 con 23 252 positivos y hash de chip coincidente. [Evidencia y alcance](evidence/e3/demo_20261111_roberto/README.md). Este run usa SAM congelado y puntos manuales. La interfaz Gradio completa, el decoder ajustado y los puntos automáticos en la demo siguen pendientes.

## Limitaciones
Una semilla; los IC bootstrap son por chips y no equivalen a réplicas independientes; la selección del decoder y del umbral usa la validación; los reservados contienen 15 chips de Bolivia y 67 de Paraguay; Sen1Floods11 sigue sin licencia verificada.
