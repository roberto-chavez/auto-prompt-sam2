# E3 — estado verificado (2026-11-11, actualizado tras la etapa 7 ejecutada)

## Qué exige el PDF (Tarea_Final_DeepLearning_PREGRADO_2026-2.pdf)
Dos filas E3, fecha 11-nov-2026: «Entrega de trabajo» (2 pts) e «Informe final (4–8 pág., IEEE/LNCS) revisado con la retroalimentación del workshop» (4 pts). El workshop (18–20 nov) es posterior al plazo: su retroalimentación está **pendiente de confirmación** (no existe aún; no se inventó). §8 cita una «plantilla de WAIMLAP» que no existe en el repositorio. Lo que pide el curso: informe y entrega. Plan del equipo (no exigido por el PDF): etapas 7–9 (A1–A3, transferencia geográfica en reservados).

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
El presupuesto es nominal **por componente**, y los componentes difieren (etiquetas vs sonda). Total real conocido solo para el automático de la etapa 5 (1 204+/2 055− official; 966+/1 593− excluido). Los conteos reales de A1 están en los `result_*.json` de la evidencia (Drive); `python scripts/e3_point_budget.py DIR` los totaliza. **Pendiente**.

## Auditoría de la etapa 7 (Drive: `MyDrive/auto-prompt-sam2/stage7/revision_stage7_1794435476494193372.json`)
Terminó. Ambos regímenes: `portable_consistency_passed_not_stage_closure`, 13 filas, `failures: []`, 1 157 (official) y 975 (excluido) máscaras de validación auditadas, `models_executed:false`, `reserved_access:[]`; ZIP verificados `ed582c1f…` (official, 1 230 generaciones) y `06f6dc83…` (excluido, 1 020). Pendientes declarados: revisión humana y **recover offline en otro runtime de Colab** (`offline_recovery_present:false`). Copia de los campos leídos: `evidence/e3/revision_stage7_*.json` (transcripción). Hallazgo: la salida del notebook muestra un intento previo fallido (`ValueError: Prior implementation changed: configs/baseline.yaml`) y dos hashes de evidence.zip por régimen; el segundo coincide con la auditoría.

## Etapa 8 (transferencia geográfica)
Condiciones que faltan (`scripts/check_stage8_gate.py`, devuelve «BLOQUEADA»): revisión externa/humana, `stage8_authorized=true` y recover offline en otro runtime. `reserved_access` sigue vacío. `notebooks/08_evaluacion_final_geografica.ipynb` solo aplica la compuerta; el evaluador de la etapa 8 **no está implementado** (se escribe tras la autorización).

## Demo
Verificada solo en pruebas locales de interfaz (17 pruebas). Máscara con pesos reales **no verificada** (sin GPU ni Hugging Face aquí). Procedimiento: `notebooks/E3_demo_verificacion_colab.ipynb` (preparado, no ejecutado). Solo puntos manuales; el flujo automático **no está integrado**.

## Limitaciones
Una semilla, sin IC; selección de época/umbral en la misma validación; sin reservados; Sen1Floods11 sin licencia verificada.
