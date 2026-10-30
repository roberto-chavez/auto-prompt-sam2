### Resultados medidos (E2) — run `20261030T013250Z_2a795a89d5_f9fbd8b0`

_Generado automáticamente desde `report.json`. Solo cifras medidas; la interpretación es del equipo._

| Brazo | IoU agua agrupado | IoU medio por evento | Precisión | Recall | Dice | Mediana IoU por chip |
| --- | --- | --- | --- | --- | --- | --- |
| frozen_decoder | 0.1704 | 0.1878 | 0.1715 | 0.9638 | 0.2912 | 0.0677 |
| finetuned_decoder_best_epoch_1 | 0.3081 | 0.2454 | 0.3988 | 0.5752 | 0.4710 | 0.1795 |

Mejor época (por IoU agrupado en val): **1**; IoU agrupado de la última época: 0.1499.

Criterio prefijado de refutación: The best-epoch pooled validation water IoU is less than or equal to the frozen-decoder pooled validation water IoU on the same chips.
Resultado mecánico del criterio: **not_refuted_on_this_subset** (congelado 0.1704 vs ajustado 0.3081). Interpretación: **PENDIENTE**.

| Época | BCE | Dice | Pérdida | IoU val agrupado | s |
| --- | --- | --- | --- | --- | --- |
| 1 | 1.1087 | 0.7543 | 1.8630 | 0.3081 | 161.0 |
| 2 | 0.2934 | 0.6841 | 0.9775 | 0.1128 | 161.1 |
| 3 | 0.2518 | 0.6833 | 0.9350 | 0.0957 | 161.0 |
| 4 | 0.2471 | 0.6293 | 0.8764 | 0.1499 | 161.2 |

Chips con IoU definido en ambos brazos: 33 (mejoran 28, empeoran 5, igual 0; mediana del delta 0.0626). Descriptivo, una semilla, sin significancia.

Congelamiento verificado: bloques congelados sin cambio = True; decoder cambió = True.
Subconjuntos: train {"chips": 30, "per_event": {"Ghana": 3, "India": 3, "Mekong": 3, "Nigeria": 3, "Pakistan": 3, "Paraguay": 3, "Somalia": 3, "Spain": 3, "Sri-Lanka": 3, "USA": 3}, "components_total": 3842, "components_min": 0, "components_max": 728, "chips_without_components": 4}, val {"chips": 40, "per_event": {"Ghana": 4, "India": 4, "Mekong": 4, "Nigeria": 4, "Pakistan": 4, "Paraguay": 4, "Somalia": 4, "Spain": 4, "Sri-Lanka": 4, "USA": 4}, "components_total": 3893, "components_min": 0, "components_max": 593, "chips_without_components": 7}.

**Interpretación, fallos y limitaciones observadas: PENDIENTE (completar tras revisar curvas y ejemplos).**
