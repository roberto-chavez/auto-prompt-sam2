# E3 — estado verificado (2026-11-11)

## Qué exige el PDF (Tarea_Final_DeepLearning_PREGRADO_2026-2.pdf)
Dos filas E3 con fecha 11-nov-2026: «Entrega de trabajo» (2 pts) e «Entrega de informe final (4–8 pág., IEEE/LNCS) revisado con la retroalimentación del workshop» (4 pts). El workshop (18–20 nov) es posterior al plazo: la revisión con sus comentarios queda **pendiente de confirmación**. §8 menciona una «plantilla de WAIMLAP» que no existe en el repositorio.

## Qué se ejecutó de verdad (todo en validación, nada en conjuntos reservados)
Cifras consolidadas por `scripts/e3_collect_results.py` desde las salidas originales: [`evidence/e3/consolidated_results.md`](evidence/e3/consolidated_results.md).

| Régimen | Puntos desde etiquetas 1+/1− (etapa 3) | Puntos DINOv3 ≤3+/3− (etapa 5) | Decoder entrenado (etapa 6) |
| --- | ---: | ---: | ---: |
| official (89 chips val) | 0.1580 | 0.2986 | 0.4841 |
| leave_paraguay_out (75 chips val) | 0.1684 | 0.3168 | 0.5058 |

## Limitaciones de interpretación (observadas)
1. Etapa 3 vs 5 **no está pareada**: presupuesto distinto (1+/1− vs hasta 3+/3− por componente) y origen de componentes distinto. Es descriptivo, no el contraste controlado.
2. Etapa 5 → 6 cambia el decoder; no se atribuye al tipo de prompt.
3. El mejor umbral/época se eligió sobre la misma validación; no es prueba de generalización.
4. La comparación pareada (etapa 7, A1 `a1_assisted` 3+/3−) está preparada y **no ejecutada**; requiere GPU/Colab.
5. Transferencia geográfica (Bolivia/Paraguay reservados): **no evaluada**.
