# E2 — Auditoría de requisitos, evidencia existente y validación

**Estado: preparada, NO ejecutada por el usuario en GPU. E2 no está acreditada.** Última actualización: 29 de octubre de 2026 (America/Lima). Ejecución: [PASOS_E2.md](PASOS_E2.md).

## 1. Requisito / evidencia existente / trabajo faltante

«Origen» distingue lo que **exige el profesor** (PDF *Tarea Final DL 2026-2*, E2) de lo que es **recomendación de implementación**. La evidencia de `docs/project/evidence/stage1-4` fue generada por un compañero: es antecedente y **no acredita la ejecución del usuario**, que la rehace en su propio Colab/Drive con los notebooks E2.

| # | Requisito | Origen | Antecedente existente (compañero) | Qué hace el usuario / estado |
| --- | --- | --- | --- | --- |
| 1 | Teardown: parámetros y FLOPs por bloque | Profesor | [Etapa 4](evidence/stage4/colab_run_20261029T162558Z/review.json): SAM 38 962 498 parámetros, 296.05 GFLOPs parciales | `E2_01`: `run_stage4.py --models sam` en su Colab. **Pendiente de ejecutar** |
| 2 | Visualización de features intermedias | Profesor | Etapa 4: figuras y NPZ | Igual (celda de teardown) |
| 3 | Mapa de campo receptivo efectivo | Profesor | Etapa 4: ERF por gradiente (sensibilidad local) | Igual |
| 4 | Latencia y memoria | Profesor | Etapa 4: 148.6/151.2 ms, 652.7 MiB (T4, batch 1, float32) | Igual; hardware y versiones se registran. Imagen vs prompt por separado en el método automático: no existe aún |
| 5 | Modelo base evaluado | Profesor (implícito) | Etapa 3: IoU agua val 0.158 (referencia asistida, sobresegmenta) | `E2_01`: `run_stage3.py`; `E2_02` recalcula el brazo congelado en su subconjunto |
| 6 | Intervención núcleo con ablación controlada | Profesor | No existía | **Implementada** (`intervention.py`, `run_e2.py`); **pendiente de ejecutar** |
| 7 | Hipótesis escrita ANTES de la ablación | Profesor | No existía | **Hecho**: `configs/e2_intervention.yaml` + artefacto publicado antes de cargar/entrenar |
| 8 | Cuaderno de decisiones: decisión, justificación, uso de IA | Profesor | `cuaderno_decisiones.md` (etapas 1–4) | **Entrada E2 añadida**; resultados e interpretación **PENDIENTES** |
| 9 | Defensa oral individual | Profesor | — | Del estudiante (ver PASOS_E2) |
| 10 | Mínimo de imágenes / épocas | **No fijado por el PDF** | — | Recomendación propia: 3 + 4 chips por evento, ≤5 épocas; ajustable solo antes de ejecutar y registrándolo |
| 11 | Datos desde cero (Sen1Floods11) en el Drive del usuario | Pedido del proyecto | Etapa 2 del compañero | `E2_01` ejecuta `prepare_stage2.py`, que **ya no depende** de la evidencia de etapa 1 ausente/ajena |
| 12 | Decoder congelado vs ajustado, solo decoder entrenado, gradientes verificados | Pedido del proyecto | — | Implementado y probado en CPU; verificación real queda en `report.json` (`freezing`, `gradient_audit_first_step`) |
| 13 | Puntos simulados como simplificación temporal | Pedido del proyecto | Etapa 3 | Reutilizado; **DINOv3→SAM automático no implementado ni usado** (DINOv3 solo opcional en el teardown) |
| 14 | Licencia de Sen1Floods11 | Riesgo de proyecto | Solicitud enviada | **Pendiente** de respuesta del publicador; la evidencia de E2 excluye imágenes de entrenamiento |

## 2. Qué se probó localmente (y qué no)
Ver [local_validation.json](evidence/e2/local_validation.json). **Sintético, CPU, sin datos reales:** 70 pruebas (50 previas + 20 de E2: pérdidas con valores calculados a mano, unión y selección por IoU *predicho*, backward de dos pasadas = autograd, congelamiento, entrenamiento + reanudación idéntica, contrato de config y, con pesos **aleatorios** de SAM 2.1 tiny, conteos de parámetros, paridad con el predictor de la etapa 3 y gradientes solo en el decoder); `tests/smoke_e2_pipeline.py` (`prepare`→`recover` con chips sintéticos); sintaxis de fuentes y celdas; notebooks sin salidas ni hashes/tokens incrustados; todos los archivos que invocan los notebooks existen en el repositorio; la lógica de clonado y fijado del commit se probó contra el repositorio público de GitHub.

**No validado localmente:** pesos reales de SAM, CUDA/T4 y tiempos, Drive, la descarga de Sen1Floods11, el bootstrap en Colab, y cualquier métrica o conclusión científica.

IA: Claude preparó código, pruebas y documentación; el equipo ejecuta, revisa y defiende.
