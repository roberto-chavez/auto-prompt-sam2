# E2 — Auditoría de requisitos, evidencia existente y validación

**Estado: preparada, NO ejecutada en GPU. E2 no está acreditada.** Última actualización: 29 de octubre de 2026 (America/Lima).

## 1. Requisito / evidencia existente / trabajo faltante

La columna «Origen» distingue lo que **exige el profesor** (PDF *Tarea Final DL 2026-2*, E2) de lo que es **recomendación de implementación** de este repositorio.

| # | Requisito | Origen | Evidencia existente | Trabajo faltante (estado) |
| --- | --- | --- | --- | --- |
| 1 | Teardown: parámetros y FLOPs por bloque | Profesor | Etapa 4 ejecutada y auditada en Colab T4: [revisión](evidence/stage4/colab_run_20261029T162558Z/review.json), SAM 38 962 498 parámetros, 296.05 GFLOPs parciales | Ninguno de código. Pendiente que el equipo lo estudie y lo cite; opcionalmente repetir con su notebook 04 |
| 2 | Visualización de features intermedias | Profesor | Etapa 4: figuras y NPZ ([interpretación](evidence/stage4/colab_run_20261029T162558Z/interpretacion_y_cierre.md)) | Ninguno |
| 3 | Mapa de campo receptivo efectivo | Profesor | Etapa 4: ERF por gradiente (sensibilidad local, no campo teórico) | Ninguno |
| 4 | Latencia y memoria | Profesor | Etapa 4: 148.6/151.2 ms, pico 652.7 MiB (batch 1, float32, T4) | Ninguno. Costos *no* medidos: imagen vs prompt por separado en el método automático (no existe aún) |
| 5 | Modelo base evaluado | Profesor (implícito en teardown + línea base) | Etapa 3: SAM 2.1 tiny congelado, 1 train + 89 val official, IoU agua 0.158 (referencia asistida) | Ninguno; E2 recalcula el brazo congelado con el mismo evaluador sobre su subconjunto |
| 6 | Intervención núcleo con ablación controlada | Profesor | No existía | **Implementado** (`intervention.py`, `run_e2.py`); **pendiente ejecutar en Colab** |
| 7 | Hipótesis escrita ANTES de la ablación | Profesor | No existía para E2 | **Hecho**: `configs/e2_intervention.yaml` + artefacto `hypothesis` publicado antes de cargar/entrenar; entrada en el cuaderno de decisiones |
| 8 | Cuaderno de decisiones: decisión, justificación, uso de IA | Profesor | `cuaderno_decisiones.md` (etapas 1–4) | **Entrada E2 añadida**; resultados e interpretación **pendientes** (se completan tras ejecutar) |
| 9 | Defensa oral individual | Profesor | — | Del estudiante (ver `PASOS_E2.md`, sección 10) |
| 10 | Mínimo de imágenes / épocas | **No fijado por el PDF** | — | Recomendación propia: 3+4 chips por evento, ≤5 épocas (`configs/e2_intervention.yaml`); ajustable solo antes de ejecutar y quedando registrado |
| 11 | Decoder congelado vs ajustado, solo decoder entrenado, gradientes verificados | Pedido del proyecto (no del PDF) | — | Implementado y probado en CPU; verificación real queda en el informe del run (`freezing`, `gradient_audit_first_step`) |
| 12 | Puntos simulados como simplificación temporal | Pedido del proyecto | Etapa 3 (`simulate_points`) | Reutilizado; **DINOv3→SAM automático no implementado y no usado** |
| 13 | Licencia de Sen1Floods11 | Riesgo de proyecto | Solicitud enviada | Pendiente de respuesta; el ZIP de evidencia no incluye imágenes de entrenamiento |

## 2. Qué se probó localmente (y qué no)

Ver [local_validation.json](evidence/e2/local_validation.json). **Sintético, CPU, sin datos reales:** 70 pruebas unitarias (50 previas + 20 de E2: pérdidas con valores calculados a mano, unión y selección por IoU *predicho*, backward de dos pasadas = autograd, congelamiento, entrenamiento + reanudación idéntica, contrato de config y, con pesos **aleatorios** de SAM 2.1 tiny, conteos de parámetros, paridad con el predictor de la etapa 3 y gradientes solo en el decoder); un humo `tests/smoke_e2_pipeline.py` de `prepare`→`recover` con chips sintéticos; sintaxis de todas las fuentes y celdas de los cuadernos (sin salidas ni contadores de ejecución); ZIP sin cuadernos, pesos ni datos, con SHA-256 incrustado en los cuadernos.

**No validado localmente:** pesos reales de SAM, comportamiento/tiempos CUDA en T4, persistencia en Drive, datos Sen1Floods11, cualquier métrica o conclusión científica.

## 3. Cómo se ejecuta
Ver `PASOS_E2.md` (paquete entregado junto a los cuadernos). Resumen: subir `auto-prompt-sam2-e2.zip` a Drive, abrir `01_entorno_colab.ipynb` y luego `02_E2_teardown_intervencion.ipynb` con GPU T4, ejecutar en orden, `recover` en runtime nuevo, devolver el cuaderno ejecutado y el ZIP de evidencia.

IA: Claude preparó código, pruebas y documentación; el equipo ejecuta, revisa y defiende.
