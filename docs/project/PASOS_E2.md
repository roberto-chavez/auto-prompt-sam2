# PASOS_E2 — Cómo ejecutar E2 desde GitHub en tu Colab

> **Actualización 2026-10-30:** estos pasos ya se ejecutaron una vez en Colab (run `20261030T013250Z_2a795a89d5_f9fbd8b0`). Resultados, evidencia y límites: [validacion_e2.md](validacion_e2.md) y [evidencia](evidence/e2/colab_run_20261030T013250Z/README.md). Para reproducir, siga usando esta guía; la acreditación de E2 sigue pendiente.

**Alcance: solo E2** (teardown del modelo base + intervención preliminar + cuaderno de decisiones). **Estado: ejecutado una vez (2026-10-30), pendiente de revisión; no acreditado.** Las métricas solo valen para ese run y sus límites (ver validacion_e2.md).

Todo se hace en **tu** cuenta de Colab y **tu** Drive. No dependes de archivos de tu compañero: la evidencia que hay en `docs/project/evidence/` es antecedente y no acredita tu ejecución.

## Qué notebook abrir (rama `_E.V_`)
| Orden | Notebook | Abrir en Colab |
| --- | --- | --- |
| 1 | `notebooks/E2_01_datos_baseline_teardown.ipynb` | https://colab.research.google.com/github/roberto-chavez/auto-prompt-sam2/blob/_E.V_/notebooks/E2_01_datos_baseline_teardown.ipynb |
| 2 | `notebooks/E2_02_intervencion_decoder.ipynb` | https://colab.research.google.com/github/roberto-chavez/auto-prompt-sam2/blob/_E.V_/notebooks/E2_02_intervencion_decoder.ipynb |

## Antes de empezar (una sola vez)
1. **Runtime:** *Entorno de ejecución → Cambiar tipo de entorno de ejecución → T4 GPU*.
2. **Secreto `HF_TOKEN`:** ya lo tienes en *Secretos* (icono de llave). Comprueba que esté activado **«Acceso desde el notebook»** para cada cuaderno. El código lo lee por sí mismo; nunca lo pegues ni lo imprimas. SAM es público: el token solo sería indispensable para DINOv3 (opcional).
3. Espacio en Drive: reserva varios GB libres en `MyDrive/auto-prompt-sam2` (datos, pesos, cachés, checkpoints).

## Orden exacto de ejecución
**Notebook 1 — `E2_01_…`** (ejecuta las celdas de arriba abajo):
1. *Entorno*: clona el código desde GitHub, fija el commit en tu Drive (`e2/code_commit.txt`), crea el entorno. Debe imprimir el commit y «Entorno listo.».
2. *GPU*: debe nombrar la GPU.
3. *Datos (etapa 2)*: descarga Sen1Floods11 a tu Drive y prepara splits/estadísticas. Es la parte más lenta; si Colab se corta, vuelve a ejecutar «Entorno» y esta celda.
4. *Verificación offline* de los datos.
5. *Línea base (etapa 3)* y su resumen.
6. *Teardown (etapa 4, solo SAM)* y sus tablas/figuras.
7. (Opcional) DINOv3: déjalo en `False`.

**Notebook 2 — `E2_02_…`**:
1. *Entorno* y *GPU* (mismo commit).
2. *Prerrequisitos*: deben decir OK para los datos y AVISO/OK para etapas 3–4.
3. *Hipótesis y configuración* (solo lectura).
4. `MODE='prepare'`: la celda larga (pruebas, paridad, época 0, entrenamiento, evaluación).
5. *Resultados* y *borrador del cuaderno de decisiones*.
6. **Descargar evidencia** (celda 8).
7. **Recuperación:** desconectar y eliminar el entorno → reconectar T4 → ejecutar solo *Entorno*, *GPU*, *Prerrequisitos* y la celda `recover`. No repitas `prepare`.

## Qué deberías ver si funciona
- `prepare`: «HYPOTHESIS (recorded before training)», «parity ok», «epoch 0 (frozen) val pooled IoU: …», una línea por época con `seconds`, y un JSON final con `frozen_val_pooled_iou`, `tuned_best_val_pooled_iou`, `best_epoch`.
- Resultados: `history.csv`, `per_chip_val.csv`, curvas y ejemplos (TP verde, FP rojo, FN azul). **Un resultado malo o negativo también es válido** y se reporta tal cual.
- `recover`: `frozen_arm_check.passed: true` y `tuned_arm_check.passed: true`.

## Si hay errores
- Detenciones intencionales (`parity`, `Gradient audit`, `Frozen blocks changed`, `PREREQUISITE MISSING`): no las saltes; envía el log.
- «NO HAY GPU»: cambia a T4 GPU y vuelve a empezar el notebook.
- Out of memory: guarda `e2/notebook_logs/prepare.log` y avísame; no cambies el tamaño del experimento sin registrarlo.
- Cambió la rama en GitHub y quieres su código: pon `UPDATE_CODE = True` **solo** si aceptas rehacer E2 desde `prepare`.
- Logs: `MyDrive/auto-prompt-sam2/e2/notebook_logs/`.

## Cómo guardar y devolver la evidencia
1. Descarga cada **notebook ejecutado** (Archivo → Descargar → .ipynb, con salidas y errores).
2. Descarga el **ZIP final** que genera la celda 8 del notebook 2 (informe, tablas, curvas, ejemplos, predicciones, puntos, borrador del cuaderno de decisiones y commit usado). No incluye checkpoints, pesos, tokens ni imágenes de entrenamiento.
3. Si hubo errores, los `*.log`.
4. Datos, pesos, cachés y checkpoints **se quedan en tu Drive**; nunca van a GitHub.
Devuélvelos en el chat o súbelos a la rama `_E.V_` en `docs/project/evidence/e2/<fecha>/` (solo archivos ligeros).

## Qué falta para acreditar E2
Ejecución real de ambos notebooks (incluido `recover`) · revisión de tus salidas · interpretación del equipo (qué pasó, por qué, fallos, límites) en la entrada de E2 de `docs/project/cuaderno_decisiones.md` · respuesta de licencia de Sen1Floods11 (pendiente) · que cada integrante pueda defenderlo. Nada de esto está hecho todavía.

## Conceptos que debes dominar para la defensa oral
Arquitectura de SAM 2.1 (encoder de imagen Hiera ~27.2 M, prompt encoder ~6 mil, mask decoder ~4.2 M; el encoder concentra el cómputo) · qué es congelar y cómo se verifica (hash de bloques, auditoría de gradientes) · por qué la API de inferencia no sirve para entrenar (`no_grad`) · salida multimáscara y selección por IoU **predicho** (nunca por IoU real) · pérdida BCE+Dice con píxeles inválidos ignorados · hipótesis previa y criterio de refutación · límites (puntos simulados desde etiquetas, subconjunto pequeño, una semilla, validación optimista, sin test, sin DINOv3, sin significancia) · métricas (IoU/Dice agrupado vs por chip, precisión/recall) · teardown: parámetros ≠ FLOPs ≠ latencia, ERF por gradiente es sensibilidad local · política de IA del curso: delegar la ejecución es válido, delegar la comprensión no.
