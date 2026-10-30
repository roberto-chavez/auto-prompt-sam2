# Evidencia E2 — ejecución en Colab del 2026-10-30 (run `20261030T013250Z_2a795a89d5_f9fbd8b0`)

Carpeta de evidencia **entregada por el usuario** (notebooks ejecutados y ZIP final de su Colab/Drive). Nada aquí se volvió a entrenar ni a calcular: los archivos de `run/` son copias byte a byte del ZIP `20261030T013250Z_2a795a89d5_f9fbd8b0-final.zip`, y los de `notebooks/` copias de los `.ipynb` ejecutados. Los notebooks **originales ejecutables** siguen en [`notebooks/`](../../../../../notebooks/) (`E2_01_datos_baseline_teardown.ipynb`, `E2_02_intervencion_decoder.ipynb`) y no fueron modificados.

Estado: **ejecutado y documentado; la calificación corresponde al profesor y no se da por acreditado** (ver [validacion_e2.md](../../../validacion_e2.md) y [auditoria_E2.md](../../../auditoria_E2.md)).

## Procedencia (qué es nativo y qué se derivó)

| Carpeta | Origen | Observación |
| --- | --- | --- |
| `notebooks/` | `.ipynb` ejecutados en Colab (T4, runtime 2026.07) | Revisados antes de publicar: 0 salidas de error, sin tokens/credenciales/correos; metadatos solo `colab.runtime_attributes`. `HF_TOKEN` se lee de los secretos de Colab y su valor no se imprime. |
| `run/` | ZIP `…-final.zip` (18 archivos, íntegro) | Nativo del run E2: `report.json`, `history.csv`, `per_chip_val.csv`, `curves.png`, `examples/` (10 chips), `points_val.json`, `predictions_val.npz`, `code_commit.txt`, `e2_resultados_para_cuaderno_decisiones.md`. |
| `derived_from_notebooks/` | **Extraído de las salidas incrustadas de `E2_01…ejecutado.ipynb`** | Tabla de FLOPs por bloque y de latencia/memoria de SAM, y figura de teardown, tal como los imprimió el notebook (HTML/PNG). **No son los archivos nativos** del run de etapa 4 (ver «Evidencia no incluida»). `recover_output_literal.txt` es la salida literal de la celda `recover` de `E2_02…ejecutado.ipynb`. |

Excluido deliberadamente: pesos (`.pt`), checkpoints, GeoTIFF originales, caches y cualquier credencial. Los PNG de `run/examples/` y `curves.png` contienen vistas RGB y máscaras de chips de Sen1Floods11: son derivados visuales, publicados con la **licencia del dataset aún pendiente de respuesta del publicador** (ver `docs/project/accesos_y_licencias.md`); si el equipo prefiere no publicarlos hasta recibir respuesta, basta retirar `run/examples/` y `curves.png` del índice.

## Identidad del experimento
- Run `20261030T013250Z_2a795a89d5_f9fbd8b0`; `code_commit.txt` = `f399047464b89a3ed18972f8dc7f1b1364d83a44` (rama `_E.V_`); `config_sha256` `3fbb165b…531e89a`; SAM 2.1 hiera-tiny, revisión `de431c40…`, checkpoint sha256 `7402e0d8…34be69`, estado `72a85430…12384d` (coincide con los de las etapas 3 y 4 auditadas).
- Tesla T4, Python 3.12.13, torch 2.7.1 + CUDA 12.6, 50 pines verificados, `pip check` correcto, `tests.exit_code = 0`.

## Qué prueba y qué no prueba `recover`
- **Existe:** salida literal de `run_e2.py recover` (celda 15, `execution_count` 8, misma sesión de notebook que `prepare`): re-evaluación del brazo congelado y del brazo ajustado (mejor checkpoint) con diferencia de IoU agrupado `0.0`, confusiones por chip idénticas, `mask_disagreement_pixels = 0`, `passed = true` en ambos. Recarga de checkpoints con `roundtrip_equal = true`.
- **No existe:** evidencia de un **reinicio real del runtime** de Colab (no hay boot IDs distintos, ni reconexión, ni `uptime`, ni nuevo `Mounted at` entre `prepare` y `recover`; ambos notebooks muestran un único montaje de Drive y `execution_count` consecutivos 1…9). `recover` corre en un **subproceso Python nuevo dentro de la misma sesión**; eso demuestra que el estado en Drive basta para reproducir las cifras, no que sobreviva a un reinicio del runtime. (Para las etapas 3 y 4 del compañero sí hay boot IDs distintos, pero no acreditan esta ejecución.)

## Evidencia no incluida (sigue solo en Drive)
La preparación de datos, la línea base y el teardown (`E2_01`) se **reejecutaron** con `prepare` en el Colab del usuario y escribieron sus artefactos en Drive; **este repositorio no contiene los ZIP nativos de esa reejecución** (sí contiene sus salidas incrustadas, en `derived_from_notebooks/`). Los archivos nativos del teardown **del equipo** (etapa 4, mismas configuración y pines) **sí están en el repositorio**: [`stage4/…/verified_artifacts/`](../../stage4/colab_run_20261029T162558Z/verified_artifacts/) (`sam_parameters.csv`, `sam_official_0_flops.csv`, `sam_official_0_timings.csv`, features/ERF JSON y figuras). Los ZIP originales de las etapas 3 y 4 se localizaron en el Drive conectado (no se descargaron por su tamaño):
- Etapa 4 (teardown), ZIP: id `1d7D_vA94a0PQztbi779zDvfFy305bUO7` → trae el CSV de parámetros por bloque, arrays de features/ERF y figuras nativas.
- Etapa 3 (línea base), ZIP: id `1o4GqQ5DYq5Mt9Z5erb9SMbEs7C-cM8pa` → trae predicciones y confusiones nativas.
Para el análisis de la ejecución propia no hacen falta; son evidencia opcional. Los parámetros por bloque (27 219 136 / 6 220 / 4 215 109 / otros 7 522 033; total 38 962 498) constan en `run/report.json` (`parameters_before`). FLOPs, latencia y memoria constan solo en las tablas incrustadas de `derived_from_notebooks/`.

## Reproducir
1. Abrir `notebooks/E2_01_datos_baseline_teardown.ipynb` y luego `E2_02_intervencion_decoder.ipynb` desde GitHub (rama `_E.V_`) en Colab con GPU T4; fijar `PINNED_COMMIT` al commit deseado. Guía: [PASOS_E2.md](../../../PASOS_E2.md).
2. Todo el estado vive en Drive: raíz `/content/drive/MyDrive/auto-prompt-sam2/`. Para este run:
   - Run y checkpoints: `e2/runs/20261030T013250Z_2a795a89d5_f9fbd8b0/checkpoints/` (mejor: generación `20261030T013915336249Z_81b60bb0014c`, sha256 `11d21c5b810415b4506671ecc7f9694e0d71f03395cac071db0b56e6b081fe1e`, 16 902 201 B; última: `20261030T014721828771Z_e57b2f09960a`, sha256 `eb227defab5c81ca0dc42612a52aee139a37ba91e23b115250daf44a3de89e4c`, 65 309 505 B). **Nunca a Git.**
   - Evidencia exportada: `e2/evidence/20261030T013250Z_2a795a89d5_f9fbd8b0-final.zip` (SHA-256 `4e67a5fb03d69460d2d1cf236ab61c42ba67e24cc0d4800c665af78a8be3eac4`).
   - Commit usado por los notebooks: `e2/code_commit.txt`.
3. La semilla es 42; el backward del upsample bilineal en CUDA no es determinista (advertencias en `report.json`, `nondeterminism_warnings`), por lo que una reejecución completa puede diferir levemente; `recover` verifica la repetición exacta de la **evaluación**, no del entrenamiento.

## Archivos de verificación independiente (carpeta superior)
[`../audit_recomputation.json`](../audit_recomputation.json) (recálculo de métricas desde `per_chip_val.csv` y control cruzado con la etapa 3 propia; script `scripts/audit_e2_recompute.py`) y [`../teardown_architecture_counts.json`](../teardown_architecture_counts.json) (conteo de sub-bloques; script `scripts/teardown_architecture_counts.py`). Resumen y veredicto: [auditoria_E2.md](../../../auditoria_E2.md).

## Índice y hashes
Ver [`SHA256SUMS.txt`](SHA256SUMS.txt) (27 archivos, incluye los dos de arriba) (verificable con `sha256sum -c SHA256SUMS.txt` desde esta carpeta). Hashes de los originales del usuario: `E2_01…ejecutado.ipynb` `09e26671…cee`, `E2_02…ejecutado.ipynb` `c5b0733b…ac16ac`, ZIP final `4e67a5fb…eac4`.
