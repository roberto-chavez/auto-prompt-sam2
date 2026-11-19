# Etapa 8 — protocolo, revisión y resultados

Estado al 18 de noviembre de 2026: **ejecución completada**. El texto anterior de esta página que decía «pendiente» quedó obsoleto; esta versión sustituye ese estado.

## Revisión humana y protocolo

El usuario confirmó haber revisado el notebook, logs/hardware, fallos e interpretación de etapa 7. El recibo [`human_review_stage7_user_20261118.json`](evidence/e3/human_review_stage7_user_20261118.json) registra esa auto-confirmación del equipo; no es una revisión externa. El preflight y autorización están en [`stage8_preflight_human_reviewed_20261118.json`](evidence/e3/stage8_preflight_human_reviewed_20261118.json).

El protocolo se congeló antes de abrir los conjuntos reservados: hash canónico `0ccacde9dbf259d1d4403085e331ef34b93fc2381291638d69cd14a0c3cbc681`. [`stage8_protocol_freeze_20261118.json`](evidence/e3/stage8_protocol_freeze_20261118.json) conserva la fecha, el hash canónico y el valor de hash de archivo declarado en ese momento.

## Ejecución y artefactos

La ejecución ocurrió en Colab T4 el 18 de noviembre. El resumen público [`stage8_results_summary.json`](evidence/e3/stage8_results_summary.json) registra estado `complete`, el commit fuente `0cd7b39c0564df2c31322eaac06fa4a866f4de37`, el hash del protocolo, el hash del par de etapa 7 y el SHA-256 esperado del reporte íntegro: `aa95eee0fb110e4e94d39728d5d86ac1fcba06881fa8ea6c940e2c4e4e771f76`.

El reporte íntegro y el log se recuperaron de Drive el 2026-11-18 (ver «Verificación del reporte»); el notebook ejecutado no se encontró en Drive. No están archivados en este clon; el conector de Drive no devolvió esos archivos en esta sesión, por lo que no se volvió a descargar ni recalcular el hash. No se afirma una verificación independiente del reporte íntegro. La tabla de resultados agregados que sigue se basa en el resumen publicado.

## Resultados agregados

| Régimen / conjunto | n chips | Asistido | Automático, congelado | Decoder ajustado |
|---|---:|---:|---:|---:|
| official / test_seen | 90 | 0.1823 | 0.2838 | 0.3974 |
| official / Bolivia reservada | 15 | 0.1961 | 0.4994 | 0.5248 |
| leave_paraguay_out / test_seen | 76 | 0.1961 | 0.3111 | 0.3319 |
| leave_paraguay_out / Paraguay reservado | 67 | 0.1796 | 0.1346 | 0.2540 |

Métrica: pooled water IoU. Las filas `a3_frozen` y `a1_points` coinciden exactamente. Prompt automático menos asistido: Bolivia +0.3033 (IC pareado 95% [0.1785, 0.3821]); Paraguay −0.0451 ([-0.1015, 0.0043]). Decoder ajustado menos congelado: Bolivia +0.0254 ([-0.0472, 0.1430]); Paraguay +0.1194 ([0.0640, 0.1782]). Los IC usan 2 000 remuestreos pareados por chip; el experimento tiene una semilla.

## Integridad y cambios después del congelamiento

Entre el commit de congelamiento `3b03524` y el commit fuente de ejecución `0cd7b39` hubo cambios de plomería:

- `src/auto_prompt_sam2/stage7.py` acepta hashes de código/configuración con saltos de línea LF o CRLF normalizados, para evitar discrepancias de plataforma al recuperar fuentes.
- `scripts/run_stage8.py` verifica el reporte y estado completado de etapa 5, pero omite leer y hashear todos los objetos de caché de etapa 5 porque la etapa 8 vuelve a calcular features y no los consume.
- `scripts/run_stage6.py` añade el parámetro `verify_artifacts`; la omisión de cachés solo se usa en la lectura de etapa 5 de la etapa 8.
- El notebook 08 reinstala pins de etapa 2 y marca el contexto de ejecución Colab.

Estos cambios no modifican filas, umbrales, selección ni pesos del protocolo; sí cambian qué payloads de caché se verifican y cómo se toleran saltos de línea. Debe citarse esta diferencia al describir la procedencia de la ejecución.

El registro de congelamiento declara `protocol_file_sha256=53cc5741…`; los bytes versionados en `3b03524` y los actuales dan `522e8695…`, mientras que el hash canónico declarado (`0ccacde9…`) sí se reproduce. **Causa confirmada (2026-11-18):** `53cc5741…` es exactamente el SHA-256 del mismo archivo con finales de línea CRLF (el congelamiento se calculó sobre una copia de trabajo de Windows); `522e8695…` es el de la versión LF almacenada en Git (`.gitattributes`: `* text=auto`). El contenido es idéntico salvo los saltos de línea. Lo mismo ocurre con `human_review_receipt_sha256=41d6ac87…`: es el hash de `human_review_stage7_user_20261118.json` con CRLF (el archivo LF versionado da `50194ab7…`). Detalle y comandos en [`stage8_hash_reconciliation_20261118.json`](evidence/e3/stage8_hash_reconciliation_20261118.json).

La auditoría de etapa 7 registra los ZIP finales con hashes `df51ddc5…` (official) y `89c24a9a…` (leave_paraguay_out). Hubo generaciones posteriores a la auditoría anterior; para reproducir, usa el par más reciente nombrado en `revision_stage7_20261118.json`, no mezcles generaciones.

## Demo y límites

La demo verificada es un backend manual congelado en un chip. No es la UI Gradio probada de punta a punta ni la demo automática DINOv3→SAM. La UI puede cargar el decoder `best` de E2, que no es el `a3_tuned` usado en etapa 8; por tanto, no representa ese brazo evaluado. Conectar el decoder de etapa 8 requiere disponer de sus pesos y validarlos en GPU. El flujo automático tampoco está conectado a la UI.

Una semilla; checkpoint seleccionado con validación; reservados pequeños; los intervalos bootstrap por chip no sustituyen semillas independientes. La actividad 9 es una extensión interna opcional, no requisito confirmado del PDF del curso.

## Pendientes de archivo y análisis opcional

- El reporte completo (verificado, ver arriba) y el notebook ejecutado, si se recupera, viven en Drive; el reporte se re-verifica con `python scripts/verify_stage8_report.py <ruta-a-stage8_report.json>`. El log está archivado en este repositorio; el reporte no, por contener resultados por chip con licencia pendiente.
- No se hizo análisis visual de casos de mejora y error. Es opcional; no publicar ejemplos de chips hasta guardar el permiso docente y aclarar su alcance para derivados.
- Más semillas y ampliar los reservados son mejoras científicas, no requisitos confirmados del PDF.

## Verificación del reporte (2026-11-18)

- `stage8_report.json` (875,056 bytes, Drive `1_xgdF0-Tz2l4AlxKoCKaAhV2rqW_2D1V`) descargado y comprobado con `python scripts/verify_stage8_report.py`: SHA-256 `aa95eee0fb110e4e94d39728d5d86ac1fcba06881fa8ea6c940e2c4e4e771f76` = esperado → **verificado**. Contiene `status: complete`, `protocol_sha256 0ccacde9…`, `source_commit 0cd7b39…`, `stage7_pair_sha256 996b3587…`. El reporte completo no se versiona (contiene resultados por chip mientras la licencia siga pendiente).
- Log de la ejecución final (`stage8_20261118T235002Z.log`): archivado en [`evidence/e3/stage8_run_log_20261118T235002Z.log`](evidence/e3/stage8_run_log_20261118T235002Z.log) (SHA-256 `67662cb0…f37f`); no contiene datos del dataset, solo progreso por conjunto y fila (90/15/76/67 chips × 4 filas) y las identidades de ejecución. Los otros logs de Drive de esa fecha están vacíos o son intentos truncados (0–721 bytes).
- **Notebook ejecutado de etapa 8: no existe en Drive** (búsqueda por título y `.ipynb` sin resultados); `notebooks/08_evaluacion_final_geografica.ipynb` en Git no tiene salidas. La evidencia de ejecución son el log, el reporte verificado y el resumen. Si el notebook sigue abierto en Colab, descargarlo con las salidas y añadirlo.
- `code_commit.txt` en Drive (`0cd7b39…`) fue modificado a las 00:33Z, después de la ejecución (23:50Z); la identidad de código de la ejecución consta además dentro del reporte (`run_identity.source_commit`).
