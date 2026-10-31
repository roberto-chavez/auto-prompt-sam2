# Evidencia de etapa 5

Estado al 2026-10-30: **concluida**, 5/9 etapas cerradas. [Estado general](../../estado_actual_fase1.md) y [cuaderno](../../cuaderno_decisiones.md). Solo desarrollo y auditoría CPU en PC; entrenamiento e inferencia en Colab T4.

## Documentos vigentes

Run aceptado: `20261029T194653Z_8073c068fa_ed6583db`.

- [Cierre y procedencia](colab_run_20261029T194653Z/closure_review.json): resultado externo definitivo de esta etapa.
- [Interpretación y límites](colab_run_20261029T194653Z/interpretacion_y_cierre.md): métricas, configuración elegida, convergencia, alcance y pendientes.
- [Auditoría aceptada](colab_run_20261029T194653Z/artifact_review_portable.json): 1 640 predicciones y selección verificadas, fuentes/generaciones, tolerancia de áreas y excepción de contadores.
- [Comprobaciones independientes](colab_run_20261029T194653Z/independent_checks.json): código del notebook, checkpoint base y alcance.
- [Notebook ejecutado original](colab_run_20261029T194653Z/05_prompts_automaticos_ejecutado_colab.ipynb): salidas conservadas de ambas sesiones.
- [Prepare](colab_run_20261029T194653Z/validation.json), [recover](colab_run_20261029T194653Z/recovery_validation.json), [configuración](colab_run_20261029T194653Z/config.json) y [entorno](colab_run_20261029T194653Z/environment.json): informes originales, sin reescribir sus flags pending; el cierre externo los complementa.
- `official_*` y `leave_paraguay_out_*`: copias legibles de probe seleccionado, bundle completo y selecciones por régimen. Las generaciones originales permanecen en Drive y ZIP.

## Historial de revisión

`review.json` registra el rechazo inicial por contadores nulos; `artifact_review.json`, `external_review.json`, `transport_checks.json`, `full_crc_scan.json` y `revision_parcial.md` documentan la primera descarga dañada. `artifact_review_redownload.json` corresponde al intento estricto con igualdad binaria de áreas físicas entre Linux y Windows. Se conservan para trazabilidad y **no representan el estado final**. La auditoría aceptada usa la tolerancia ya establecida en etapa 3; confusiones y demás métricas permanecen exactas. `reproduce_portable_review.py` reproduce las dos adaptaciones documentadas sin cambiar las fuentes experimentales.

## Archivos fuera de Git

ZIP original íntegro: `artifacts/stage5-evidence/20261029T194653Z_8073c068fa_ed6583db.zip` (ignorado), 1 517 820 977 bytes, SHA-256 `504f2b9da5cdef97bb933be59a89c7b8af03f00b8bee939217244700ae115cef`. Fue archivado desde colab_temp después de su auditoría; se preservó su contenido.

Drive conserva el árbol completo `stage5/environment/runs/<run_id>/` y las generaciones de etapas previas. El ZIP compacto no contiene 627 payloads de features: la revisión externa comprueba sus marcadores y la verificación declarada por Colab, sin recalcular sus hashes ni el scaler desde todos esos arrays. No contiene pesos base ni imágenes crudas.

## Evidencia local de preparación

[local_validation.json](local_validation.json) y [bundle_validation.json](bundle_validation.json) son registros históricos de la preparación: 66 tests CPU y transporte fuente sin modelos/datos. El protocolo [validacion_etapa5.md](../../validacion_etapa5.md) conserva el texto previo a ejecutar, incluido su estado pendiente original, porque forma parte del snapshot de fuentes auditado. Para conocer el cierre actual, usar los documentos vigentes de arriba. Una exportación futura puede tener otro hash de fuente sin reemplazar ese registro histórico.
