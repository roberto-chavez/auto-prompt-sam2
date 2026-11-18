# Etapa 8 — protocolo congelado y ejecución

Fecha: 2026-11-18 (America/Lima). Estado: preflight aprobado; ejecución geográfica todavía pendiente.

## Revisión y autorización

El usuario confirmó haber revisado la etapa 7 y autorizó continuar. El recibo humano es [`human_review_stage7_user_20261118.json`](evidence/e3/human_review_stage7_user_20261118.json); el snapshot original [`revision_stage7_20261118.json`](evidence/e3/revision_stage7_20261118.json) permanece inalterado. El preflight [`stage8_preflight_human_reviewed_20261118.json`](evidence/e3/stage8_preflight_human_reviewed_20261118.json) pasa `scripts/check_stage8_gate.py` y conserva `reserved_access=[]`.

## Corrección del protocolo antes de acceso reservado

El borrador comparaba `a3_tuned` (ruta directa de entrenamiento) con `a1_points` (API predictor). Eso mezcla el cambio del decoder con una diferencia de ruta. Antes de leer imágenes o etiquetas reservadas se añadió `a3_frozen`:

- `a3_tuned` vs. `a3_frozen`: misma ruta directa, mismos prompts; cambia solo el checkpoint del decoder.
- `a3_frozen` vs. `a1_points`: mismo decoder base y política de prompts; cuantifica la diferencia entre ruta directa y predictor.
- `a1_points` vs. `a1_assisted`: contraste descriptivo de fuente; la fila asistida usa etiquetas para simular puntos y no es desplegable.

El protocolo está congelado en [`configs/stage8_protocol.json`](../../configs/stage8_protocol.json). Hash SHA-256 canónico: `0ccacde9dbf259d1d4403085e331ef34b93fc2381291638d69cd14a0c3cbc681`. El [registro de congelamiento](evidence/e3/stage8_protocol_freeze_20261118.json) conserva la fecha, el hash de archivo y que no hubo acceso reservado previo.

## Ejecución en Colab

1. Abrir [`notebooks/08_evaluacion_final_geografica.ipynb`](../../notebooks/08_evaluacion_final_geografica.ipynb) desde el commit fijado en `MyDrive/auto-prompt-sam2/auto-prompt-sam2/stage8/code_commit.txt`.
2. Usar runtime Colab GPU con Python 3.12/Linux x86_64. Se requiere `MyDrive/auto-prompt-sam2/auto-prompt-sam2` con artefactos originales de etapas 1–7 y checkpoints. El cuaderno detiene la ejecución si falta algún requisito; no cambia pines ni modelo.
3. Ejecutar celdas en orden. La revisión, autorización y hash se verifican antes de abrir imágenes/etiquetas. No reutilizar el notebook de etapa 7 para esto.
4. Primera ejecución con `RESUME=False`. Tras una interrupción técnica, continuar con el mismo código, root, hash y runtime mediante `RESUME=True`.
5. Guardar `stage8_report.json`, el notebook ejecutado y el log. Máscaras y salidas por chip permanecen en Drive privado; no incluirlas en Git mientras la licencia del dataset siga pendiente.

El runner conserva resultados por chip, hashes de máscara y métricas en Drive; los resultados resumidos informan IoU pooled, métricas por evento, mediana/IQR por chip, bootstrap pareado de 2 000 muestras, comparación de ruta y cambio visto/reservado. No se elige ningún umbral, checkpoint o fila con datos reservados. Los análisis son de una semilla y el contraste entre conjuntos geográficos es descriptivo.

## Estado de ejecución

- Preflight y autorización: aprobados.
- Protocolo: congelado antes del acceso reservado.
- Código/notebook: preparado; todavía sin ejecución Colab.
- Imágenes/etiquetas reservadas leídas: no.
- Resultados de etapa 8: no disponibles todavía.
- Licencia de Sen1Floods11: respuesta del publicador pendiente.
