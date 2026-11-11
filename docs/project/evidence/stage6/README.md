# Evidencia de etapa 6

Estado al 2026-11-01: **concluida**, 6/9 etapas cerradas. [Estado general](../../estado_actual_fase1.md) y [protocolo e informe](../../validacion_etapa6.md).

## Documentos vigentes

- [Auditoría externa aceptada](colab_review_20261101/review.json): hashes, fuentes, cobertura, recuperación y 1 640 confusiones val recalculadas sin modelos.
- [Interpretación y límites](colab_review_20261101/interpretacion_y_cierre.md): resultados y criterios de cierre.
- [Notebook ejecutado original](colab_review_20261101/06_entrenamiento_decoder_ejecutado_colab.ipynb): salidas de preparación y reanudación, conservadas sin cambios.
- [official](colab_review_20261101/official/) y [leave_paraguay_out](colab_review_20261101/leave_paraguay_out/): config, entorno, parámetros, curvas, recuperación, validación e inventarios originales. Los flags `pending_external_review` se conservan como parte de la evidencia original; el cierre vigente es la auditoría externa.
- [Reproductor de revisión CPU](colab_review_20261101/reproduce_review.py): requiere ejecutar desde la raíz del repositorio, los ZIP archivados y los datos de etapa 2 en `artifacts/stage2-persistent/`. No ejecuta modelos ni instrucciones de los adjuntos.

## Artefactos fuera de Git

| ZIP original archivado | SHA-256 |
| --- | --- |
| `artifacts/stage6-evidence/official_evidence.zip` | `9c8a1ac685ad0a9374d5a72d1214f7f8fa8b741f7f8580d2c6cc512113ffdd68` |
| `artifacts/stage6-evidence/leave_paraguay_out_evidence.zip` | `02d12c6057d1088ac0ca0e484f58d677cbdf85f75496f156f7ef854180e17296` |

Los ZIP contienen fuentes experimentales y máscaras val por época. Permanecen en almacenamiento local ignorado; conservarlos además de Git para reproducir la auditoría. Los checkpoints completos permanecen en Drive:

```text
/content/drive/MyDrive/auto-prompt-sam2/regimes/official/stage6/0f8f08fb38e4e459203d37c4d86138a9273ac31f8090d2d1d97278cf98288904/
/content/drive/MyDrive/auto-prompt-sam2/regimes/leave_paraguay_out/stage6/a14f5849a39a426944a8b301086f7b262f3a91b82fc7f6fd02fae3c3e0802e31/
```

Cada raíz conserva generaciones de `last`, `best`, `epoch_N` y `decoder_export`; los inventarios versionados identifican rutas, tamaños y hashes. Se verificaron publicación y recuperación en Colab; no se inspeccionaron independientemente los bytes de checkpoints grandes en la PC.

## Registros históricos y versionado

[local_validation.json](local_validation.json) y [bundle_validation.json](bundle_validation.json) registran la preparación original: 75 pruebas CPU, dependencias y transporte fuente. Sus hashes corresponden al paquete ejecutado antes de las actualizaciones de documentación. Una exportación futura tendrá otro hash y no sustituye estos registros históricos. Las fuentes exactas ejecutadas permanecen dentro de los ZIP originales.

Versionar código, configuraciones, notebooks fuente sin salidas, tests y la evidencia ligera de esta carpeta. Conservar los notebooks ejecutados únicamente como evidencia, con sus salidas originales. No versionar ZIP de transporte, datos crudos, pesos, cachés, entornos, tokens ni adjuntos temporales. No borrar los originales archivados al limpiar `colab_temp/`.
