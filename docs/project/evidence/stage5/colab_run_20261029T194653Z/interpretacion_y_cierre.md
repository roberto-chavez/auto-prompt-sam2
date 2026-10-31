# Etapa 5: interpretación y cierre — 2026-10-30

Se acepta el run `20261029T194653Z_8073c068fa_ed6583db` tras revisión del notebook, reportes y nueva descarga. **Etapa 5 concluida; 5 de 9 etapas cerradas.** Etapas 6–9 y demo sin iniciar.

## Integridad y revisión

El ZIP nuevo coincide con el SHA impreso por recover: `504f2b9da5cdef97bb933be59a89c7b8af03f00b8bee939217244700ae115cef`, 1 517 820 977 bytes. Está conservado bajo `artifacts/stage5-evidence/` y no se repara ni se ejecuta código contenido en él. La descarga anterior con SHA `771d…fd77` fue sustituida por el usuario; se preservan sus diagnósticos de 1 067 entradas ilegibles como antecedente, ya resuelto. La causa precisa de aquella alteración no se identifica.

[Auditoría completa](artifact_review_portable.json), [comprobaciones independientes](independent_checks.json) y [cierre](closure_review.json). CRC e inventario sin duplicados, generaciones incluidas verificadas por tamaño/hash/identidad, 55 fuentes idénticas al paquete entregado, 3 078 unidades del índice. Las 1 640 predicciones corresponden a ocho filas de puntos y dos de malla sobre 89/75 val; confusiones, métricas y selección se recalcularon sin modelos. Test, Bolivia y Paraguay reservado no se abrieron. Manifiestos y preprocesamiento de cada régimen coinciden con etapa 2.

El ZIP compacto omite 627 payloads de features, que permanecen en Drive: se verifican sus marcadores y el reporte de comprobación completa de Colab, no sus bytes de manera independiente ni el scaler recalculado desde todos esos arrays. Se reconstruyen desde disco las probabilidades, componentes/puntos del primer val por régimen y la aceptación semántica de malla; las métricas de todas las predicciones se comprueban con etiquetas val permitidas. Esta limitación prevista se conserva y no se presenta como auditoría externa completa de todas las features.

Dos celdas del notebook tienen contador nulo pero salidas preservadas. Se comparó el código de todas las celdas: coincide exactamente con el original salvo completar el SHA del bundle. Las salidas y reportes inmutables acreditan prepare/recover sin errores. Se aceptó esa evidencia en lugar del contador obligatorio del auditor estricto. No se modificó el notebook. Las áreas físicas EPSG:6933 recalculadas en Windows difieren hasta 2.970919013e-6 m²: se aplica la tolerancia ya usada en etapa 3 (atol=1e-5 m², rtol=1e-10); todos los demás campos de métricas y TP/FP/FN/TN se exigen exactos. Los agregados usan las áreas originales únicamente tras verificar individualmente la tolerancia. [Adaptador reproducible de revisión](reproduce_portable_review.py); fuentes experimentales intactas. Los intentos estrictos fallidos permanecen para trazabilidad.

## Ejecución y aislamiento

Prepare y recover se realizaron en Colab Tesla T4, Python 3.12.13/PyTorch 2.7.1, boot IDs diferentes. Ocho entrenamientos de probe en prepare, ninguno en recover; 66 pruebas y pip check correctos por sesión. DINOv3 y SAM conservan hashes de checkpoints y fingerprints de estado antes/después; los checkpoints coinciden con los revisados en etapa 4. Decoder congelado, cero campaña final. La recuperación offline repite features del primer val y reproduce exactamente puntos y máscaras, con scores dentro de tolerancias previas. La malla se reconstruye desde candidatos persistidos, sin repetir AMG/NMS.

Official usa 252 train y 89 val; excluido 213 y 75, con scaler/probe y estadísticas propios. Targets de entrenamiento: official 19 538 agua, 192 017 no agua y 46 493 ignorados; excluido 16 660, 158 771 y 42 681. Los parches sin validez están incluidos en los ignorados (25 667/24 377). No se mezclaron parches train/val. El flujo automático consume únicamente imagen/features/validez de entrada; etiquetas posteriores sirven a evaluación y targets train, sin validez manual en prompts.

## Selección e interpretación

Ambos regímenes eligen, independientemente, C=0.1, sin balanceo y umbral positivo 0.7; 128/129 iteraciones (official/excluido). El probe C=1 balanceado de cada régimen agotó el presupuesto sin convergencia: sus filas se conservaron y excluyeron según el criterio previo. No se amplió la búsqueda. Umbral semántico de malla seleccionado 0.5 en ambos, con probe fijado por puntos.

| Métrica val | official puntos | excluido puntos | official malla | excluido malla |
| --- | ---: | ---: | ---: | ---: |
| IoU agua acumulada | 0.298615 | 0.316842 | 0.178537 | 0.208799 |
| Precisión | 0.327153 | 0.354323 | 0.921761 | 0.905603 |
| Recall | 0.773923 | 0.749701 | 0.181284 | 0.213444 |
| Dice | 0.459897 | 0.481215 | 0.302980 | 0.345465 |
| Mediana IoU por chip con agua válida | 0.113111 | 0.139504 | 0 | 0 |

Los puntos logran mayor recall con muchos falsos positivos; la malla es más restrictiva, con alta precisión y baja cobertura. Son resultados de calibración en poblaciones diferentes; la diferencia official/excluido no prueba mejor transferencia geográfica. Tampoco contrasta H1–H3 ni sustituye las ablaciones controladas o la evaluación reservada. No se infieren categorías de fallo (sombras/vegetación) sin evidencia cualitativa específica.

La configuración elegida produce cero componentes en 28/89 y 24/75 chips; se conservaron las máscaras vacías. Los presupuestos reales suman 1 204+/2 055− en official y 966+/1 593− en excluido, con hasta 3+/3− por llamada y déficits permitidos. La mediana baja y los vacíos son limitaciones relevantes para supervisión futura del decoder; no se corrigen ahora con puntos de etiqueta ni nuevas heurísticas.

## Decisiones, límites y parada

NDWI permanece aplazado tal como se registró antes de ejecutar. El probe es supervisado con etiquetas manuales train y selección con val; no es un sistema libre de etiquetas. Los bundles/probes seleccionados se conservan también como JSON legibles en esta carpeta, con generaciones originales en Drive. Los reportes originales mantienen su flag pending; este cierre externo los complementa sin reescribirlos.

No falta repetir entrenamiento ni recuperación por la descarga dañada. Mantener artefactos completos de Drive y el ZIP íntegro; la consulta histórica de licencia del dataset y la revisión/comprensión académica humana siguen pendientes. La demora de recover no tiene diagnóstico temporal por bloque y no se atribuye con certeza a una operación. Se detiene el trabajo aquí, sin iniciar etapa 6.

Asistencia IA: Codex auditó fuentes, notebook, bytes y arrays en CPU, interpretó métricas y actualizó decisiones/estado. No ejecutó modelos, entrenamiento local, inferencia GPU ni comunicaciones externas.
