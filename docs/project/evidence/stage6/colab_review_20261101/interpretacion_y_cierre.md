# Cierre de etapa 6 — 2026-11-01

Etapa **Concluida**, con límites de evidencia explícitos. Se revisaron notebook original con salidas y los dos ZIP; CRC, SHA coincidentes con las salidas de Colab, fuentes experimentales coincidentes con el repositorio, identidades, métricas, selección y recuperación correctos. Los originales permanecen sin modificar. ZIP completos archivados en `artifacts/stage6-evidence/`; notebook y reportes en esta carpeta.

| Régimen | Mejor época | IoU agua val | Etapa 5 | Cambio (pp) | Pasos |
| --- | ---: | ---: | ---: | ---: | ---: |
| official | 10 | 0.484067 | 0.298615 | +18.5452 | 410 |
| leave_paraguay_out | 8 | 0.505815 | 0.316842 | +18.8974 | 350 |

Diez épocas por régimen, AdamW 1e-5, FP32, acumulación 4 y bases SAM independientes. No son métricas de evaluación final. La hipótesis de mejora en validación recibe apoyo en ambas ejecuciones; no demuestra transferencia geográfica ni significancia estadística con una sola semilla. Se conserva la caída del régimen excluido tras época 8: época 9 = 0.489551, época 10 = 0.493663; best permanece época 8. No se amplió presupuesto ni se hicieron ablaciones.

Se recalcularon exactamente TP/FP/FN/TN de 1 640 máscaras val (890 official y 750 excluido) con etiquetas permitidas verificadas por hash. El revisor portátil recalculó agregados y best para train/val. No se recalcularon áreas físicas desde las máscaras en esta auditoría; se verificó su coherencia agregada. Train corresponde a métricas online durante aprendizaje, no a una pasada independiente del checkpoint final.

Colab Tesla T4, CUDA disponible, Python 3.12.13 y PyTorch 2.7.1. Interrupción tras época 1; boot IDs 323cb76a-f894-4b8d-bb74-d43771680860 y d64509fb-d52a-4372-a067-97df906f2b25. Restauración exacta de estados y replay de un paso con rollback acreditados por reportes y código coincidente. Gradientes finitos/no nulos solo en decoder, parámetros congelados sin gradientes, hashes fuera del decoder intactos en las diez épocas. Prueba de pérdida invariante y gradiente cero en inválidos ejecutada. DINO no fue instanciado: features congeladas reutilizadas con linaje verificado.

Cobertura: official 162/252 train con prompts, de los cuales uno completamente inválido; 161 contribuyentes, 90 sin componentes. Excluido 139/213 contribuyentes, 74 sin componentes. En val: 61/89 y 51/75 con prompts; los restantes conservan predicción vacía y cuentan en métricas. Esta cobertura parcial es una limitación sustancial del método, aceptada como resultado de la política prefijada; no justifica inventar respaldo asistido ni exigir mejora en todos los chips. El presupuesto completo sí recibió actualizaciones suficientes para demostrar la intervención. No se resuelve aquí una política alternativa.

Last, best, epoch_N y decoder_export constan en inventarios con rutas persistentes, tamaño, SHA e identidad. La publicación y lectura verificadas por hash y la recuperación fueron ejecutadas en Colab. Los bytes de checkpoints grandes no se incluyeron en el transporte: no se afirma auditoría independiente local de sus tensores. Conservar las carpetas de Drive y sus inventarios para reutilización posterior.

El contador de prepare está vacío aunque conserva su salida completa; entorno/resume/inspección muestran contadores 1/2/3. Se acepta como limitación documental, corroborada por la interrupción, curvas, hashes y boot IDs, sin alterar el notebook. No se recibieron los archivos de log separados; se preservan las salidas del notebook y reportes. No hubo errores en sus salidas de cómputo.

Sin pendientes técnicos que impidan el cierre de etapa 6 dentro de este alcance de evidencia. Persisten la salvedad histórica de licencia, conservación de artefactos en Drive y revisión académica humana. Asistencia IA: implementación y auditoría CPU por Codex; ejecución Colab por usuario. Sin modelos locales, sin reservados, sin campaña final ni demo. Etapas 7–9 permanecen sin iniciar.
