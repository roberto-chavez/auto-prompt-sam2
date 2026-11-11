# Etapa 6: entrenamiento exclusivo del mask decoder

Actualizado: 2026-11-01, America/Lima. Estado **Concluida**, tras revisar la ejecución del usuario en Colab T4 y la recuperación en otro runtime. En la PC solo se realizaron desarrollo, pruebas sintéticas y auditoría de resultados guardados, sin modelos ni entrenamiento.

## Resultado y cierre

| Régimen | Mejor época | IoU agua de validación | Mejora frente a etapa 5 |
| --- | ---: | ---: | ---: |
| `official` | 10 | 0.484067 | +18.5452 puntos porcentuales |
| `leave_paraguay_out` | 8 | 0.505815 | +18.8974 puntos porcentuales |

Ambos regímenes completaron diez épocas, con 410 y 350 pasos de optimizador. Se verificaron fuentes, hashes, selección de `best`, recuperación exacta y las confusiones de 1 640 máscaras de validación. Estos resultados apoyan la hipótesis de mejora en validación; la evaluación final y la transferencia geográfica siguen pendientes.

La [auditoría externa](evidence/stage6/colab_review_20261101/review.json), la [interpretación y los límites](evidence/stage6/colab_review_20261101/interpretacion_y_cierre.md) y el [índice de evidencia](evidence/stage6/README.md) documentan el cierre. Los informes originales conservan sus flags de revisión pendiente; el cierre externo los complementa sin reescribirlos.

## Protocolo fijado antes de ejecutar

Hipótesis: ajustar exclusivamente el decoder con prompts automáticos fijos puede aumentar IoU agua de validación frente a etapa 5; cobertura incompleta y errores del probe pueden limitar o empeorar el resultado. Se conservarán resultados negativos. No se elige configuración entre regímenes ni se implementan ablaciones o evaluación final.

`configs/training.yaml` fija 10 épocas, semilla 42, AdamW, LR 1e-5, weight decay 0.01, FP32 sin autocast/AMP, microbatch 1, acumulación de 4 chips que contribuyen y clipping 1. Scheduler constante; sin parada temprana. Último grupo dividido por número real de contribuyentes. Máximo 630/540 actualizaciones para official/excluido respectivamente (63/54 por época); casos sin gradiente reducen ese máximo. Una sola ejecución por régimen, cada una desde SAM base original verificado, nunca desde el decoder oficial entrenado.

Se congelan todos los parámetros SAM salvo `sam_mask_decoder.*`. Los encoders permanecen eval, decoder train durante aprendizaje y eval en validación. DINO no se instancia: se reutilizan exclusivamente features congeladas de etapa 5, verificadas con generaciones, SHA, ID, rol, régimen, modelo y preprocesamiento. No se ajusta probe ni scaler. Se verifica el inventario persistente de etapas 1–5; los reportes previos están fijados en `configs/stage6_dependencies.json`. Etapa 2 usa `stage2_inputs` y los hashes del linaje anterior. No se accede a imágenes/etiquetas reservadas.

Una llamada por componente automático, selección por mayor IoU predicha (primer índice en empate), interpolación bilineal a 512, unión por máximo de logits. Su sigmoid es exactamente el máximo de probabilidades; BCEWithLogits evita saturación numérica de BCE en probabilidades. Pérdida = BCE media de píxeles válidos + Dice de chip, pesos 1/1 y suavizado 1. Etiquetas únicamente en pérdida y métricas posteriores, nunca en prompts ni selección. No se redimensionan etiquetas. El argmax de scores es discreto; el gradiente llega a máscaras seleccionadas y máximos ganadores. La cabeza de scores sin pérdida propia y ramas no seleccionadas pueden no recibir gradiente; no se afirma que cada parámetro del decoder cambie.

Sin componentes: predicción vacía, sin actualización, incluidos en métricas y contadores. Completamente inválidos: sin actualización, contados. No se usan puntos asistidos de respaldo. Cobertura cero bloquea; cobertura baja debe revisarse con evidencia antes del cierre, sin reemplazos ocultos. Se aplica validez de entrada a las predicciones para mantener el contrato de etapa 5; validez de etiquetas solo en pérdida/métricas.

La ruta directa respeta la API SAM fijada y conserva gradientes en `conv_s0/conv_s1`, aunque esas proyecciones se ejecuten antes de la llamada principal al decoder. Referencias de implementación: [SAM2Base fijado](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/modeling/sam2_base.py) y [predictor fijado](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/sam2_image_predictor.py).

## Reproducir la ejecución en Colab

La ejecución y la reanudación descritas abajo ya fueron realizadas y revisadas. Estas instrucciones se conservan para reproducibilidad; no es necesario repetirlas para cerrar la etapa. Una nueva exportación incorpora la documentación actualizada y tendrá un hash distinto del paquete histórico ejecutado.

1. Exportar `artifacts/auto-prompt-sam2-stage6.zip` con `.venv-validation/Scripts/python.exe scripts/export_stage6_bundle.py`. Copiar SHA impreso. Subir el ZIP a MyDrive.
2. Abrir `notebooks/06_entrenamiento_decoder.ipynb` en Colab GPU. Usar la raíz histórica con TODOS los artefactos de Drive, incluidas features que no viajaron en el ZIP compacto de etapa 5. Configurar SHA y ejecutar entorno y `prepare` una sola vez.
3. `prepare` ejecuta una época por régimen y persiste interrupción controlada. Guardar notebook con salidas, desconectar y eliminar runtime. En otro runtime GPU, ejecutar entorno y `resume` con mismo paquete y configuración. Confirmar que no queda un escritor anterior.
4. Reanudación verifica bit a bit decoder/optimizer/scheduler/AMP/RNG/progreso y repite un paso train dos veces desde el mismo estado guardado. Compara estados exactos y revierte ambos pasos diagnósticos. Continúa desde época 2. Si hay fallo no controlado, se repite solo la época no comprometida desde `last`, nunca se presentan esos pasos perdidos como avance confirmado.
5. Entregar notebook con salidas de ambas sesiones y ambos `regimes/<regime>/stage6/<identity>/evidence.zip`, con sus SHA. Contienen config, entorno, parámetros/hashes, prompts fijos, dependencias, curvas, métricas por chip, recovery_validation, validation e inventario. Adjuntar logs y failure.json en caso de error. Checkpoints grandes permanecen en Drive; compartir referencias persistentes y archivos concretos que solicite el revisor. No compartir tokens ni pesos base.

Persistencia por generaciones completas verificada mediante `ArtifactStore`: `epoch_N`, `last`, `best` contienen decoder, optimizador, scheduler, AMP deshabilitado, RNG Python/NumPy/Torch CPU/CUDA, progreso y muestreador al límite de época, config, hashes y curvas. `best` selecciona IoU agua acumulada val estrictamente superior; empate conserva primera época. Se repara desde `epoch_N` si hubo interrupción entre publicaciones. `decoder_export` es el mejor decoder. No hay cambio silencioso de batch, precisión, resolución ni número de componentes por OOM.

## Validaciones y límites de la evidencia

Las 75 pruebas locales y `pip check` pasaron. Verifican matemática de pérdida con tensores sintéticos sin backward, exclusión de inválidos, unión, aislamiento, corrupción de persistencia, comparación de estados y guardas Colab. El [reporte local histórico](evidence/stage6/local_validation.json) acredita la preparación; la auditoría externa acredita la ejecución posterior en Colab.

La evidencia Colab acredita gradientes finitos y no nulos en el decoder, bloques congelados intactos, pérdida invariante y gradiente cero en posiciones inválidas, interrupción y reanudación, `last`/`best` y curvas completas. Los checkpoints grandes permanecen en Drive con sus inventarios; no se inspeccionaron independientemente sus tensores en la PC.

La cobertura de entrenamiento fue 161/252 chips contribuyentes en `official` y 139/213 en el régimen excluido. Los chips sin componentes conservan predicción vacía y cuentan en métricas; no hubo respaldo asistido. El contador de `prepare` quedó vacío, con salidas originales preservadas. Se mantienen la salvedad histórica de licencia y la revisión académica humana. Asistencia IA: Codex preparó código, pruebas, protocolo, notebook y auditoría; el usuario ejecutó Colab. Etapas 7–9 y demo sin iniciar.

Revisión portátil posterior: `python scripts/review_stage6_evidence.py <evidence.zip> --output <review.json>`. No ejecuta código del adjunto ni deserializa checkpoints. Verifica fuentes, inventario, métricas agregadas y best; no sustituye revisión humana ni recálculo desde máscaras/etiquetas permitidas.

La interpolación bilineal diferenciable de logits se fija en CPU del runtime Colab, con copias autograd GPU→CPU→GPU; decoder y resto del cómputo de modelos siguen en GPU. Esto evita la incompatibilidad conocida de backward bilineal CUDA con determinismo estricto de PyTorch 2.7. Coste de transferencia adicional, sin cambiar resolución ni precisión. No se ejecuta ningún backward en la PC. [Referencia oficial](https://docs.pytorch.org/docs/2.7/generated/torch.use_deterministic_algorithms.html).
