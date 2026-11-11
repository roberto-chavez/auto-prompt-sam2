# Etapa 5: probe y prompts automáticos

Fecha: 2026-10-29, America/Lima. **En curso: implementación preparada; falta ejecución Colab del usuario y revisión de su evidencia.** Las etapas 1–4 están cerradas y se reutilizan sin recalcular sus splits, radiometría o revisiones. Este documento no acredita entrenamiento ni resultados GPU.

## Alcance y decisiones previas

Se leyó el [resumen WAIMLAP](../WAIMLAP/Abstract_CostOfThePrompt.md), especialmente metodología y evaluación, y la [hoja de ruta](hoja_de_ruta_fase1.md), sección 2.4 y etapa 5. El alcance es train/val: official 252/89 y leave_paraguay_out 213/75. Nunca se abren test, Bolivia ni Paraguay reservado. Se comparte código, pesos originales y búsqueda prefijada; cada régimen conserva sus propios parámetros de etapa 2, features, scaler, probe y selección. El registro de un régimen no modifica la búsqueda del otro.

Configuraciones JSON compatibles con YAML: [probe](../../configs/probe.yaml) y [prompts](../../configs/prompts.yaml). Recomendaciones adoptadas antes de observar resultados de etapa 5:

- DINOv3 congelado, salida `last_hidden_state` después de LayerNorm; se eliminan CLS y registros mediante `features.token_layout/tokens_to_grid`. Shards float32 Hpatch×Wpatch×C = 32×32×1024, un chip por shard. RGB y normalización de etapa 2, 512×512, parches de 16×16, sin aumentos, padding ni muestreo aleatorio.
- Targets manuales sobre el total de píxeles **válidos de la etiqueta** por parche: agua ≥0.50, no agua ≤0.05. Cero válidos o fracción intermedia: -1, ignorado. Mínimo válido 1, sin otro filtro. Todos los parches train inequívocos, sin submuestreo. Se conserva balance por rol e IDs/hashes. Los parches val son diagnósticos y no entran al optimizador.
- Scaler de media/desviación poblacional por feature, ajustado solo con esos parches train; desviación cero → 1. Regresión logística binaria CUDA, intercepto sin regularizar, inicio cero, LBFGS con strong Wolfe, 150 iteraciones máximas, historial 20. Objetivo: media de BCE ponderada + `sum(w²)/(2*C*N)`. `C` = 0.1/1; ponderación ninguna/balanceada `N/(2*N_clase)`. Cuatro probes por régimen. No se añade sklearn ni se altera el entorno fijado.
- Convergencia registrada: norma infinita de gradiente ≤1e-5 o parada LBFGS antes del máximo (tolerancia de cambio 1e-9). Se guardan iteraciones, objetivo, gradiente y estado. Un candidato que agota el máximo sin el criterio de gradiente queda persistido, evaluado y excluido de selección. Si ninguno es elegible, la ejecución falla y se conserva el diagnóstico; no amplía la búsqueda automáticamente.
- Dos umbrales positivos 0.5/0.7 por probe: ocho filas, seleccionadas únicamente por IoU agua de segmentación acumulada en toda validación. Orden de desempate: C ascendente, ponderación ninguna antes que balanceada, umbral 0.5 antes que 0.7; primer máximo exacto. Accuracy y log-loss de parches val son auxiliares. Esta calibración prepara la etapa 5 y no realiza A1–A3 ni genera `evaluation_lock.json` de etapa 7.
- **NDWI aplazado, no incluido.** Decisión registrada antes de ejecutar esta búsqueda, no condicionada al rendimiento. El flujo actual es un probe supervisado con etiquetas train y calibración manual val. No se denomina libre de etiquetas. Una extensión futura exige presupuesto y configuración separados antes de etapa 7; si se usa calibración manual o decoder supervisado conserva esas dependencias.

## Contrato de inferencia

`probe.probabilities` acepta features y probe portable, verificando régimen. `automatic_prompts` y `mask_selection` no reciben etiquetas ni su validez. El lector de imagen obtiene exclusivamente máscara/nodata/valores finitos de la entrada. El runner deshabilita los lectores comunes de etiquetas durante cada llamada automática; el evaluador las lee después de retornar para calcular métricas comunes. Los targets manuales no se usan para enmascarar features o probabilidades de inferencia.

La interpolación es bilineal en centros de píxel, `align_corners=False`, replicando bordes: coordenada de parche `(x+.5)/16-.5`. Los centros nativos de parches extremos son (7.5,7.5) y (503.5,503.5). El mapa nativo es 512×512. Componentes sobre `probabilidad >= umbral_positivo` y validez de **imagen**, conectividad 8 y mínimo 16 píxeles nativos. Orden por primer píxel en filas; los IDs pueden contener saltos por descarte de componentes pequeños. Sin apertura/cierre morfológico ni eliminación basada en etiquetas.

Por componente: hasta 3 positivos, elegidos como máximo global restante tras suprimir por distancia euclídea ≥16 píxeles respecto de todos los puntos ya elegidos en esa llamada; mesetas y bordes participan. No se exige un máximo local estricto. Empates: probabilidad descendente, fila y columna ascendentes. Hasta 3 negativos en toda la imagen válida con probabilidad ≤0.1, orden inverso de probabilidad y mismo desempate/espaciado respecto a positivos y negativos. No se impone separación entre llamadas de componentes diferentes. Si faltan candidatos se usan los disponibles y se registra déficit, sin reemplazo, relajación o consulta de etiquetas. No hay componentes → máscara vacía sin llamar SAM.

Se reutilizan `sam_adapter.segment_components/select_multimask`: un embedding de imagen por chip con componentes, una llamada multimáscara por componente, mayor IoU **predicha** y primer índice en empate, unión binaria nativa con logit >0, recortada solo por validez de entrada. No se entrena SAM ni su decoder.

La rama malla usa `SAM2AutomaticMaskGenerator` del código fijado de SAM: 32 puntos por eje (1024 posiciones), centros normalizados `(i+.5)/32` que el generador oficial escala por ancho/alto; batch 64, un crop completo, multimáscara, sin m2m. IoU predicha >0.8, estabilidad ≥0.95 con desplazamiento de logits 1.0, NMS de cajas 0.7, sin limpieza de regiones pequeñas ni relleno de huecos. Los filtros y NMS oficiales ocurren antes del scoring semántico. Su orden/desempates NMS depende del código/dispositivo fijados y queda reflejado en candidatos persistidos; no se afirma reproducibilidad binaria de NMS entre GPU distintas. No se usa IoU real. Se persisten las máscaras agnósticas que sobreviven, IoU predicha y estabilidad. Una máscara se acepta como agua si su probabilidad DINO media sobre máscara ∩ validez de imagen ≥0.5/0.7; se elige ese umbral por IoU acumulada val, primer umbral en empate, con probe fijado por la rama puntos. Máscaras vacías o sin píxeles válidos de entrada se rechazan. Salida: unión de regiones aceptadas ∩ validez de entrada. Los dos scores de calibración permanecen separados y no sustituyen la campaña de ablaciones.

## Ejecutar exclusivamente en Colab

En la PC se puede ejecutar el validador ligero y construir el transporte de fuentes:

```powershell
.venv-validation/Scripts/python.exe scripts/validate_stage5_local.py
.venv-validation/Scripts/python.exe scripts/export_stage5_bundle.py
```

Subir `artifacts/auto-prompt-sam2-stage5.zip` a MyDrive y abrir [05_prompts_automaticos.ipynb](../../notebooks/05_prompts_automaticos.ipynb) en Colab con GPU. Copiar el SHA-256 emitido por el exportador y usar la raíz Drive histórica que contiene `raw/`, `stage2/`, `regimes/` y pesos base en caché. El notebook comprueba ZIP, monta Drive, reutiliza bootstrap/venv y pines de etapas anteriores; token opcional `HF_TOKEN` en Secrets, nunca impreso. Se ejecutan primero configuración de entorno, `prepare` y resumen. La ausencia de GPU/Drive/contexto Colab bloquea antes de entrenar o cargar modelos.

Estimación de almacenamiento por shapes, no medición: 629 shards ×4 MiB de features float32 ≈2.46 GiB sin comprimir, más máscaras/probabilidades/candidatos. Durante la carga train se necesita RAM para concatenación; DINO se libera antes de ajustar probes y SAM se carga después. Un OOM se registra como fallo; no cambia tamaños, batch, modelo o búsqueda silenciosamente. Para continuar un run interrumpido, detener primero el proceso/sesión anterior y seleccionar explícitamente `resume` en la misma celda, con idénticas fuentes/config/raíz. Cada unidad completa se reutiliza por identidad/hash; unidades pendientes se recalculan. No hay estado de optimizador a mitad de LBFGS: se reinicia solo el probe incompleto, sin cambiar su inicio/config.

Después de guardar el notebook con salidas: desconectar/eliminar runtime, crear otro con GPU y ejecutar **solo** entorno + `recover` + entrega. `recover` exige boot ID distinto, pesos offline desde caché, verifica todas las generaciones de features/targets/probes/predicciones y repite el primer val de cada régimen. No reentrena ni recalibra: contrasta features con atol=1e-6/rtol=1e-5, probabilidades con las mismas tolerancias, puntos exactos, máscara exacta y scores SAM con atol=1e-6/rtol=1e-5. La malla se reconstruye desde candidatos agnósticos persistidos; no vuelve a ejecutar AMG/NMS. No se exige portabilidad bit a bit entre GPU distintas para nuevas generaciones de AMG. Se guardan fingerprints de parámetros/buffers antes y después en ambos modelos.

## Evidencia y revisión pendiente

`stage5/environment/runs/<run_id>/` en Drive conserva generaciones completas, hashes, fuente ZIP, entorno, linaje por régimen, features/validez de entrada, targets, pesos/scaler/config de cada probe, curva final de convergencia, bundles portables, ocho filas de puntos y dos de malla por régimen, mapas, componentes, puntos/presupuestos, scores, predicciones y métricas por chip/evento/conjunto. No se exportan pesos base ni datos crudos en el transporte de revisión.

Compartir:

1. Notebook ejecutado con salidas de ambas sesiones y errores preservados.
2. `stage5/evidence/<run_id>.zip` y `validation.json`, `recovery_validation.json`, `config.json`, `environment.json`.
3. Logs de `stage5/notebook_logs/` y `failure.json` si hubo fallo.

El ZIP compacto incluye todos los probes/targets/predicciones/candidatos, todos los marcadores y solo el primer shard val por régimen; los demás payloads de features quedan en Drive para no trasladar varios GiB. La revisión independiente podrá verificar sus marcadores y la verificación completa reportada en Colab, pero **no** recalcular sus hashes ni scaler desde todos esos arrays omitidos. Si hay discrepancias se solicitarán los shards concretos, sin presentar esa limitación como verificación completa externa.

El revisor CPU `scripts/review_stage5_evidence.py --evidence <zip> --notebook <ipynb> --root <raíz_etapa2> --output <review.json>` no ejecuta código del ZIP ni modelos. Verifica generaciones incluidas, fuentes, splits/linaje, todos los agregados/confusiones de predicciones contra etiquetas **val** permitidas, selección prefijada y semántica de malla; reconstruye probabilidades/puntos del primer val desde disco. El estado automatizado no sustituye la inspección de las salidas, interpretación, decisión humana y actualización de estado/cuaderno. Para cerrar faltan esa evidencia real y su revisión. No hay resultados de probe o SAM automáticos acreditados a esta fecha.

## Verificación local y límites

La evidencia local se escribe en [local_validation.json](evidence/stage5/local_validation.json); el inventario de transporte en [bundle_validation.json](evidence/stage5/bundle_validation.json). Pasaron 66 pruebas sintéticas CPU (16 de etapa 5 y 50 previas), pip check, sintaxis de fuentes/celdas e integridad del ZIP de fuentes. Verifican targets, geometría, conectividad, separación/cantidad/empates, déficits, ausencia de componentes, multimáscara, malla, aislamiento, reconstrucción portable y barreras Colab. Nunca ajustan un probe ni construyen modelos, ni siquiera pequeños. El ajuste real LBFGS, convergencia, memoria y comportamiento con datos reales quedan pendientes de Colab.

Asistencia de IA: Codex preparó código, pruebas, configuración, notebook y documentación. No se han generado resultados GPU ni conclusiones de calidad mediante IA. Se mantiene la consulta pendiente sobre licencia del dataset. Detenerse en esta etapa; etapas 6–9, decoder, campaña final y demo quedan sin iniciar.
