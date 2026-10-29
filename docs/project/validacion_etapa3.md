# Etapa 3: referencia asistida y evaluador común

**Fecha:** 2026-10-29 (America/Lima). **Estado: Concluida.** Implementación, pruebas CPU y ejecución/recuperación GPU en Colab revisadas con los archivos del usuario. Véase la auditoría al final de este documento. Se conservan el cierre de etapas 1–2 y la consulta de licencia pendiente.

## Entregables y alcance

- `notebooks/03_linea_base_asistida.ipynb`: ejecución del usuario en Colab GPU, persistencia en el mismo Drive y recuperación offline en runtime nuevo.
- `configs/baseline.yaml`: política prefijada y compartida para ambos regímenes; hashes y parámetros de etapa 2 independientes.
- `reference_prompts.py`: único generador que ve etiquetas; `sam_adapter.py`: imagen + prompts espaciales, sin lector de etiquetas ni IoU real. `metrics.py`: evaluador común. `baseline.py`: integración, acceso a roles permitidos y persistencia.
- `scripts/run_stage3.py`: modos `prepare`, `resume`, `recover`, snapshot de código/configuración, control Colab/GPU, congelamiento y evidencia exportable. `export_stage3_bundle.py`: transporte de fuentes sin pesos ni GeoTIFF. `validate_stage3_local.py` y `tests/test_stage3.py`: comprobaciones ligeras.

No se implementan etapas 4–9, DINOv3, funciones de pérdida, entrenamiento, selección experimental, evaluación final ni demo. `-1` no aporta a ninguna confusión ni métrica; no hay pérdidas que calcular en esta etapa congelada.

## Decisiones fijadas antes de resultados

Referencia asistida **no desplegable**, tampoco cota matemática garantizada. Se conocen desde etiquetas tanto componentes como ausencia de agua; esta ventaja se informa en el notebook y en cada registro de prompts.

Conectividad 8, todos los componentes sin filtro de tamaño, orden por primer píxel en fila/columna. Por llamada: hasta un positivo `label==1` y un negativo `label==0`, excluyendo `-1` y toda el agua, incluso de otros componentes. Distancia mínima 8 px entre todos los puntos de la llamada; puede reutilizarse un negativo entre llamadas. Se permutan candidatos sin reemplazo con PCG64 y semilla de chip derivada por SHA-256 de semilla global 42, régimen e ID. No se relaja distancia ni se eliminan componentes: se registra presupuesto solicitado, puntos efectivos y faltantes. Estas elecciones deben alinearse con los presupuestos y geometría automáticos en la etapa que les corresponda; no se ha hecho una ablación A2.

SAM 2.1 tiny, revisión de pesos `de431c4043854a71d8101e17995dfe596bf101a5`, archivo `sam2.1_hiera_tiny.pt`, fuente SAM `2b90b9f5ceec907a1c18123530e92e794ad901a4`, configuración `configs/sam2.1/sam2.1_hiera_t.yaml`: se reutilizan las referencias de `configs/environment.yaml`. El SHA-256 de los pesos se medirá **en Colab** y quedará fijado para la recarga. Se usa `build_sam2` con archivo/revisión explícitos: el helper HF oficial no permite fijar la revisión de descarga en su llamada interna. Se desactiva posprocesamiento adicional, se usa float32, eval, sin autocast ni gradientes y se congelan todos los parámetros. Se registra el hash de parámetros y buffers antes/después y tras reconstruir el modelo. No se han descargado pesos en la PC.

`sam_predictor_image` de etapa 2 produce HWC float32 [0,1]; el predictor aplica una vez resize/normalización. Puntos `(x,y)` en la malla nativa; el wrapper los transforma a 1024 y devuelve máscaras en 512². Una llamada multimáscara por componente, máximo de IoU **predicha**, primer índice en empates, unión binaria sin recorte por etiquetas. Si no hay agua válida, máscara vacía sin llamar a SAM; si todo es inválido, además no se aportan confusiones.

Fuentes primarias revisadas: [predictor oficial en la revisión fijada](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/sam2_image_predictor.py) y [constructor oficial](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/build_sam.py). Las pruebas comparan las coordenadas y reutilizan la igualdad de normalización con `SAM2Transforms`, sin construir un modelo.

## Datos, métricas y persistencia

No se reconstruyen manifiestos ni se ajusta radiometría. `stage2_inputs` verifica marcadores y hashes de adquisición, informe, manifiestos y parámetros, y carga **las generaciones indicadas por etapa 2**. La ausencia de una generación compatible produce error. `load_preprocessing` verifica que las fuentes sigan siendo solo train del régimen; se mantienen 0.4796 official y 0.4907 excluido. El pipeline admite únicamente ambos regímenes fijados, un chip train (primero por ID) y toda val (89/75); train y val se reportan separados. Los checks de pertenencia preceden toda lectura de imagen/etiqueta. No se lee contenido reservado ni se cambian splits.

IoU de agua acumulada: sumar TP/FP/FN/TN válidos antes de dividir. Media por evento: acumular sus píxeles y promediar eventos con igual peso, informando los indefinidos excluidos. Por chip: IoU individual solo si hay agua real válida, mediana, Q25/Q75 e IQR con cuantiles lineales. Precisión, recall, Dice y mIoU de agua/no agua usan los mismos acumuladores. Cocientes con denominador cero son `null` con motivo; mIoU promedia solo clases definidas y enumera cuáles participaron. No se imputa silenciosamente IoU perfecta a agua ausente.

Chips sin agua permanecen en todos los agregados y se reportan sus FP en píxeles/m²; solo su IoU individual queda excluida. Chips completamente inválidos aportan cero y se cuentan aparte. El área se calcula reproyectando bordes a EPSG:6933 (WGS84, área equivalente), por fila para los rasters EPSG:4326 sin rotación auditados. No se interpreta affine en grados como metros ni se sustituye por 100 m² nominales. CRS/geometrías fuera de ese contrato producen error.

Cada run conserva `ArtifactStore` con identidad de configuración, fuentes, modelo, manifiestos, parámetros y sus generaciones. Cada chip guarda NPZ sin pickle: máscara binaria 512² y JSON con prompts, semillas, conectividad, presupuesto, faltantes, scores candidatos, índice seleccionado, número de llamadas, geometría, confusión y métricas. Los índices conservan generación y SHA-256; hay resúmenes por rol/evento/conjunto. `source_snapshot` conserva también cambios aún sin commit.

Ubicaciones en Drive:

```text
stage3/environment/runs/<run_id>/  # estado, config, fuente, entorno, modelo, pruebas, informes
regimes/<regime>/runs/<run_id>/assisted/  # predicciones y resúmenes inmutables
stage3/evidence/<run_id>/         # JSON cómodos para revisión
stage3/evidence/<run_id>.zip      # generaciones, marcadores, fuente y predicciones; sin pesos/raw
stage3/notebook_logs/            # stdout/stderr del notebook
base_models/huggingface/          # checkpoint persistente, solo adquirido en Colab
```

`prepare` crea un run. `resume` reutiliza el último compatible después de detener el escritor anterior, recupera chips completos y repite solo trabajo que falta; no hay recuperación a mitad de componente. `recover` requiere run completo, no regenera predicciones perdidas: verifica todas, reconstruye resúmenes y carga SAM offline. `complete` es el estado del cálculo persistente, **no el cierre de etapa**, que permanece pendiente de revisión externa.

## Validación local realizada

Comando: `.venv-validation/Scripts/python.exe scripts/validate_stage3_local.py` (entorno CPU existente, sin instalar modelos). [Informe íntegro](evidence/stage3/local_validation.json): sintaxis de fuentes/celdas correcta, notebook limpio sin ejecución, `pip check` correcto y **40 tests correctos** (21 previos + 19 de etapa 3). Solo arrays, rasters sintéticos, transforms sin modelo y stubs de funciones; sin pesos, llamadas a SAM, datos reales de desarrollo ni contenido reservado. Las métricas sintéticas no se presentan como resultados de segmentación.

Cobertura: casos perfectos/error total, TP/FP/FN/TN calculables, `-1`, todas las divisiones nulas, clases ausentes, chips sin agua con FP, completamente inválidos, media por evento distinta del promedio por chip, IQR, área física, geometría y puntos extremos, conectividad 4/8 y puentes inválidos, semillas y faltantes, selección por score predicho distinta de selección oráculo, empates, scores no finitos, unión, identidad entre regímenes, recuperación NPZ, cambios en buffers y rechazo de parámetros no congelados, denegación de reservados antes de leer y reutilización de generaciones de etapa 2 sin ajuste.

## Protocolo de ejecución y entrega de evidencia

1. En la PC: `python scripts/export_stage3_bundle.py`. Subir `artifacts/auto-prompt-sam2-stage3.zip` a Drive y abrir `notebooks/03_linea_base_asistida.ipynb` en Colab con GPU. Copiar el SHA-256 mostrado; indicar la raíz Drive de etapa 2.
2. Ejecutar secciones 1–3. `prepare` solo procesa un train y toda val por régimen. Mantener la configuración prefijada común; no abrir reservados ni adaptar la segunda rama con los resultados de la primera. Si se interrumpe, conservar logs y usar `resume` con el mismo ZIP; no dos escritores.
3. Guardar salidas, **desconectar y eliminar runtime**, conectar otro con GPU y ejecutar sección 1 y sección 4 (`recover`) con el mismo ZIP y Drive. Se requiere GPU para la recarga/repetición; no sustituir por PC ni CPU. Hash exacto de pesos/estado, puntos idénticos, cero píxeles distintos, scores con `atol=rtol=1e-5` e igual índice seleccionado. GPU/hardware, PID y boot ID quedan registrados. Cualquier divergencia se investiga y no se cambia la tolerancia a posteriori.
4. Compartir notebook ejecutado con salidas/orden/errores, ZIP impreso por recover y `validation.json` + `recovery_validation.json`. Ante fallo compartir log y ZIP de fallo; no hace falta repetir todo antes de revisar. No compartir tokens ni pesos.
5. El asistente debe conservar originales y hashes y contrastar fuentes/configuración, hardware, accesos, conteos, puntos, scores, métricas, congelamiento, integridad y recarga. Comprobar que las 89/75 validaciones y la repetición con prompts reales están presentes. El boot ID permite distinguir nuevo proceso de runtime; no deducir reinicio solo de un PID distinto. Marcar la etapa concluida únicamente tras esa revisión.

## Revisión de la ejecución Colab recibida: cierre

El usuario aportó el ZIP `20261029T134519Z_1894f1a3ae_ec3f3bcc.zip` y `03_linea_base_asistida_ejecutado_colab.ipynb`. Se conservan originales sin modificaciones, JSON y [revisión externa](evidence/stage3/colab_run_20261029T134519Z/review.json) en [evidencia](evidence/stage3/colab_run_20261029T134519Z/). La auditoría CPU reproducible está en `scripts/review_stage3_evidence.py`; solo lee documentos/arrays y etiquetas de desarrollo, sin ejecutar código adjunto ni modelos.

Se verificaron CRC del ZIP, SHA-256/tamaño/identidad de **182 generaciones**, snapshot de **33 archivos idénticos** al código entregado y configuración intacta. El único cambio de fuente del notebook ejecutado es completar el SHA-256 del paquete original; no hay errores de celdas ni literales con forma de token HF. Las salidas de prepare y recover están presentes (contadores reiniciados); boot IDs diferentes confirman runtimes distintos. Ambas sesiones registran Tesla T4, CUDA disponible, 50 pines comprobados, pip check y **40 tests correctos**.

Los manifiestos, parámetros y generaciones coinciden exactamente con la evidencia aceptada de etapa 2. Se auditaron **90 predicciones official (1 train + 89 val)** y **76 excluidas (1 train + 75 val)**. Pertenencias, accesos, puntos 1/0 válidos, semillas, presupuesto, distancia y máximos de score correctos; no test/Bolivia/Paraguay reservado. Se recalcularon todas las confusiones y métricas por chip desde máscaras guardadas y etiquetas de desarrollo cuyos hashes coinciden con los manifiestos. Se reconstruyeron agregados iguales y se verificó área FP por georreferenciación (tolerancia de auditoría de área, ajena a inferencia: atol 1e-5 m², rtol 1e-10). No se repitió inferencia en la PC. La regeneración local de puntos del chip de repetición coincide en ambos regímenes.

SAM: **38 962 498 parámetros; 0 entrenables**, todos los módulos eval, float32 sin autocast. Checkpoint SHA-256 `7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69`. Hash de estado `72a85430eb2c93ee9fe3fed9a2ed1e097a8756f15c0becb212586afe7112384d` idéntico antes/después de inferencia y tras recarga offline. Los informes confirman puntos idénticos, cero píxeles discrepantes y scores dentro de tolerancia en `Ghana_103272` por régimen. Todos los índices, máscaras y resúmenes se recuperaron sin cambios. Esta conclusión se apoya en artefactos íntegros y fuente inspeccionada; no es una ejecución GPU independiente del auditor. No se conservaron las máscaras candidatas por componente: se contrastó la selección con scores persistidos y la implementación de unión con sus pruebas sintéticas.

Resultados **de validación**, no evaluación final; valores entre 0 y 1:

| Métrica | official | leave_paraguay_out |
| --- | ---: | ---: |
| IoU agua acumulada | 0.158044 | 0.168434 |
| Media por evento | 0.171922 | 0.193338 |
| Mediana por chip | 0.068207 | 0.080098 |
| Q25–Q75 por chip | 0.019265–0.202591 | 0.017478–0.269739 |
| IQR | 0.183326 | 0.252261 |
| Precisión agua | 0.161073 | 0.169336 |
| Recall agua | 0.893660 | 0.969341 |
| Dice | 0.272950 | 0.288307 |
| mIoU | 0.287881 | 0.284738 |

official incluye 77 chips en IoU individual, excluye 9 sin agua y 3 totalmente inválidos; excluido incluye 66, excluye 6 sin agua y 3 inválidos. Los chips sin agua tienen FP cero por la política asistida de máscara vacía, no por haber demostrado detección automática de ausencia de agua. La precisión baja y recall alto indican sobresegmentación considerable. No se altera configuración para elevar la IoU y no se interpreta la diferencia entre regímenes como mejora causal: difieren las poblaciones de validación, radiometría y semillas derivadas. No hay umbral de IoU exigido para cerrar esta reproducción técnica.

**Cierre:** etapa 3 Concluida, 3/9; etapas 4–9 sin iniciar. No queda ejecución técnica requerida pendiente de esta etapa. Permanecen consulta al publicador sobre licencia y revisión académica humana; comparaciones/ablaciones/análisis de fallos pertenecen a las etapas posteriores y no se ejecutan. Los JSON originales conservan `closure=pending_external_review...`: no se reescriben; esta revisión externa acredita el cierre. IA: Codex auditó documentos, hashes, arrays, puntos y métricas y actualizó el seguimiento, sin subagentes ni modelos locales. [Cuaderno](cuaderno_decisiones.md).
