# Etapa 4: disección instrumentada

Fecha: **2026-10-29, America/Lima**. Estado: **Concluida tras revisión Colab prepare/recover**. Cuatro de nueve etapas concluidas. Evidencia en [auditoría](evidence/stage4/colab_run_20261029T162558Z/review.json) e [interpretación con medidas y límites](evidence/stage4/colab_run_20261029T162558Z/interpretacion_y_cierre.md). Las instrucciones siguientes describen cómo reproducir la ejecución ya revisada.

Se consultaron la [hoja de ruta, etapa 4 y §2.4](hoja_de_ruta_fase1.md#7-etapa-4-disección-instrumentada-de-sam-21-y-dinov3), las [bases del curso, A y política IA](../course/tarea_final_deep_learning.md) y el [resumen WAIMLAP, metodología y evaluación](../WAIMLAP/Abstract_CostOfThePrompt.md). Se conserva la salvedad histórica de licencia de Sen1Floods11; no se declara respuesta nueva del publicador.

## Entregables y reutilización

| Archivo | Función |
| --- | --- |
| [Notebook 04](../../notebooks/04_diseccion_modelos.ipynb) | Entorno, ejecución GPU, tablas/figuras, recuperación y exportación. |
| [Configuración](../../configs/profiling.yaml) | Protocolo previo común para ambos regímenes. |
| [features.py](../../src/auto_prompt_sam2/features.py) | DINO congelado revisionado, tokens/parches, captura de features y ERF. |
| [profiling.py](../../src/auto_prompt_sam2/profiling.py) | Propiedad exclusiva de parámetros, contador FLOPs, cobertura, latencia/memoria y limpieza de hooks. |
| [dissection.py](../../src/auto_prompt_sam2/dissection.py) | Selección train, lectura RGB sin etiquetas, ruta tensorial SAM y figuras/arrays. |
| [Runner Colab](../../scripts/run_stage4.py) | `prepare/recover`, identidad, persistencia y evidencia. |
| [Exportador](../../scripts/export_stage4_bundle.py) | ZIP de fuentes, sin pesos, datos ni features reales. |
| [Auditor](../../scripts/review_stage4_evidence.py) | Revisión posterior de ZIP/notebook en CPU sin ejecutar su código. |
| [Pruebas](../../tests/test_stage4.py), [validador local](../../scripts/validate_stage4_local.py) | Contratos sintéticos y guardas de ejecución. |

Se reutilizan `load_frozen_sam`, `require_colab_gpu`, `assert_frozen`, `state_fingerprint`, `baseline.stage2_inputs`, `data.input_rgb`, `preprocessing.prepare/rgb_unit/sam_predictor_image/Coordinates`, `runtime` y `ArtifactStore/Run`. La biblioteca de métricas de etapa 3 se conserva y sigue probándose; no se calcula calidad de segmentación en etapa 4 ni se crea otro evaluador. El predictor oficial de etapa 3 verifica la equivalencia de la ruta tensorial SAM que permite instrumentación/gradientes. No cambian sus pesos, configuración ni máscaras históricas.

No se amplían dependencias: PyTorch/transformers/Pillow y pines de etapas 1–2. No hay entorno de entrenamiento local nuevo. El cargador DINO solo puede descargar/construir después de la guarda Colab+GPU+Drive; usa `trust_remote_code=False`, snapshot local, eval, float32 y cero parámetros entrenables.

## Protocolo prefijado

- Regímenes independientes `official` y `leave_paraguay_out`; **dos primeros IDs lexicográficos de train** por régimen. No se lee ninguna etiqueta. Se valida membresía antes del archivo y su SHA-256 antes de RGB. Sin val, test, Bolivia ni Paraguay reservado; en official, Paraguay solo sería admisible si pertenece a train. Los artefactos ajustados de la otra rama no se usan.
- Manifiestos y estadísticas de etapa 2 verificados por generación/hash; no adquisición, reajuste ni normalizador nuevo. RGB B4/B3/B2, límites train 0.4796/0.4907 conservados por sus artefactos. `prepare` normaliza una sola vez.
- SAM 2.1 tiny, revisión `de431c4043854a71d8101e17995dfe596bf101a5`, fuente `2b90b9f5ceec907a1c18123530e92e794ad901a4`. DINOv3 ViT-L/16 SAT-493M, revisión **autorizada** `f692fa42da72c6797b67cd73494a168d1120d3ee`, `model.safetensors`. Se conservan en Drive y se registran sus SHA-256 y hashes de estado completo. No se redistribuyen pesos.
- SAM 1×3×1024×1024; DINO 1×3×512×512, patch 16, grid 32×32. Float32, sin AMP/TF32, semilla 42 y determinismo de etapa 1. Atención DINO SDPA explícita. OOM, acceso denegado o kernel incompatible se registran como fallo; no se sustituyen modelo, tamaño, precisión o lugar de ejecución.
- **Puntos diagnósticos espaciales fijos** nativos `(256,256)` con signo + y `(32,32)` con signo −, dos puntos, una llamada multimáscara. Se transforman con `Coordinates` a SAM y se precargan en GPU antes de cronometrar. No proceden de agua real ni de un generador automático y no son una comparación de calidad. Se retornan los tres logits nativos y scores; no se usa ground truth, selección oráculo ni unión por componentes.

## Qué se instrumenta

**Parámetros:** objetos únicos, una única atribución por parámetro. SAM: image encoder, prompt encoder, mask decoder, memory attention, memory encoder y proyecciones de objetos presentes; resto en residual con nombres explícitos, incluidos tokens a nivel raíz. DINO: embeddings, RoPE, cada `layer.0`–`layer.23`, norma final y residual. La suma debe igualar `model.parameters()` y todos los entrenables ser cero. RoPE puede carecer de parámetros aunque tenga operaciones.

**FLOPs:** `torch.utils.flop_counter.FlopCounterMode` de PyTorch 2.7.1, solo forward sobre primer train por régimen, propietario activo exclusivo y residual fuera de los bloques registrados. Se exige que su suma coincida con el global del contador. Convención 2 FLOPs por multiply-add (no confundir con 1 MAC). Se persisten llamadas por operación, bloque y presencia de fórmula **después de la descomposición** que realiza PyTorch. Las operaciones sin fórmula no reciben costo real cero: el total es **parcial**, nunca una medida exacta de todo el trabajo. Sesgos, normalización, activaciones, softmax, RoPE, interpolación, transferencias y Python pueden estar ausentes o parcialmente representados; la lista medida es la evidencia. Las copias/vistas también aparecen como operaciones sin fórmula aunque no sean aritmética.

La ruta estática no ejecuta memory encoder/attention ni proyecciones de objetos: se conservan sus parámetros y FLOPs/latencia `null`. `forward_image` llama `conv_s0/s1` del decoder antes de su `__call__`; sus operaciones aparecen en el **residual de ejecución**, aunque sus pesos pertenezcan al decoder. Esa distinción está declarada y no se duplican sumas. El replay del decoder recibe esas features ya proyectadas; la ruta completa sí incluye las proyecciones.

**Features:** Hiera primer/medio/último bloque, nombres concretos y forma observada BHWC, transposición explícita a BCHW; no tiene CLS/registers. DINO capas 0/11/23 antes de norma final y `last_hidden_state` después de ella, con forma BNC real. Orden: CLS índice 0; registros `[1,1+R)`; patches desde `1+R`, sin incluir especiales en el grid. Se valida N=1+R+32² y C=1024; reshape por filas a B×32×32×1024 y permutación BCHW. El número de registros se lee del config autorizado y se persiste. El centro de parche nativo a tamaño 512 es `(16*col+7.5,16*fila+7.5)`; `Coordinates` documenta el mapeo e inversa. Se registran también shapes de entrada/salida de todos los bloques ejecutados.

Visualización de canales fijos 0/1/2, sin PCA, scaler ni otra proyección aprendida. Rangos min/max del panel solo para color, nunca para la extracción; los arrays crudos quedan guardados. No implica que los canales tengan semántica de agua.

**Campo receptivo efectivo:** escalar canal 0 de posición espacial central en `forward_image(...)[vision_features]` de SAM y último grid normalizado DINO. `autograd.grad` solo respecto al tensor de entrada normalizado; valor absoluto, suma RGB y masa unitaria. No `backward()` ni optimizador. Se guardan objetivo B/C/y/x, shapes, escalar, mapa, masa fuera de mitad central y overlay sobre RGB train. Un gradiente nulo/no finito bloquea el paso. La visualización logarítmica mejora legibilidad, no cambia el array. Es sensibilidad local para ese chip/canal/objetivo, no campo teórico, explicación causal global ni atención uniforme. El mapa respecto a entrada normalizada no es directamente el gradiente respecto a reflectancia original.

**Latencia y memoria:** primer train de cada régimen, batch 1, **5 warmups y 20 repeticiones**, mediana de pared con `cuda.synchronize()` antes y después de cada repetición. Se guardan muestras, min/max, precisión, tamaños, puntos, GPU/VRAM/driver/CUDA y versiones. `reset_peak_memory_stats()` después del calentamiento; picos allocated/reserved absolutos, baseline allocated/reserved y delta allocated. El máximo reserved puede conservar caché del warmup; no equivale a memoria incremental del forward. Un modelo reside a la vez y se libera caché solo entre modelos.

La ruta completa se cronometra **sin hooks**. Excluye carga/descarga de pesos, TIFF/IO, transforms CPU, H2D/D2H, features/hooks, gradientes, figuras y persistencia. Incluye en SAM encoder, proyecciones, reshape/no_mem, prompts/PE, decoder y resize de los tres logits a 512; en DINO embeddings/RoPE, 24 capas y norma final. Los bloques se reproducen aisladamente con entradas de la primera llamada precargadas, sin hooks/transferencias durante el reloj; sus tiempos **no son aditivos** ni sustituyen el tiempo completo. No se mide el método automático de etapas posteriores. Hardware puede variar entre sesiones; la recuperación verifica las medidas guardadas y **no las reemplaza por nuevos tiempos**.

## Ejecución del usuario en Colab

1. En la PC, ejecutar `python scripts/export_stage4_bundle.py`. Subir `artifacts/auto-prompt-sam2-stage4.zip` a Drive y copiar el SHA-256 mostrado al notebook 04. No reutilizar el ZIP de otra etapa ni editar fuentes/configuración entre prepare y recover.
2. Abrir notebook 04 con GPU. Configurar `PERSIST_ROOT` con los artefactos históricos y el secreto `HF_TOKEN` autorizado para DINO si fuera necesario. Ejecutar entorno, `prepare` y visualización; guardar notebook con todas las salidas.
3. Desconectar/eliminar el runtime. Abrir GPU nueva; ejecutar **solo entorno y `recover`**, y la celda final de entrega. No ejecutar de nuevo prepare ni borrar sus salidas previas.
4. Compartir notebook ejecutado, ZIP `stage4/evidence/<run_id>.zip` y sus JSON `validation`, `recovery_validation`, `config`, `environment`. Logs en `stage4/notebook_logs/`; ante fallo, compartir salida y ZIP parcial con `failure.json`. No incluir tokens ni pesos.

Se usa la persistencia inmutable `stage4/environment/runs/<run_id>/<artefacto>/<generación>/{payload,COMPLETE.json}`. Identidad: fuentes, configuración, revisiones, hashes de manifiestos/estadísticas, generaciones e IDs train por régimen. `config` conserva versiones y snapshot de código; `validation` enlaza CSV/PNG/NPZ/JSON por generación/hash. Las copias con extensión legible están en `stage4/evidence/<run_id>/`. Solo se publica `complete` tras verificar todos los pasos; una interrupción mantiene generaciones anteriores. `prepare` crea otro run al reintentar: no sobrescribe un resultado parcial. No se promete reanudar cómputo a mitad de una medición.

`recover` exige un boot ID nuevo, verifica todos los hashes, carga pesos offline y compara features/ERF de ambos chips de cada rama con **atol=1e-5 y rtol=1e-4**, fijados antes de ejecutar. Comprueba nuevamente suma de parámetros, cero gradientes acumulados, eval, hooks eliminados y hashes completos de parámetros/buffers intactos. No se cambian tolerancias para forzar aceptación; incompatibilidades entre GPU se revisan como tales. Las figuras/tablas originales se recuperan sin regenerarlas ni retocarlas.

## Evidencia local y revisión Colab

Comando permitido en PC: `.venv-validation/Scripts/python.exe scripts/validate_stage4_local.py`. [Informe local](evidence/stage4/local_validation.json): sintaxis de fuentes/celdas limpia, **50 pruebas CPU sintéticas y pip check correctos**. Casos de tokens con especiales y orden por filas, prohibición de reservados antes de leer, ownership con alias/residual, hooks tras excepciones, ERF sintético de masa calculable, mediana y persistencia PNG/NPZ. Incluye una comprobación del contador sobre matrices sintéticas CPU 2×3 @ 3×4 = 48 FLOPs, sin red neuronal; la guarda Colab se reemplaza exclusivamente en esa prueba aritmética. No prueba kernels GPU, ejecución real de modelos, equivalencia SAM, FLOPs de modelos ni tiempos GPU. No se descargaron pesos ni se extrajeron features reales en esta PC. Posteriormente se revisaron en CPU los artefactos reales aportados desde Colab, sin ejecutar modelos.

Auditoría posterior reproducible (CPU, solo artefactos; no ejecuta código adjunto):

```powershell
python scripts/review_stage4_evidence.py RUTA_RUN.zip RUTA_NOTEBOOK.ipynb --root artifacts/stage2-persistent --output docs/project/evidence/stage4/colab_run_ID
```

El auditor archiva originales, verifica snapshots/generaciones, linaje de etapa 2, roles/IDs, matrices finitas y masa ERF, inventario, recuperación, sumas y medianas; deja explícitamente pendiente la lectura visual/científica. **Codex debe inspeccionar PNG, notebook y tablas reales**, contrastar H4a–c, documentar interpretación y límites y actualizar cuaderno/estado antes de marcar Concluida. No ejecutar las celdas del adjunto como parte de su revisión.

**Revisión Colab completada 2026-10-29:** notebook y ZIP originales del usuario verificados; 49 generaciones, 39 artefactos y fuentes coincidentes. T4 en dos boot IDs, 50 tests/pip check en ambas sesiones, revisión de las ocho figuras/arrays y contraste de H4a–c. Pesos, hooks, sumas y recuperación correctos. [Informe completo](evidence/stage4/colab_run_20261029T162558Z/interpretacion_y_cierre.md). Sin validación técnica pendiente. Mantener salvedad de licencia y comprensión/revisión académica humana. Sin etapas 5–9 ni demo.

## Referencias de implementación

Se contrastó la API con fuentes primarias fijadas: [DINOv3 en transformers 4.56.2](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dinov3_vit/modeling_dinov3_vit.py), [predictor SAM revisionado](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/sam2_image_predictor.py), [SAM2Base revisionado](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/modeling/sam2_base.py) y [contador PyTorch 2.7.1](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/utils/flop_counter.py). Estas lecturas no son una ejecución experimental. IA: Codex preparó código, protocolo, pruebas y documentación; sin subagentes, resultados inventados ni comunicaciones externas. El equipo debe revisar y comprender las decisiones.
