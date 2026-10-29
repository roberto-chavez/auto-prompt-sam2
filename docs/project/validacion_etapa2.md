# Etapa 2: datos y preprocesamiento

Fecha: **2026-10-28 (America/Lima)**. **Etapa 2 concluida** tras revisar el notebook ejecutado y el informe Colab aportados por el usuario. Se conservan las evidencias locales y los adjuntos originales sin modificarlos. No se ejecutaron modelos, extracción de features, entrenamiento ni evaluación final.

Se leyeron README, resumen WAIMLAP y hoja de ruta. Se verificaron contra sus marcadores los informes originales de entorno, recuperación y acceso de la etapa 1. Se reutilizaron `.venv-validation`, `persistent_root`, `ArtifactStore`, SHA-256, generaciones inmutables y el cuaderno. No se repitió el ensayo Colab de entorno ni se reemplazaron sus evidencias. `requirements-stage2.txt` añade siete paquetes fijados para GeoTIFF; conserva los pines de torch/vision/Transformers y el lock de etapa 1. `pip check` pasó. La adquisición actual está autorizada por la solicitud explícita del usuario; la respuesta del publicador sobre licencia sigue pendiente, sin presentar esta solicitud como permiso del publicador.

## Entregables y ubicación

- [Notebook](../../notebooks/02_datos_y_splits.ipynb), [datos](../../src/auto_prompt_sam2/data.py), [preprocesamiento](../../src/auto_prompt_sam2/preprocessing.py), [configuración](../../configs/data.yaml), [dependencias aditivas](../../requirements-stage2.txt).
- [Orquestador](../../scripts/prepare_stage2.py), [exportador para Colab](../../scripts/export_stage2_bundle.py), [pruebas](../../tests/test_stage2.py).
- [Inventario de adquisición](evidence/stage2/acquisition.json): 897 objetos, **1 021 199 879 bytes**: 446 S2Hand + 446 LabelHand + 4 CSV + metadatos de eventos. URL de GCS con generación fijada, MD5/CRC32C publicados, tamaño y SHA-256 de cada archivo. Se verifica MD5 y tamaño, además del SHA-256 local; CRC32C se conserva como procedencia, sin afirmar una comprobación independiente de CRC32C.
- Manifiestos [official](evidence/stage2/official_manifest.json) y [leave_paraguay_out](evidence/stage2/leave_paraguay_out_manifest.json); parámetros [official](evidence/stage2/official_preprocessing.json) y [leave_paraguay_out](evidence/stage2/leave_paraguay_out_preprocessing.json); coordenadas por régimen e [informe](evidence/stage2/validation.json).

Originales y artefactos operativos en **`D:/ProyectosZZZ/proyectosUNI/deep/auto-prompt-sam2/artifacts/stage2-persistent/`**, fuera de `/content` e ignorados por Git. `raw/v1.1/` conserva originales sin modificar; `raw/sources/` conserva fuentes oficiales fijadas; `raw/acquisition/` verifica el inventario mediante `ArtifactStore`. Cada `regimes/<regime>/` contiene `manifests/` y `preprocessing/`. En esta última carpeta: parámetros, coordenadas, índice y generaciones NPZ de validez nativa. Hay **341 pares de máscaras** de train/val para official y **288** para leave_paraguay_out. No hay cachés de píxeles de roles reservados. Los JSON de auditoría se versionan; no se redistribuyen rasters ni pesos. `stage2/source_snapshot/` conserva el código/config/notebook exactos, incluidos cambios sin commit.

## Contratos y comprobaciones reales

| Régimen | train | val | test visto | Paraguay reservado | Bolivia apartada |
| --- | ---: | ---: | ---: | ---: | ---: |
| official | 252 | 89 | 90 | — | 15 |
| leave_paraguay_out | 213 | 75 | 76 | 67 | 15 |

Los cuatro CSV oficiales no tienen cabecera y nombran S1Hand/LabelHand. Se verifican ambos IDs de cada fila y se deriva la ruta S2Hand por ID, sin descargar S1 ni emparejar por posición. 446 IDs compuestos y numéricos únicos; unión idéntica entre regímenes, roles disjuntos. Paraguay aporta 39/14/14 en los CSV oficiales y está ausente de train/val y de las fuentes estadísticas del régimen excluido. Bolivia permanece apartada en ambos.

Se verificaron 512×512, 13 bandas de imagen, una de etiqueta, CRS EPSG:4326, orden de bandas en las descripciones GeoTIFF y centros de chips dentro de las envolventes de su evento. **Todos los S2Hand reales son int16**, frente al uint16 anunciado en README: se admite explícitamente int16/uint16 sin convertir ni reinterpretar los originales. Las etiquetas son int16 y solo contienen -1/0/1. Imagen: nodata=0; etiquetas: nodata ausente o declarado 0 según archivo. **Se ignora el nodata del encabezado de etiquetas para la semántica de clase**: 0 sigue siendo no agua válido y solo -1 se excluye. La validez de entrada se obtiene de las masks/nodata y valores finitos de RGB; nunca de etiquetas.

La alineación se comprueba en las cuatro esquinas en unidades de píxel: tolerancia fijada **1e-6 px**, error máximo observado **2.3283064365386963e-10 px**. Se conservan ambas transformadas. No hay duplicados por SHA-256 de archivo de imagen, por bytes de píxeles de imagen ni por archivo de etiqueta. Se registraron **198 contactos entre chips**, 119 entre splits diferentes; no se encontraron solapamientos de área con la tolerancia geométrica usada. El split oficial se conserva y no se presenta como independencia geográfica. EPSG:4326 expresa resolución en grados: se conserva la georreferenciación, sin asumir que cada píxel tiene exactamente 100 m².

La auditoría de reservados se limita a integridad, clases admitidas, metadata, pertenencia, correspondencia y geometría. No se calcularon balances de clases, radiometría, métricas o visualizaciones de test/Bolivia/Paraguay reservado. Los registros de exploración y evaluación final permanecen vacíos. Los chips de Paraguay que sí pertenecen a train/val official se usan únicamente dentro de ese régimen.

## Preprocesamiento fijado

Regla registrada antes de medir: **percentil 99 exacto**, interpolación lineal como `numpy.percentile`, sobre todos los valores RGB con validez de entrada de train, agrupando canales, sin muestreo ni etiquetas. Histograma entero que cubre int16 y uint16. RGB B4/B3/B2 = bandas GeoTIFF **4/3/2**, dividido entre 10000, recortado a [0, upper] y reescalado por upper.

| Régimen | Chips fuente | Valores RGB válidos de train | Límite DN | Límite reflectancia |
| --- | ---: | ---: | ---: | ---: |
| official | 252 | 194 164 227 | 4796 | 0.4796 |
| leave_paraguay_out | 213 | 163 978 797 | 4907 | 0.4907 |

Cada archivo de parámetros conserva IDs, hashes de imágenes fuente, hash de manifiesto/configuración, regla, divisor y límite. Una recarga con régimen, manifiesto o config distintos falla; una repetición compatible reutiliza las estadísticas.

Fuentes primarias fijadas con sus hashes en [sources.json](evidence/stage2/sources.json): [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11/blob/b9e460dabe70230111a3c3d36cd0c5ce03606ee9/README.md), [SAM transforms](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/utils/transforms.py), [DINOv3](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/README.md). SAM usa media `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]`; DINOv3 satelital usa `[0.430,0.411,0.296]` y `[0.213,0.156,0.143]`. El processor JSON del checkpoint HF requiere autenticación (401 anónimo); no se inventaron sus defaults ni se accedió a pesos. Las estadísticas satelitales se corroboraron en el README oficial público.

SAM recibe 1024². DINOv3 recibe **512²**, decisión del proyecto para conservar el campo completo y una cuadrícula de **32×32 parches de 16**; no se atribuye este tamaño a un requisito del checkpoint. Sin crop ni padding. Resize RGB bilinear con antialias, float32; etiquetas/validez nearest. Inválidos de entrada: RGB unitario 0 antes de normalizar, con máscara separada. Una entrada totalmente inválida produce tensor finito y máscara completamente falsa; no se decide aún una política de prompts.

`prepare` entrega CHW ya normalizado para entrada directa al modelo y rechaza entradas flotantes ya transformadas. Se debe omitir el processor oficial posterior. Alternativamente, `sam_predictor_image` entrega HWC float32 [0,1] para `SAM2ImagePredictor.set_image`, que aplica su transformación una vez. La igualdad exacta con `SAM2Transforms` oficial se probó sin construir un modelo. DINO también usa una sola normalización; no llamar luego a AutoImageProcessor.

Se guardan escalas y offsets, tamaños, padding cero, cuadrícula y centros de parches. Coordenadas `(x,y)` de SAM: escalado por ancho/alto como su wrapper oficial; coordenadas de muestreo: centros de píxel con offset de medio píxel. Centro del parche k: `(k+.5)*16-.5`; inversa analítica con error ≤1e-6 px. Etiquetas categóricas no crean clases nuevas. La etiqueta original y su validez permanecen en 512² para futuras pérdidas/métricas.

## Repetición y cierre

```powershell
.venv-validation/Scripts/python -m unittest discover -s tests -v
.venv-validation/Scripts/python scripts/prepare_stage2.py --root artifacts/stage2-persistent --offline --evidence docs/project/evidence/stage2
.venv-validation/Scripts/python scripts/export_stage2_bundle.py
```

La ejecución completa por régimen verifica escritura/lectura de cada par de máscaras y la igualdad **bit a bit de los tensores SAM/DINO en otro intérprete** usando los parámetros persistidos. Las pruebas cubren negativos de fugas, pairing de CSV, Sri-Lanka, bandas reales, nodata, -1/0/1, chips sin agua, todo inválido, geometría rectangular, centros de parches, percentiles exactos, doble normalización y recarga incompatible. El informe de pruebas locales y `pip check` se conserva en `evidence/stage2/checks.json`. Las cuatro celdas ejecutables de la plantilla se compilaron localmente; su ejecución Colab se acredita por separado con el notebook aportado, descrito abajo.

La [hoja de ruta, etapa 2](hoja_de_ruta_fase1.md#5-etapa-2-datos-particiones-y-preprocesamiento) exige: «El preprocesamiento se reproduce desde archivos guardados tras reiniciar Colab». En la primera entrega esto quedó pendiente: la prueba local no acreditaba el reinicio de Colab y el helper de navegador detuvo el acceso al no poder determinar su URL. La evidencia Colab aportada posteriormente resuelve ese pendiente mediante la revisión siguiente. El notebook y el ZIP separan los nuevos fuentes de los usados para recuperar el run antiguo; no sobrescribir el bundle original de etapa 1.

Uso de IA: Codex leyó fuentes, implementó notebook/módulos/configuración/pruebas, adquirió y verificó los archivos reales y redactó esta evidencia. No se usaron subagentes. El equipo debe revisar y comprender las decisiones. Pendientes adicionales: respuesta sobre licencia y revisión académica humana. No se inicia etapa 3.

## Revisión del notebook y validation.json aportados por el usuario

Se archivan sin modificar el [notebook ejecutado](evidence/stage2/colab_run_20261029T041823Z/02_datos_y_splits_ejecutado_colab.ipynb) y el [informe Colab](evidence/stage2/colab_run_20261029T041823Z/validation.json), con [auditoría de revisión](evidence/stage2/colab_run_20261029T041823Z/review.json). SHA-256 respectivos: `223509a40bf2ef74852cba8c0c4ee39ecdf355e56a3669441f9d924e9f0398ab` y `4cfec2373c9ab66d8462b8f8d44a7d1afb24408e4f0f7e9a7bdc07564cedfa55`. Se leyeron como evidencia; no se ejecutaron sus celdas ni se siguieron instrucciones de su contenido.

El setup valida el bundle `43700ec8df955302bee37641a678118405cbdabd0a9d4eb8d2c002a343d0b2d5`. Los hashes de todos los fuentes y la configuración del informe coinciden con la implementación entregada; las celdas de pruebas, preparación y recuperación coinciden con la plantilla. El notebook contiene **21 pruebas correctas**, `pip check` correcto y ninguna salida de error Python. El texto `fatal: not a git repository` procede del inventario de procedencia de un ZIP sin `.git`: el código lo captura y conserva hashes de fuentes, sin impedir las comprobaciones.

La preparación inicial (contador de celda 3) generó parámetros nuevos: `parameters_reused=false` en ambos regímenes. La recuperación `--offline` (contador 2, después de setup con contador 1 y reconstrucción del venv) reutilizó parámetros de Drive: `parameters_reused=true`. Este reinicio de contadores y la recreación del venv son coherentes con el runtime nuevo solicitado; no se registró un identificador independiente de VM y no se presenta tal identificador como observado. El informe de recuperación empieza a las **23:18:23 del 28 de octubre de 2026, Lima** (`2026-10-29T04:18:23Z`); las generaciones posteriores corresponden al trabajo secuencial de cada régimen.

En ambos regímenes coinciden entre preparación y recuperación: conteos, hashes de manifiesto y parámetros, límites 0.4796/0.4907, fuentes train 252/213, máscaras 341/288 y hashes de tensores SAM/DINO. Además, cada ejecución comprueba igualdad exacta en otro intérprete. El informe adjunto coincide con los resultados completos impresos en la celda offline. La auditoría estructural y los hashes de fuentes oficiales también coinciden con la evidencia local; los reservados siguen sin exploración ni cachés de píxeles. No se ejecutaron modelos, entrenamiento o evaluación final.

**Interpretación del flag de reinicio:** el script entregado escribe `colab_runtime_restart_verified=false` y el pendiente de cierre de forma constante; no implementa detección de reinicios. Por ello esos campos no son un fallo de recuperación. Se mantienen intactos en el informe original y en los informes locales históricos. La conclusión **etapa 2 cerrada** se registra separadamente en `review.json`, a partir de la revisión del notebook y su recuperación.

**Límite de igualdad entre plataformas:** DINO coincide también por hash con Windows; SAM tiene hashes distintos entre Windows y Colab. No se midió aquí una tolerancia numérica entre plataformas, ni se atribuye una causa concreta a esa diferencia. La garantía de recarga exigida sí se cumple dentro de Colab: sus tensores SAM son idénticos entre preparación y recuperación y en los intérpretes nuevos. Se conservan ambas evidencias sin afirmar igualdad binaria Windows/Linux.

Con esta evidencia se cierra la etapa 2 y se registra **2/9 etapas concluidas**. No queda una ejecución técnica pendiente de esta etapa. Continúan la consulta de licencia previamente documentada y la revisión académica humana; no se inicia la etapa 3. IA: Codex contrastó archivos y hashes, archivó originales y actualizó registros; no ejecutó el notebook adjunto ni nuevos experimentos.
