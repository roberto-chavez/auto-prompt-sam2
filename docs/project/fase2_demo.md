# Fase 2: demo con puntos manuales

Actualización: **11 de noviembre de 2026, America/Lima**. Fase 1 conservada; esta fase corresponde a la demo, no a `E2_02` ni a una nueva evaluación académica.

La implementación permite cargar un chip Sentinel-2, añadir puntos positivos y negativos, y solicitar segmentación con SAM 2.1 tiny congelado y, opcionalmente, el mejor decoder ajustado de E2. El código de inferencia está preparado, pero **no se ha ejecutado con pesos reales durante este desarrollo**. No se ha publicado un Space. En la PC solo se preparan imágenes/puntos y se realizan pruebas con arrays sintéticos.

## Qué está disponible y qué falta

| Modalidad | Requisitos | Estado |
| --- | --- | --- |
| Preparar imagen y puntos en la PC | Dependencias de `requirements-demo.txt`; `preview` predeterminado. | Verificado localmente. El botón Segmentar está deshabilitado. |
| Segmentar con SAM original | Colab/Spaces con GPU, dependencias de inferencia, `DEMO_MODE=inference` y pesos base verificados. | Implementado; pendiente de prueba real. El base es público: no requiere el decoder del compañero. |
| Comparar original y decoder ajustado | Lo anterior más `best/payload`, comprobado contra el reporte E2. | Pendiente de recibir/integrar el checkpoint y validar ambos brazos en GPU. |
| Generar puntos automáticamente | Un generador de prompts y su evaluación. | Fuera de esta demo; no implementado. |

El flujo es **imagen → puntos manuales → predicción por píxel → máscara y superposición**. Los positivos indican la región que se quiere incluir y los negativos, qué excluir. Añadir puntos cambia la solicitud de segmentación; no actualiza los pesos. La opción **Automático** del selector solo reconoce el formato de entrada, no coloca puntos.

## Base revisada y resultados preservados

Se revisaron ambos [notebooks ejecutados](evidence/e2/colab_run_20261030T013250Z/notebooks/), sus celdas y salidas, [report.json](evidence/e2/colab_run_20261030T013250Z/run/report.json), el cierre de fase 1 y los módulos de preprocesamiento, predictor y checkpoints. Los títulos «NO EJECUTADO» incrustados son textos de plantilla históricos; las salidas sí registran ejecución. No se editaron esos archivos.

El run aceptado es `20261030T013250Z_2a795a89d5_f9fbd8b0`, commit experimental `f399047464b89a3ed18972f8dc7f1b1364d83a44`. IoU agua agrupado: congelado **0.17037849358838072**, mejor época 1 **0.3080519029370676**, última época 4 **0.14992778300691617**. Son resultados históricos de 40 chips de validación con puntos simulados desde etiquetas, no mediciones de esta demo. `recover` acredita un nuevo proceso en la misma sesión de Colab, no un reinicio completo.

La identidad del preprocesamiento existente coincide con E2: `a48c5416612393d1d1d96c4cfa0aac29588e8c96581538d6e2c89bbeb9c271fd` (SHA-256 de JSON canónico, no de cualquier formato textual). Se reutiliza `official_preprocessing.json` sin volver a ajustar estadísticas: B4/B3/B2, divisor 10000, límite de entrenamiento DN **4796**, reflectancia **0.4796**, inválidos rellenados con 0. El predictor recibe float32 HWC en [0,1] antes de su propia normalización. Nunca recibe los puntos pintados ni un tensor previamente normalizado.

## Uso local sin modelos

Desde la raíz del repositorio, PowerShell, Python 3.10 o superior:

```powershell
python -m venv .venv-demo
.\.venv-demo\Scripts\python.exe -m pip install -r requirements-demo.txt
.\.venv-demo\Scripts\python.exe scripts/check_demo_artifacts.py
.\.venv-demo\Scripts\python.exe app.py
```

Abrir [la interfaz local](http://127.0.0.1:7860). El valor predeterminado `preview` permite cargar imágenes, hacer clic, revisar coordenadas, deshacer y borrar. **Segmentar queda deshabilitado y no aparecen máscaras ficticias.** No se instala torch/SAM con esas dependencias. Incluso cambiando a `inference`, el backend rechaza una PC antes de importar torch o descargar pesos. No se modifican las restricciones Colab del código experimental de fase 1.

1. Cargar un archivo de hasta 32 MB. El modo **Automático** reconoce PNG/JPEG por extensión y trata los TIFF como Sen1Floods11; para TIFF de tres bandas elegir explícitamente RGB. Los GeoTIFF requieren chips **512×512**; recortar escenas mayores fuera de la demo, manteniendo las bandas alineadas y su escala. No se leen archivos `.SAFE` ni bandas sueltas `.jp2`.
2. **GeoTIFF Sen1Floods11:** 13 bandas en orden B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B10, B11, B12; se leen índices 4/3/2. **GeoTIFF RGB:** exactamente 3 bandas, orden B4/B3/B2. Ambos requieren DN int16/uint16. Para productos Sentinel-2 distintos, corregir previamente offsets radiométricos si los hay; la demo no infiere nivel de producto ni aplica offsets de procesamiento automáticamente.
3. **PNG/JPEG:** admite RGB, gris y transparencia, con tamaños distintos (hasta 16 millones de píxeles). Se convierte a RGB y se adapta a un lienzo 512×512 conservando proporciones; el espacio sobrante y los píxeles transparentes no admiten puntos ni predicción visible. Las coordenadas corresponden al lienzo preparado, cuya transformación queda en los metadatos. Se usa RGB/255, sin escalado DN. Las capturas con títulos o bordes se pueden abrir, pero esos elementos también llegan al modelo: usar preferentemente la imagen limpia. La carga no identifica el sensor: una visualización de Sentinel-1 radar es una entrada exploratoria fuera del dominio Sentinel-2 de E2, sin rendimiento evaluado.

4. Elegir positivo (+) sobre agua o negativo (−) sobre fondo y hacer clic. Se muestran coordenadas del lienzo `(x,y)` (nativas para TIFF; adaptadas para PNG/JPEG), origen arriba a la izquierda; máximo 64 puntos, sin duplicados, al menos un positivo para inferencia. No se admiten puntos sobre nodata de un TIFF.
5. En Colab/Spaces con inferencia habilitada, pulsar **Segmentar**. Se muestran superposiciones azules y máscaras binarias PNG descargables, sin georreferencia. Nodata se oculta en las máscaras y no se interpreta como agua. Los detalles conservan CRS/transformación de entrada, pero no se exporta GeoTIFF de salida.

`artifacts/demo-synthetic.tif` es un archivo artificial de pruebas, con 13 bandas int16; algunos visores comunes lo muestran oscuro o no lo admiten. La aplicación selecciona 4/3/2 y aplica la escala de visualización. No es una escena real ni una muestra de segmentación. No se incluye en GitHub: vive en `artifacts/`, ignorado por Git.

Todos los puntos se envían juntos como **un único prompt de objeto** a cada brazo. E2 hacía una llamada por componente obtenido de etiquetas y unía sus máscaras; aquí no hay etiquetas ni búsqueda de componentes. Se conserva `multimask_output=True`, elección del primer máximo de score predicho, umbral 0, FP32 sin autocast y sin postprocesamiento adicional. Los scores de SAM **no son IoU medida**. La diferencia visual no acredita mejora ni generalización. Para otra región, borrar puntos y repetir.

Un cambio de archivo, formato, puntos o comparación limpia las salidas anteriores. Cada sesión conserva sus propios puntos e imagen. La cola y el bloqueo del backend serializan las llamadas; cada predictor libera sus embeddings al terminar. Se usan dos instancias de SAM para no sobrescribir el decoder congelado al cargar el ajustado. Las imágenes se almacenan temporalmente por Gradio; se programa limpieza horaria de archivos con más de una hora, no un borrado instantáneo al cerrar la pestaña.

## Archivos exactos que pedir al compañero

Raíz de su Drive: `/content/drive/MyDrive/auto-prompt-sam2/`. Pedir archivos por sus rutas, no «el último checkpoint»: hay múltiples generaciones y `last` no equivale a `best`.

| Prioridad | Archivo relativo a esa raíz | Uso |
| --- | --- | --- |
| **Necesario para comparar** | `e2/runs/20261030T013250Z_2a795a89d5_f9fbd8b0/checkpoints/best/20261030T013915336249Z_81b60bb0014c/payload` | Pesos y metadatos del mejor decoder, época 1. Archivo PyTorch sin extensión, **16 902 201 bytes**. |
| Pedir junto con `payload` | `e2/runs/20261030T013250Z_2a795a89d5_f9fbd8b0/checkpoints/best/20261030T013915336249Z_81b60bb0014c/COMPLETE.json` | Marcador original de generación, identidad, tamaño y SHA. Trazabilidad; la app verifica directamente los bytes contra `report.json` y no necesita este marcador en runtime. |
| Alternativa a descargar SAM público | `base_models/huggingface/models--facebook--sam2.1-hiera-tiny/snapshots/de431c4043854a71d8101e17995dfe596bf101a5/sam2.1_hiera_tiny.pt` | Modelo base completo. Si es un enlace de la caché, compartir sus **bytes reales**, no un enlace roto. También puede descargarse en GPU remota desde el repo/revisión fijados. |
| Recomendado para prueba real privada | `raw/v1.1/data/flood_events/HandLabeled/S2Hand/Ghana_1033830_S2Hand.tif` y `.../S2Hand/India_1018317_S2Hand.tif` | Dos chips originales de validación ya presentes en los ejemplos E2, no capturas ni montajes PNG. Son suficientes para una primera prueba manual; no hace falta enviar todo el dataset. |
| Opcional para auditoría | `raw/v1.1/data/flood_events/HandLabeled/LabelHand/Ghana_1033830_LabelHand.tif` y `.../LabelHand/India_1018317_LabelHand.tif` | Revisar resultados offline. La interfaz no solicita ni usa etiquetas. |
| Opcional, **no usar en la demo como best** | `e2/runs/20261030T013250Z_2a795a89d5_f9fbd8b0/checkpoints/last/20261030T014721828771Z_e57b2f09960a/payload` y su `COMPLETE.json` | Respaldo/estudio de degradación o reanudación; **65 309 505 bytes**. No hace falta para inferencia con best. |

Hashes SHA-256 esperados:

```text
best/payload    11d21c5b810415b4506671ecc7f9694e0d71f03395cac071db0b56e6b081fe1e
SAM base        7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69
last/payload    eb227defab5c81ca0dc42612a52aee139a37ba91e23b115250daf44a3de89e4c
Ghana S2Hand    aebdf646e05e688da22d47ba947045f6af5df5e9d4b0d669903f998509043c69
India S2Hand    5abc50957e9f234ee9b273f017fa0a633e1c69e02ee363344df23f7d0d681f31
```

Ya están disponibles `report.json`, `history.csv`, `per_chip_val.csv`, `points_val.json`, `predictions_val.npz`, curvas, figuras, commit y los dos notebooks ejecutados. **No hace falta pedir nuevamente esos resultados para desarrollar la demo.** El archivo `e2/evidence/20261030T013250Z_2a795a89d5_f9fbd8b0-final.zip` excluye pesos y rasters por diseño (celda 17 de `E2_02`).

Tampoco falta el preprocesamiento: la copia del repositorio coincide con el reporte. Si necesita entregar el original de ese run, pedir `regimes/official/preprocessing/parameters/20261030T005758723167Z_d392212a6ef0/payload` y su `COMPLETE.json`; el payload es JSON. Su copia legible está en `stage2/evidence/official_preprocessing.json`. Los ZIP nativos de baseline/teardown de `E2_01` sirven para ampliar la auditoría, no bloquean la demo.

Si el compañero tiene **otro run**, pedir además su `report.json`, configuración E2, `code_commit.txt`, preprocesamiento completo, ambos notebooks ejecutados y evidencia de evaluación. No sustituir silenciosamente las cifras de fase 1. El cargador soporta otro reporte con la misma arquitectura y revisión de SAM, siempre que el hash de preprocesamiento y los bytes de su `best` coincidan. Arquitectura distinta o `state_dict` exportado sin el contenedor E2 requiere adaptar y validar el cargador, no renombrar archivos para engañarlo.

## Incorporar archivos mediante configuración

Configuración base: [configs/demo.json](../../configs/demo.json). Se puede crear una copia privada en `artifacts/demo-config.json` y seleccionarla con `DEMO_CONFIG`. Todas las rutas relativas se resuelven respecto de la raíz del repositorio, independientemente del directorio de la terminal. No subir pesos, rasters ni tokens a Git; usar `weights/`, `data/` y `artifacts/` ya ignorados. En particular, guardar el archivo sin extensión `payload` dentro de `weights/`.

| Variable | Función |
| --- | --- |
| `DEMO_MODE` | `preview` por defecto; `inference` solo con GPU en Colab/Spaces. |
| `DEMO_BASE_CHECKPOINT` | Ruta al SAM base completo. Si falta, se usa el repo/revisión/archivo del reporte cuando las descargas están habilitadas. |
| `DEMO_DECODER_CHECKPOINT` | Ruta directa a `best/.../payload`. Vacía significa que no hay decoder local. |
| `DEMO_DECODER_REPO_ID`, `DEMO_DECODER_REVISION`, `DEMO_DECODER_FILENAME` | Alternativa al decoder local: repositorio de modelos HF, commit de 40 caracteres y nombre del archivo. Los tres juntos. No se admite `main`. |
| `DEMO_ALLOW_DOWNLOAD` | `0` por defecto; `1` permite descargar pesos únicamente al pedir inferencia remota. |
| `DEMO_REPORT_PATH`, `DEMO_PREPROCESSING_PATH` | Referencias de integridad para otro run compatible. No se permite ignorar el hash. |
| `DEMO_CACHE_DIR` | Caché de pesos; por defecto `.cache/demo`. |
| `HF_TOKEN` | Solo si se accede a un repo privado. Configurarlo como secreto del servidor; nunca en JSON ni en la interfaz. |

Para comprobar los archivos recibidos en la PC sin deserializarlos:

```powershell
$env:DEMO_BASE_CHECKPOINT = 'weights/sam2.1_hiera_tiny.pt'
$env:DEMO_DECODER_CHECKPOINT = 'weights/e2-best/payload'
.\.venv-demo\Scripts\python.exe scripts/check_demo_artifacts.py
```

El preflight no descarga archivos ni importa torch. Con la configuración predeterminada informa que el base tiene origen remoto sin verificar y que el decoder no está configurado; **salida 0 no significa listo para inferencia**. Con rutas locales verifica existencia, SHA-256 y tamaño de `best`. Un error de integridad produce salida 1. La deserialización, verificación de época/configuración y `load_state_dict(strict=True)` quedan para GPU remota. El lector usa `torch.load(..., weights_only=True)` y carga únicamente `payload['best']['state']`, no el optimizador ni `decoder_state` de `last`.

## Desplegar en Hugging Face Spaces

Se prepara **Gradio dentro de un Docker Space**, con [Dockerfile](../../Dockerfile), `.dockerignore`, dependencias separadas y metadatos YAML al principio del README raíz. Docker permite fijar SAM por commit e instalarlo con `SAM2_BUILD_CUDA=0`. Puerto 7860, usuario UID 1000, caché escribible; no se descargan pesos durante el build. Configuración conforme a la [documentación de Docker Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker).

1. Crear un Space con SDK **Docker** y subir el código revisado. Comprobar en la cuenta la disponibilidad/coste del hardware; la demo de inferencia necesita **GPU CUDA dedicada**, no CPU ni ZeroGPU. En CPU se puede revisar el modo `preview`. Esta tarea no crea recursos ni contrata hardware.
2. El build instala Python 3.12, torch 2.7.1, torchvision 0.22.1 y SAM commit `2b90b9f5ceec907a1c18123530e92e794ad901a4`. La UI se fija en Gradio 6.31.0. Las dependencias directas están fijadas, pero no se afirma que exista un lock transitivo Linux ni que el contenedor se haya construido aquí. Conservar el log de build y `pip freeze` del primer despliegue validado.
3. Mantener inicialmente `DEMO_MODE=preview` y verificar arranque/carga de imagen. Para activar SAM base configurar Variables `DEMO_MODE=inference` y `DEMO_ALLOW_DOWNLOAD=1`. El modelo público se obtiene desde la revisión HF `de431c4043854a71d8101e17995dfe596bf101a5` y se verifica su hash.
4. Para comparar, alojar los bytes exactos de `best/payload` en un repo de modelos del equipo, privado si corresponde. Configurar `DEMO_DECODER_REPO_ID`, el commit fijo en `DEMO_DECODER_REVISION` y `DEMO_DECODER_FILENAME` (por ejemplo `e2-best/payload`). Añadir `HF_TOKEN` de lectura como **Secret** si hace falta. No introducir secretos como variables públicas ni incrustarlos en Docker. [Variables y secretos de Spaces](https://huggingface.co/docs/hub/spaces-overview#managing-secrets-and-environment-variables).
5. Reiniciar el Space tras cambios y seguir la aceptación remota de abajo. Si no se configura decoder, SAM congelado sigue disponible. Si se configura uno ausente, corrupto o incompatible, se conserva la predicción congelada y se indica el fallo del ajustado; nunca se copia la máscara del base para simular comparación.

El Dockerfile copia solo código, configuración y dos JSON de referencia, sin rasters ni pesos. El entrenamiento y los entornos anteriores no cambian. Antes de publicar ejemplos de Sen1Floods11, resolver su licencia pendiente según [accesos_y_licencias.md](accesos_y_licencias.md); no se agregaron chips públicos. El contenedor no necesita DINOv3 ni acceso a su repositorio.

## Alternativa de validación en Colab

Usar un checkout **separado de fase 2** y un entorno nuevo, sin modificar el entorno experimental ni el archivo `e2/code_commit.txt` del run cerrado. En GPU Colab:

```bash
python -m venv /content/demo-venv
/content/demo-venv/bin/pip install -r requirements-demo-inference.txt
/content/demo-venv/bin/pip install setuptools==80.9.0 wheel==0.45.1
SAM2_BUILD_CUDA=0 /content/demo-venv/bin/pip install --no-build-isolation --no-deps \
  git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4
```

Montar Drive desde el notebook de Colab. Exportar `AUTO_PROMPT_EXECUTION_CONTEXT=google_colab_user_notebook`, `DEMO_MODE=inference`, `DEMO_BASE_CHECKPOINT` y `DEMO_DECODER_CHECKPOINT` con las rutas exactas de la tabla, manteniendo `DEMO_ALLOW_DOWNLOAD=0` si todo está en Drive. Ejecutar el preflight. Para una UI accesible desde Colab, lanzar en ese entorno un script que importe `build_demo` desde `src` y llame `build_demo().launch(share=True)`; el enlace compartido expone temporalmente la aplicación. `app.py` mantiene `share=False` por defecto. No ejecutar `run_e2.py prepare` ni `resume`: esta fase solo necesita inferencia.

## Verificaciones y aceptación pendiente

Para preparar el commit y revisar qué archivos subir, seguir [publicacion_github.md](publicacion_github.md).

Pruebas locales, sin modelos:

```powershell
.\.venv-demo\Scripts\python.exe scripts/validate_demo_local.py
```

El script valida sintaxis, configuración, dependencias, pruebas de `tests/demo/`, arranque HTTP de Gradio en memoria y conservación de los archivos versionados de fase 1 contra HEAD. Registra el resultado en `artifacts/demo-local-validation.json`. Las pruebas quedan en una subcarpeta sin `__init__.py` para no introducir Gradio en el descubrimiento de tests de los notebooks de fase 1. No ejecutar los orquestadores antiguos para validar la demo: algunos escriben evidencia histórica.

La validación local cubre RGB/B4-B3-B2, escala fija, nodata, dimensiones/tipos/límites, coordenadas, selección por score, limpieza de estados, formatos de checkpoints con diccionarios sintéticos y ausencia de importaciones de modelos. Un recorder de pruebas devuelve arrays artificiales únicamente dentro de tests; la aplicación no tiene backend de segmentación ficticia.

**Falta antes de declarar la demo completa y desplegada:**

- Recibir `best/payload` y verificar tamaño/hash; conservar su marcador. Obtener uno o dos chips originales por canal privado.
- Construir el contenedor o instalar el entorno separado de Colab. Verificar dependencias Linux/CUDA, carga real de SAM y carga estricta del decoder. No se ha validado consumo GPU, latencia, estabilidad ni presupuesto del Space.
- Probar primero SAM congelado y luego comparación con imagen y puntos idénticos; guardar configuración, commit de fase 2, hashes, máscaras y capturas reales. No reutilizar cifras E2 como métricas de los prompts manuales.
- Comprobar positivos/negativos, esquinas, nodata, deshacer/borrar, cambio de chip, decoder ausente/corrupto, dos sesiones concurrentes y recarga tras reinicio. Verificar en navegador que el clic coincide con el píxel al cambiar el tamaño de la ventana.
- Si se desea paridad numérica con E2, ejecutar aparte su protocolo por componentes con `points_val.json`, imágenes originales y el mismo evaluador; los clics humanos no permiten atribuir una diferencia al decoder de forma equivalente al experimento original.
- Completar publicación y URL, permisos de ejemplos y revisión del equipo. Fase C de estrés, informe final, póster y defensa siguen siendo entregas separadas.

La implementación usa la [API oficial del predictor SAM fijado](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/sam2_image_predictor.py) y los [eventos de Gradio 6.31.0](https://github.com/gradio-app/gradio/blob/gradio%406.31.0/gradio/events.py). El estado de la revisión local y sus límites se registra en [validacion_fase2.md](validacion_fase2.md).

## E3: verificación con pesos reales y flujo automático (2026-11-11)

**Alcance probado hasta ahora:** solo pruebas locales de interfaz (17). **La inferencia de punta a punta con pesos reales no se ha ejecutado** (sin GPU ni Hugging Face en el entorno de redacción; en Drive no existe `e3_demo_smoke.json`). Para ejecutarla: abrir `notebooks/E3_demo_verificacion_colab.ipynb` en Colab (GPU T4), ejecutar las celdas en orden y descargar `/content/e3_demo_smoke.json` y `/content/e3_demo_mask.png`. Modo cubierto por ese notebook: **puntos manuales con SAM 2.1 tiny congelado**. No guardar la visualización en GitHub (licencia de Sen1Floods11 pendiente).

**Cómo conectar el flujo automático (no implementado):** en `demo_backend.py`, añadir `segment_automatic(image)` que (1) cargue DINOv3 con `features.load_frozen_dino` y obtenga la rejilla de características, (2) calcule la probabilidad con `probe.probabilities` y el bundle de sonda del régimen (`official_complete_bundle`, generado en la etapa 5, almacenado en Drive), (3) llame a `automatic_prompts.infer_points(predictor, image, probability, valid, policy, positive_threshold)` con la política y el umbral fijados (`configs/prompts.yaml`, C=0.1, umbral 0.7) y (4) devuelva la máscara. Requisitos: `HF_TOKEN` (DINOv3 gated), el bundle de sonda y el decoder `a3_tuned` de la etapa 6 (de Drive). Debe probarse en Colab contra la fila `a1_points` de la etapa 7 (mismo chip, mismo IoU).
