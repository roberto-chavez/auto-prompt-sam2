# Validación local de la fase 2

**Fecha:** 11 de noviembre de 2026 (America/Lima). **Resultado: verificaciones locales aprobadas; inferencia y despliegue pendientes.**

Se ejecutó `scripts/validate_demo_local.py` en un entorno aislado `.venv-demo` sobre Windows, Python **3.10.10**, Gradio **6.31.0**, NumPy **2.2.6**, Pillow **11.3.0**, rasterio **1.4.3** y huggingface-hub **2.4.0**. Ese entorno no instala torch ni SAM. Los requisitos de inferencia son independientes y se reservan para Colab/Spaces.

| Verificación | Resultado observado |
| --- | --- |
| Sintaxis de los nuevos módulos, app, scripts y tests | Correcta. |
| `pip check` | Sin dependencias incompatibles en el entorno local. |
| Suite `tests/demo/test_demo.py` | **17 tests aprobados**, con bloqueo de importaciones de torch, torchvision, SAM y transformers. Incluye PNG RGBA y gris, adaptación sin deformación y exclusión del relleno. |
| Contrato de imagen | Orden B4/B3/B2 y RGB, escalado fijo, DN, nodata, tamaño 512², límite de archivo y rechazo de formatos ambiguos comprobados con archivos sintéticos. |
| Puntos y segmentación | Coordenadas `(x,y)`, negativos/positivos, duplicados, nodata, selección por score, limpieza del predictor y mismos inputs para ambos brazos comprobados con recorders, sin redes neuronales. |
| Decoder ausente/inválido | El backend conserva solo la salida congelada y deja la ajustada ausente; no sustituye una máscara por otra. Prueba con recorders. |
| Identidad del preprocesamiento | Hash canónico coincide exactamente con `report.json` del run E2 conservado. |
| Checkpoints | Verificadores de tamaño/hash y esquema `best` probados con bytes/diccionarios sintéticos. **No se verificó ni abrió un checkpoint real**, pues no está disponible. |
| HTTP de Gradio | Página raíz y `/config` responden correctamente con TestClient, sin servidor de inferencia. |
| Navegador local | Arranque de `app.py`, carga de GeoTIFF sintético de 13 bandas, punto positivo, punto negativo, tabla de coordenadas, deshacer y borrado comprobados. `Segmentar` deshabilitado en `preview`; sin máscaras generadas. |
| Conservación de fase 1 | `git diff` contra HEAD sin modificaciones de notebooks, evidencia, código experimental, configuraciones experimentales, dependencias de Colab e informe de resultados. Solo se añaden referencias de continuidad en documentos generales y una entrada nueva al cuaderno de decisiones. |
| Espacios en diff | `git diff --check` correcto. |

El reporte reproducible completo queda en el archivo local ignorado `artifacts/demo-local-validation.json`, con versiones, hashes de fuentes y salidas de pruebas. La captura `artifacts/demo-preview.jpg` muestra **un chip artificial para probar la interfaz**, no un dato Sentinel-2 ni una predicción. El servidor temporal de la revisión fue detenido; se inicia de nuevo con `python app.py` en el entorno indicado.

**Ampliación de carga:** se comprobó además el PNG adjuntado por el usuario (`descargar.png`, 636×658, RGBA) mediante el controlador real de carga y adición de puntos, sin importar modelos. El título visible identifica una visualización S1Hand de Sentinel-1; la demo admite prepararla, pero no se ha evaluado segmentación sobre ese dominio. No se copió el archivo del usuario a la evidencia versionada.

Las pruebas de comparación inyectan únicamente recorders y un context manager de Python para recorrer la lógica de despacho: no cargan modelos, tensores de pesos ni runtimes CUDA. En modo local, un intento de inferencia real falla antes de importar torch. No se ejecutó la suite experimental completa de los notebooks, que requiere sus dependencias y no fue modificada.

**No verificado:** construcción Docker/Linux, resolución transitiva completa del entorno de GPU, disponibilidad real de los checkpoints de Drive, deserialización real, compatibilidad de pesos con SAM instalado, inferencia GPU, paridad numérica, latencia/memoria, concurrencia real de sesiones con modelos, responsive/clics en distintos tamaños de pantalla, persistencia remota, URL pública o permisos de redistribución de ejemplos. No hay resultados nuevos de segmentación ni entrenamiento.

La [guía de fase 2](fase2_demo.md) especifica qué archivos pedir, cómo incorporarlos, cómo preparar Spaces y las pruebas de aceptación remota. No se declara fase 2 terminada ni desplegada por haber pasado estas comprobaciones locales.

## Revisión documental para GitHub

El 11 de noviembre se revisaron README, índice, guía de demo, contexto, hoja de ruta, cuaderno y referencias WAIMLAP. Se añadieron inicio rápido para Windows/Linux/macOS, explicación de puntos manuales y botón deshabilitado, formatos de imagen y una guía de publicación con selección explícita de **22 archivos** de código/configuración/documentación.

Se comprobaron **129 enlaces relativos** de los nueve documentos revisados, existencia de los archivos de la lista de publicación, bloques de código cerrados, espacios finales y `git diff --check`. Se confirmó que Git ignora los directorios de datos, pesos, artefactos, entorno virtual y `.env`. Los dos materiales del curso ya presentes sin seguimiento quedan fuera de la lista de publicación.

Esta revisión modifica documentación; conserva el resultado previo de 17 pruebas de implementación sin repetir modelos ni entrenamiento. No se realizó staging, commit, push ni despliegue. Los comandos para revisión y publicación por el equipo están en `publicacion_github.md`.
