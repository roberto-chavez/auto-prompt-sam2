---
title: El costo del prompt
emoji: 🌊
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# auto-prompt-sam2

**El costo del prompt: segmentación de agua superficial en Sentinel-2 con SAM 2.1.**

Proyecto de **Deep Learning (1INF52), semestre 2026-2**, Ingeniería Informática, **Pontificia Universidad Católica del Perú (PUCP)**, con materiales para WAIMLAP 2026.

> **Estado al 11 de noviembre de 2026: fase 1 concluida; fase 2 implementada hasta la integración de pesos y validación remota.** Se conservan los resultados del equipo y los notebooks `E2_01` y `E2_02`. La demo Gradio permite preparar imágenes y puntos sin modelos en la PC; la inferencia y el despliegue quedan por verificar en GPU remota. [Guía de la demo y archivos que pedir al compañero](docs/project/fase2_demo.md).

La propuesta original estudiaba prompts automáticos derivados de DINOv3 para SAM 2.1. La fase 1 realizada se concentra en **datos, referencia asistida, disección del modelo y ajuste exclusivo del mask decoder**, con puntos simulados desde las etiquetas. El cierre acepta ese alcance y conserva sus resultados y limitaciones; no supone haber implementado el flujo automático DINOv3→SAM ni evaluado la generalización geográfica.

## E3 — punto de entrada (actualizado 2026-11-11)

- **Informe (IEEE, 4 págs.):** [`reports/e3/informe_E3.pdf`](reports/e3/informe_E3.pdf); fuente IEEEtran (conference): [`informe_E3_IEEEtran.tex`](reports/e3/informe_E3_IEEEtran.tex) con `fig_iou.pdf` y `fig_curvas.pdf`. Compilar: `pdflatex` dos veces. Retroalimentación del workshop: pendiente de confirmación.
- **Estado, límites y compuerta de etapa 8:** [`docs/project/e3_estado.md`](docs/project/e3_estado.md).
- **Resultados consolidados (validación, A1–A3 incluidos):** [`consolidated_results.md`](docs/project/evidence/e3/consolidated_results.md); regenerar con `python scripts/e3_collect_results.py`.
- **Notebook etapa 7 ejecutado:** [`notebooks/07_ablaciones_y_seleccion_ejecutado.ipynb`](notebooks/07_ablaciones_y_seleccion_ejecutado.ipynb); métricas extraídas en `docs/project/evidence/e3/stage7_metrics_from_notebook.json`.
- **Demo:** pruebas locales de interfaz; verificación con pesos reales en Colab: `notebooks/E3_demo_verificacion_colab.ipynb` (preparada, no ejecutada). Solo puntos manuales.
- **Licencias:** Sen1Floods11 pendiente; mensaje al publicador en [`accesos_y_licencias.md`](docs/project/accesos_y_licencias.md).
- **Etapa 8:** evaluador y protocolo listos y probados con datos sintéticos (`src/auto_prompt_sam2/evaluation.py`, `configs/stage8_protocol.json`); compuerta bloqueada (`python scripts/check_stage8_gate.py <revision_stage7_*.json>`); conjuntos reservados cerrados.
- **Pendiente:** totales reales de puntos (`scripts/e3_point_budget.py`), revisión humana y recover offline de la etapa 7, etapa 8, licencia de Sen1Floods11.

## Probar la demo

La demo permite cargar una imagen y colocar **puntos positivos sobre agua y negativos sobre fondo**. Con inferencia habilitada, SAM usa esas indicaciones para delimitar la región y mostrar su máscara. La comparación entrega la misma imagen y los mismos puntos a SAM original y al modelo con el decoder ajustado.

**El modo predeterminado es `preview`: “Segmentar” está deshabilitado.** En la PC se puede revisar la carga y los puntos; todavía no se ha validado inferencia real ni publicado una URL de demo. Para activar el brazo original se necesita GPU en Colab/Spaces y SAM base, que puede descargarse públicamente. Para comparar hace falta además el `best/payload` del compañero. Recibir ese archivo no activa inferencia en la PC.

Desde la raíz del repositorio, Python 3.10 o superior, en PowerShell:

```powershell
python -m venv .venv-demo
.\.venv-demo\Scripts\python.exe -m pip install -r requirements-demo.txt
.\.venv-demo\Scripts\python.exe app.py
```

En Linux/macOS, para preparar la interfaz:

```bash
python3 -m venv .venv-demo
.venv-demo/bin/python -m pip install -r requirements-demo.txt
.venv-demo/bin/python app.py
```

Abrir [la interfaz local](http://127.0.0.1:7860). En **Automático (según extensión)** se reconocen PNG/JPEG y TIFF de Sen1Floods11; esa opción elige el formato del archivo, **no genera los puntos**.

| Entrada | Tratamiento |
| --- | --- |
| GeoTIFF Sentinel-2/Sen1Floods11 | Chip 512×512, 13 bandas DN int16/uint16; RGB B4/B3/B2 con preprocesamiento fijo de train. Para TIFF RGB de tres bandas, elegir su formato explícitamente. |
| PNG/JPEG | RGB, gris o transparencia; se adapta sin deformar a un lienzo 512×512. Preferir imágenes limpias, sin títulos ni bordes. |
| Visualización Sentinel-1, como un PNG de `S1Hand` | Se puede cargar como imagen exploratoria; el rendimiento del decoder ajustado con Sentinel-2 no está evaluado sobre radar. |

Los puntos son **manuales** en esta demo; colocarlos no entrena al modelo. En E2 se simularon desde etiquetas. El generador automático DINOv3→SAM de la propuesta original sigue sin implementarse.

Consultar la [guía completa y los checkpoints que pedir](docs/project/fase2_demo.md), la [validación local y sus límites](docs/project/validacion_fase2.md) y la [guía para publicar estos cambios en GitHub](docs/project/publicacion_github.md). La configuración está en [configs/demo.json](configs/demo.json); el despliegue se prepara con [Dockerfile](Dockerfile).

## Fase 1 concluida: dos notebooks

| Parte | Notebook | Trabajo realizado |
| --- | --- | --- |
| 1. Datos, línea base y disección | [E2_01_datos_baseline_teardown.ipynb](notebooks/E2_01_datos_baseline_teardown.ipynb) | Entorno Colab, Sen1Floods11, splits y preprocesamiento, SAM 2.1 tiny congelado, parámetros, FLOPs parciales, features, campo receptivo efectivo, latencia y memoria. |
| 2. Intervención del decoder | [E2_02_intervencion_decoder.ipynb](notebooks/E2_02_intervencion_decoder.ipynb) | Hipótesis previa y comparación controlada de decoder congelado frente a ajustado, BCE+Dice, auditoría de gradientes, métricas, curvas, ejemplos y recuperación. |

Los notebooks ejecutables se mantienen sin salidas; las [copias ejecutadas y resultados originales](docs/project/evidence/e2/colab_run_20261030T013250Z/README.md) están separados como evidencia. Los notebooks `01`–`04` y sus validaciones se conservan como antecedentes reutilizados por el flujo de dos partes.

## Resultados conservados

Run `20261030T013250Z_2a795a89d5_f9fbd8b0`, régimen `official`, Tesla T4 en Colab, semilla 42, **30 chips de entrenamiento y 40 de validación**. Fuente: [report.json](docs/project/evidence/e2/colab_run_20261030T013250Z/run/report.json).

| Configuración | IoU agua agrupado | Precisión | Recall | Dice |
| --- | ---: | ---: | ---: | ---: |
| Decoder congelado, época 0 | 0.170378 | 0.171475 | 0.963819 | 0.291151 |
| Decoder ajustado, mejor época (1) | **0.308052** | 0.398793 | 0.575162 | 0.471009 |
| Decoder ajustado, última época (4) | 0.149928 | — | — | — |

La mejor época aumenta la IoU en **0.137673** (13.77 puntos porcentuales): reduce falsos positivos y también pierde agua real. Las épocas 2–4 empeoran respecto del decoder congelado. Se conservan ambos resultados; la causa de la degradación no fue demostrada. Solo se configuraron como entrenables los **4 215 109 parámetros del decoder**, de 38 962 498 en SAM 2.1 tiny; los bloques congelados mantuvieron sus huellas.

La referencia asistida sobre validación completa del antecedente `03` obtuvo IoU **0.158044** (`official`, 89 chips) y **0.168434** (`leave_paraguay_out`, 75 chips). Son poblaciones distintas de los 40 chips de E2. El segundo régimen en preparación, baseline y disección no constituye una evaluación en Paraguay reservado.

El [informe de resultados](docs/project/resultados_fase1.md) reúne configuración, interpretación, artefactos y límites. La [validación de E2](docs/project/validacion_e2.md), el [teardown de SAM](docs/project/teardown_sam2.md) y la [auditoría](docs/project/auditoria_E2.md) conservan el detalle técnico.

## Alcance y límites del cierre

- Los prompts proceden de las etiquetas: **la referencia es asistida y requiere etiquetas para generar sus puntos**. Los chips sin agua reciben una predicción vacía por construcción.
- DINOv3 se instrumentó en la disección anterior, pero no se utilizó para generar prompts ni en el entrenamiento de E2. No hay probe de agua entrenado ni pipeline automático terminado.
- Una semilla, subconjuntos pequeños y mejor época seleccionada y reportada en la misma validación; no se afirma significancia estadística ni rendimiento final en test, Bolivia o Paraguay reservado.
- FLOPs y campo receptivo tienen cobertura y límites metodológicos declarados. `recover` de E2 reprodujo la evaluación en un proceso nuevo de la misma sesión; no hay evidencia de reinicio real del runtime de E2.
- La consulta de licencia de Sen1Floods11 sigue pendiente según el [registro de accesos y licencias](docs/project/accesos_y_licencias.md). El cierre de seguimiento no modifica ese estado.

**Concluir la fase 1 del proyecto no equivale a una calificación académica de E2 ni al cumplimiento de todos los entregables finales del curso.** Las [bases del curso](docs/course/tarea_final_deep_learning.md) se conservan como referencia de requisitos.

## Organización y documentación

| Recurso | Uso |
| --- | --- |
| [Estado y contexto de fase 1](docs/project/estado_actual_fase1.md) | Fuente vigente para continuar: fase concluida, evidencia y límites. |
| [Hoja de ruta adaptada](docs/project/hoja_de_ruta_fase1.md) | Correspondencia entre las dos partes ejecutadas y las nueve etapas iniciales. |
| [Guía de reproducción](docs/project/PASOS_E2.md) | Ejecución opcional en Colab; no es un requisito adicional de cierre. |
| [Cuaderno de decisiones](docs/project/cuaderno_decisiones.md) | Hipótesis, decisiones, resultados y uso de IA; incluye la decisión de cierre. |
| [Índice del proyecto](docs/project/README.md) | Documentos vigentes, antecedentes y evidencia preservada. |
| [Materiales WAIMLAP](docs/WAIMLAP/README.md) | Resumen actualizado y estado del póster original. |
| [Propuesta inicial](docs/project/draft/propuesta_inicial_deep_learning.md) | Antecedente del alcance original con DINOv3. |
| [Plan original de nueve etapas](docs/project/hoja_de_ruta_fase1_original.md) | Archivo histórico; las antiguas etapas 5–9 no son tareas activas de fase 1. |

El código está en `src/auto_prompt_sam2/`, los orquestadores en `scripts/`, las configuraciones en `configs/` y las pruebas en `tests/`. Las dependencias se fijan en [requirements-colab.txt](requirements-colab.txt) y [requirements-stage2.txt](requirements-stage2.txt). Se usan RGB B4/B3/B2 de Sen1Floods11 (446 chips manuales); las etiquetas describen agua superficial, incluida agua permanente, y los píxeles inválidos se excluyen de pérdidas y métricas.

## Reproducción y conservación

Seguir [PASOS_E2.md](docs/project/PASOS_E2.md) para reproducir voluntariamente las dos partes en Google Colab con T4 y Drive. El código del run conservado corresponde al commit `f399047464b89a3ed18972f8dc7f1b1364d83a44`; cada nueva ejecución debe registrar su propio commit y resultados. Todo entrenamiento se mantiene exclusivamente en Colab. La inferencia de la demo de fase 2 se prepara para Colab o Hugging Face Spaces con GPU; la PC se reserva para desarrollo, preparación de imágenes y verificaciones sin modelos.

Las métricas, predicciones, puntos y notebooks ejecutados están versionados como evidencia. Los datos originales, pesos y checkpoints permanecen en Drive; ubicación y hashes de los checkpoints `best` y `last` constan en el [índice del run](docs/project/evidence/e2/colab_run_20261030T013250Z/README.md). Esta actualización documental no reentrena ni sustituye resultados.

## Fase 2 y entregas posteriores

**Implementado:** [app.py](app.py), carga de chips Sentinel-2 de 512×512 y visualizaciones PNG/JPEG, puntos positivos/negativos manuales, superposición y máscaras binarias, configuración de pesos locales o de Hugging Face, y Dockerfile para Spaces. La PC utiliza `preview` por defecto, sin cargar SAM ni producir máscaras. La inferencia queda restringida a Colab/Spaces con GPU; el decoder ajustado es opcional y se verifica contra el reporte E2 antes de cargarlo.

El [inicio rápido](#probar-la-demo) permite revisar la interfaz. La [guía de fase 2](docs/project/fase2_demo.md) incluye configuración, rutas y hashes exactos de checkpoints, pasos para Spaces y aceptación remota. El [registro de validación](docs/project/validacion_fase2.md) documenta 17 pruebas locales sin modelos. **Pendiente:** recibir `best/payload` para la comparación, probar inferencia GPU y completar el despliegue. SAM base puede obtenerse públicamente y validarse sin esperar al decoder ajustado.

El informe final, el póster actualizado, la demo y la defensa oral se gestionan por separado. Los requisitos académicos de estrés y análisis de fallos siguen siendo referencia para esas entregas; cerrar la fase 1 reducida no los acredita. Las fechas consignadas en las bases son E1: 7 de octubre; E2: 21 de octubre; trabajo e informe final: 11 de noviembre; póster WAIMLAP: 18–20 de noviembre de 2026.

## Equipo

- **Eliezer Guido Villarreal Quispe**
- **Roberto Abel Chavez Javier**

Los datos institucionales y de contacto aún incompletos en los materiales académicos deben completarse antes de su entrega.
