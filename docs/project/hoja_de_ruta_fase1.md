# Hoja de ruta adaptada: fase 1 concluida en dos partes

**Actualización:** 3 de noviembre de 2026 (America/Lima). **Estado: CONCLUIDA.** Por instrucción del usuario se conservan los resultados publicados por el equipo y se cierra la fase 1 con el alcance de los notebooks `E2_01` y `E2_02`.

El [plan inicial de nueve etapas](hoja_de_ruta_fase1_original.md) queda archivado como antecedente. La simplificación cambia el alcance aceptado; no convierte los experimentos no realizados en resultados. El [estado actual](estado_actual_fase1.md) y el [informe de resultados](resultados_fase1.md) son las referencias vigentes.

## Organización realizada

| Parte | Notebook | Cobertura | Estado |
| --- | --- | --- | --- |
| 1 | [E2_01_datos_baseline_teardown.ipynb](../../notebooks/E2_01_datos_baseline_teardown.ipynb) | Entorno y datos; splits y RGB; línea base asistida SAM 2.1 tiny; parámetros, FLOPs parciales, features, ERF, latencia y memoria. | Concluida |
| 2 | [E2_02_intervencion_decoder.ipynb](../../notebooks/E2_02_intervencion_decoder.ipynb) | Hipótesis previa; ajuste solo del mask decoder; comparación congelado/ajustado con puntos idénticos, auditoría de gradientes, curvas, métricas, ejemplos y recuperación. | Concluida |

La parte 1 reutiliza `prepare_stage2.py`, `run_stage3.py` y `run_stage4.py --models sam`. La parte 2 utiliza `run_e2.py` y `configs/e2_intervention.yaml`. Los notebooks `01`–`04`, sus módulos y evidencias continúan disponibles como antecedentes; no constituyen una lista nueva de trabajo pendiente.

## Relación con las nueve etapas originales

| Etapa del plan original | Tratamiento en el cierre actual |
| --- | --- |
| 1. Entorno, accesos y persistencia | Antecedente concluido con salvedad de licencia; entorno reutilizado por ambas partes. |
| 2. Datos, splits y preprocesamiento | Cubierta en la parte 1; se conservan los manifiestos y validaciones previos. |
| 3. Línea base asistida | Cubierta en la parte 1 y como brazo congelado de la parte 2. |
| 4. Disección SAM/DINOv3 | Parte 1: SAM; se preserva la disección previa de SAM y DINOv3 como antecedente. |
| 5. Probe y prompts automáticos | Fuera del alcance cerrado; no implementados. |
| 6. Decoder con prompts automáticos por régimen | Sustituida en el alcance reducido por la intervención asistida E2 en `official`; no se afirma cumplir el protocolo automático original. |
| 7. Ablaciones A1–A3 y selección completa | Sustituida por una comparación congelado/ajustado en 40 chips con los mismos puntos; la matriz original no se ejecutó. |
| 8. Evaluación final y geográfica | Fuera del alcance cerrado; sin resultados en test, Bolivia o Paraguay reservado. |
| 9. Análisis y cierre de artefactos | Cubierta para el experimento reducido con cuaderno, resultados e índice de evidencia; no existe el paquete automático sin etiquetas previsto inicialmente. |

## Evidencia y criterio de cierre adoptado

- [x] Dos notebooks de reproducción y sus [copias ejecutadas en Colab](evidence/e2/colab_run_20261030T013250Z/README.md) disponibles.
- [x] Hipótesis previa, configuración y comparación controlada del decoder documentadas en [validacion_e2.md](validacion_e2.md).
- [x] Métricas y resultados negativos preservados: IoU 0.170378 → 0.308052 en la mejor época; 0.149928 en la última.
- [x] Interpretación, fallos, límites y uso de IA registrados en el [cuaderno](cuaderno_decisiones.md).
- [x] Artefactos versionados y ubicación/hashes de checkpoints en Drive identificados; no se presupone un paquete de inferencia automática.
- [x] Alcance reducido aceptado y fase 1 concluida por instrucción del usuario el 2026-11-03.

La aceptación corresponde al seguimiento del proyecto. La evaluación académica, la licencia del dataset y las entregas posteriores conservan su estado propio.

## Reproducción y continuidad

La [guía PASOS_E2](PASOS_E2.md) permite reproducir los experimentos si se necesita. No hace falta repetirlos para mantener este cierre. Todo entrenamiento y toda tarea que requiera o se beneficie de GPU se ejecuta exclusivamente en Google Colab; la PC se reserva para documentación, desarrollo y verificaciones ligeras de CPU.

La fase 2 pendiente contempla Gradio y Hugging Face Spaces. Debe partir de los componentes realmente disponibles y definir la fuente de prompts: el flujo cerrado requiere etiquetas. Automatizar los prompts, completar A1–A3 y probar transferencia geográfica son posibles ampliaciones con alcance y validación propios; no se reactivan por las instrucciones archivadas del plan original.
