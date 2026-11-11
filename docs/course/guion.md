# Guion de exposición: Entregable 2

**Expositores:** Roberto y Eliezer. **Duración orientativa:** 16 min 20 s, más preguntas. **Presentación:** [entregable2.pptx](entregable2.pptx). **Corte de la evidencia:** 3 de noviembre de 2026.

La exposición principal comprende las diapositivas 1–18. Las diapositivas 19–20 son de respaldo. Los tiempos incluyen pausas para señalar figuras y cambios de expositor. Roberto tiene aproximadamente 8 min 05 s y Eliezer 8 min 15 s. El reparto organiza la exposición, no atribuye autoría individual de la implementación.

## Reparto y cambios de expositor

| Diapositivas | Expositor | Tema | Intervalo |
| --- | --- | --- | --- |
| 1–3 | Roberto | Problema, datos y referencia asistida | 00:00–02:20 |
| 4–8 | Eliezer | Arquitectura y teardown instrumentado | 02:20–06:50 |
| 9–13 | Roberto | Hipótesis, entrenamiento y resultados | 06:50–11:35 |
| 14–17 | Eliezer | Imágenes, decisiones y reproducción | 11:35–15:20 |
| 18 | Roberto | Conclusión y límites | 15:20–16:20 |
| 19–20 | Ambos | Respaldo para preguntas y fuentes | Según preguntas |

## Preparación

- Mantener abiertos el [cuaderno de decisiones](../project/cuaderno_decisiones.md) y el [índice de evidencia E2](../project/evidence/e2/colab_run_20261030T013250Z/README.md). Ambos tienen enlaces en la presentación.
- Ensayar las diferencias entre IoU predicho e IoU real, IoU agrupada y promedio por chip, parámetros entrenables y parámetros con gradiente.
- Los valores 0.1580 y 0.1704 corresponden a conjuntos distintos. La mejora de E2 compara 0.1704 con 0.3081 en los mismos 40 chips.
- La exposición usa resultados conservados. No necesita volver a entrenar ni ejecutar los notebooks.

## Texto por diapositiva

### 1. El costo del prompt en SAM

**Expone:** Roberto. **Tiempo:** 00:00–00:35.

**Texto sugerido**

Somos Roberto y Eliezer. Estudiamos la segmentación de agua en imágenes satelitales con SAM 2.1. En este entregable presentamos la disección del modelo base, una intervención preliminar en su decoder y las decisiones que conservamos en GitHub. El alcance que sí ejecutamos utiliza puntos simulados desde etiquetas. La pregunta de reemplazarlos por puntos automáticos sigue fuera de estos resultados.

**Evidencia:** [resultados_fase1.md](../project/resultados_fase1.md), [cuaderno_decisiones.md](../project/cuaderno_decisiones.md).

### 2. Datos y alcance de E2

**Expone:** Roberto. **Tiempo:** 00:35–01:25.

**Texto sugerido**

Usamos Sen1Floods11, con imágenes Sentinel-2 y etiquetas manuales de agua. El repositorio auditó 446 pares de chips de 512 por 512 píxeles. En el split oficial hay 252 chips de entrenamiento, 89 de validación, 90 de test y 15 de Bolivia separados. Para la intervención tomamos tres chips de entrenamiento y cuatro de validación por evento, ordenados por identificador, sin seleccionar por el resultado. Son 30 y 40 chips. El preprocesamiento ajusta sus estadísticas solo con train. No evaluamos test ni los conjuntos reservados.

**Evidencia:** [validacion_etapa2.md](../project/validacion_etapa2.md), [e2_intervention.yaml](../../configs/e2_intervention.yaml), [report.json](../project/evidence/e2/colab_run_20261030T013250Z/run/report.json).

### 3. La referencia es asistida por etiquetas

**Expone:** Roberto. **Tiempo:** 01:25–02:20.

**Texto sugerido**

La etiqueta permite identificar cada componente conectado de agua y generar un punto positivo y uno negativo. SAM recibe esos puntos y la imagen, produce tres máscaras candidatas y elegimos la que tiene mayor IoU predicho. Ese puntaje no es el IoU calculado con la etiqueta. Unimos las máscaras elegidas para obtener la segmentación del chip. Mantener exactamente esta política en ambos brazos permite estudiar el ajuste del decoder. Hay una ventaja importante de esta asistencia: si la etiqueta no tiene agua, no generamos componentes y la predicción queda vacía por construcción.

**Al señalar la diapositiva:** Distinguir IoU predicho del IoU real de evaluación.

**Cambio de expositor:** «Eliezer, con esta referencia asistida definida, explícanos qué encontramos al abrir SAM por dentro.»

**Evidencia:** [e2_intervention.yaml](../../configs/e2_intervention.yaml), [intervention.py](../../src/auto_prompt_sam2/intervention.py).

### 4. Anatomía de SAM 2.1 Hiera tiny

**Expone:** Eliezer. **Tiempo:** 02:20–03:20.

**Texto sugerido**

SAM separa el procesamiento de la imagen del procesamiento del prompt. Hiera reduce la resolución espacial y aumenta el número de canales en cuatro etapas. El FPN integra características y entrega el embedding principal de 256 canales por 64 por 64 posiciones, además de mapas de alta resolución. El prompt encoder representa los puntos. El decoder cruza los tokens con la imagen mediante un transformer bidireccional y construye las máscaras. Esta separación permite reutilizar el embedding de una imagen para varios prompts. Los módulos de memoria sirven para vídeo y no se ejecutan en nuestra ruta de imagen estática.

**Evidencia:** [teardown_sam2.md](../project/teardown_sam2.md).

### 5. El encoder concentra el cómputo contado

**Expone:** Eliezer. **Tiempo:** 03:20–04:15.

**Texto sugerido**

Contamos casi 39 millones de parámetros. El encoder de imagen concentra el 69.86 por ciento de los parámetros y el 98.22 por ciento de los FLOPs contados. El decoder tiene el 10.82 por ciento de los parámetros y el 1.23 por ciento del cómputo contado. Esto justifica explorar una intervención pequeña sin ajustar todo el backbone. La memoria de vídeo conserva 7.52 millones de parámetros, aunque no recibe llamadas. Los 296.05 GFLOPs son parciales: el contador de PyTorch solo cubre operaciones con fórmula registrada. No podemos presentarlos como todo el trabajo real del modelo.

**Detalle para la defensa:** `conv_s0/s1` pertenecen al decoder por parámetros, pero se ejecutan durante `forward_image`; su costo queda en el residual del contador. Las filas de parámetros y FLOPs usan esa atribución explícita y no duplican el costo. Los FLOPs omiten operaciones sin fórmula registrada, como algunas interpolaciones y activaciones.

**Evidencia:** [teardown_sam2.md](../project/teardown_sam2.md), [stage4_table_1_from_notebook.csv](../project/evidence/e2/colab_run_20261030T013250Z/derived_from_notebooks/stage4_table_1_from_notebook.csv).

### 6. Características intermedias de Hiera

**Expone:** Eliezer. **Tiempo:** 04:15–05:05.

**Texto sugerido**

Estos paneles son salidas reales del notebook. Mostramos el canal cero de los bloques cero, seis y once. La resolución baja de 256 por 256 a 64 por 64 y luego a 32 por 32, mientras aumentan los canales. Al inicio se ven texturas finas, y al final estructuras más gruesas. Eso describe la jerarquía espacial. No asignamos a estos canales el significado de agua, y tampoco comparamos intensidades de color entre paneles porque cada uno tiene su propia escala. Son observaciones de dos chips de entrenamiento, no una evaluación de generalización.

**Detalle para la defensa:** las formas que se muestran son H × W × C, sin el batch 1. Los hooks registran BHWC y el código transpone a BCHW para visualizar. Los recortes conservan el canal 0 y sus rangos originales.

**Evidencia:** [teardown_sam2.md](../project/teardown_sam2.md), [stage4_sam_figure_from_notebook.png](../project/evidence/e2/colab_run_20261030T013250Z/derived_from_notebooks/stage4_sam_figure_from_notebook.png).

### 7. Campo receptivo efectivo

**Expone:** Eliezer. **Tiempo:** 05:05–06:00.

**Texto sugerido**

El campo receptivo efectivo lo aproximamos con el gradiente de una salida del encoder respecto a la entrada normalizada. Tomamos el canal cero en la posición central, usamos el valor absoluto del gradiente y sumamos RGB. Normalizamos la masa a uno. En dos chips de Ghana del régimen que excluye Paraguay, aproximadamente el 38 y el 37 por ciento de esa masa queda fuera de la mitad central de la imagen. Hay sensibilidad a regiones distantes. Pero este mapa depende del chip, el canal y la posición seleccionados. No es el campo receptivo teórico ni una explicación causal de la máscara final.

**Al señalar la diapositiva:** Señalar el mapa de gradiente y después su superposición sobre RGB.

**Detalle para la defensa:** la figura extraída de E2 corresponde al régimen `leave_paraguay_out`; los porcentajes mostrados corresponden a ese mismo régimen. Ambos regímenes usan los mismos dos chips train de Ghana para esta disección. Las masas exactas del excluido son 38.1874% y 36.5307%; en official son 38.1625% y 36.5267%. Esta comparación describe sensibilidad y no evalúa generalización a Paraguay.

**Evidencia:** [teardown_sam2.md](../project/teardown_sam2.md), [stage4_sam_figure_from_notebook.png](../project/evidence/e2/colab_run_20261030T013250Z/derived_from_notebooks/stage4_sam_figure_from_notebook.png).

### 8. Latencia y memoria en una Tesla T4

**Expone:** Eliezer. **Tiempo:** 06:00–06:50.

**Texto sugerido**

La ruta completa de imagen tarda una mediana de 152.6 milisegundos en el régimen oficial y 154.9 en el régimen que excluye Paraguay. El pico de memoria asignada es 652.7 MiB. Medimos con batch uno, float32, cinco calentamientos y veinte repeticiones, sincronizando CUDA antes y después. El tiempo excluye lectura de archivos, transferencias y carga de pesos. Además, es una ruta con dos puntos fijos, no la ejecución de todos los componentes de un chip. Los tiempos aislados por módulo no se pueden sumar porque son replays distintos.

**Cambio de expositor:** «Con el costo y los módulos identificados, Roberto presentará la hipótesis y el ajuste que hicimos.»

**Evidencia:** [teardown_sam2.md](../project/teardown_sam2.md), [stage4_table_2_from_notebook.csv](../project/evidence/e2/colab_run_20261030T013250Z/derived_from_notebooks/stage4_table_2_from_notebook.csv).

### 9. Hipótesis previa e intervención

**Expone:** Roberto. **Tiempo:** 06:50–07:45.

**Texto sugerido**

La hipótesis registrada antes de entrenar fue que ajustar solamente el mask decoder aumentaría la IoU agrupada de agua, usando los mismos puntos, chips y evaluador. La motivación es que el baseline sobresegmentaba: recuperaba mucha agua a costa de muchas falsas detecciones. Esperábamos que el decoder redujera falsos positivos. La refutación prefijada ocurre si la IoU de la mejor época no supera al congelado. Es un criterio de este experimento exploratorio, y elegir la mejor época con la misma validación introduce optimismo. La hipótesis no predice una magnitud ni generalización a otros eventos.

**Evidencia:** [e2_intervention.yaml](../../configs/e2_intervention.yaml), [cuaderno_decisiones.md](../project/cuaderno_decisiones.md).

### 10. Cómo entrenamos el decoder

**Expone:** Roberto. **Tiempo:** 07:45–08:45.

**Texto sugerido**

La API de inferencia desactiva los gradientes, así que el entrenamiento recorre explícitamente el prompt encoder y el decoder. Las características de imagen quedan desacopladas del grafo y los encoders permanecen congelados. Para cada componente elegimos una de las tres máscaras por IoU predicho. La unión usa el máximo de los logits por píxel, y sobre esa unión calculamos BCE más Dice en píxeles válidos. Una implementación en dos pasadas reduce memoria y conserva el gradiente de esa unión. Hay un límite importante: las cabezas de IoU y presencia de objeto no reciben gradiente con esta pérdida. Su selección podría desalinearse al cambiar las máscaras.

**Detalle para la defensa:** `argmax` del IoU predicho está desacoplado del gradiente. BCE+Dice supervisa la máscara seleccionada, no el puntaje de IoU ni la presencia de objeto. El máximo por píxel dirige el gradiente al componente ganador. La implementación de dos pasadas se contrastó con autograd directo en pruebas del proyecto.

**Evidencia:** [intervention.py](../../src/auto_prompt_sam2/intervention.py), [e2_intervention.yaml](../../configs/e2_intervention.yaml), [cuaderno_decisiones.md](../project/cuaderno_decisiones.md).

### 11. Protocolo experimental fijado

**Expone:** Roberto. **Tiempo:** 08:45–09:30.

**Texto sugerido**

La ejecución usa una Tesla T4, float32 y semilla 42. Elegimos AdamW con tasa cinco por diez a la menos cinco y acumulación de cuatro chips. El máximo era cinco épocas, con paciencia tres. De los treinta chips de entrenamiento, cuatro no tienen componentes de agua y no aportan gradiente. Cada época tuvo siete pasos del optimizador. La auditoría confirmó que los bloques congelados conservaron su hash y el decoder cambió. Esto respalda que el cambio de pesos quedó dentro del módulo que queríamos intervenir.

**Evidencia:** [e2_intervention.yaml](../../configs/e2_intervention.yaml), [report.json](../project/evidence/e2/colab_run_20261030T013250Z/run/report.json).

### 12. La IoU mejora y el recall disminuye

**Expone:** Roberto. **Tiempo:** 09:30–10:35.

**Texto sugerido**

En los mismos cuarenta chips, la IoU agrupada pasa de 0.1704 a 0.3081. Son 13.77 puntos porcentuales de diferencia absoluta. La precisión aumenta de 0.1715 a 0.3988, pero el recall cae de 0.9638 a 0.5752. El modelo reduce falsas detecciones y también pierde agua real. El área agregada de falsos positivos baja alrededor del 82 por ciento. Por tanto, la mejora de IoU no significa que detectemos más agua. Estos resultados corresponden a la mejor época, la primera. El baseline completo de 89 chips tiene otra IoU y no se usa como control de esta comparación.

**Al señalar la diapositiva:** Comparar cada pareja de barras y enfatizar la caída de recall.

**Evidencia:** [report.json](../project/evidence/e2/colab_run_20261030T013250Z/run/report.json), [resultados_fase1.md](../project/resultados_fase1.md).

### 13. La mejor época fue la primera

**Expone:** Roberto. **Tiempo:** 10:35–11:35.

**Texto sugerido**

La curva muestra una advertencia importante. La pérdida de entrenamiento baja en las cuatro épocas, pero la IoU de validación cae después de la primera y termina en 0.1499, por debajo del congelado. La paciencia tres detiene el entrenamiento al terminar la cuarta época. Esto es compatible con sobreajuste, con una tasa de aprendizaje demasiado alta o con desalineación del puntaje de selección. No hicimos las ablaciones necesarias para identificar la causa. Conservamos la mejor y la última época para mostrar ambos resultados. El criterio de refutación no se cumple en este subconjunto, pero tampoco demuestra robustez.

**Cambio de expositor:** «Eliezer mostrará ahora cómo se ve este intercambio en las imágenes y cómo lo registramos en GitHub.»

**Evidencia:** [history.csv](../project/evidence/e2/colab_run_20261030T013250Z/run/history.csv), [cuaderno_decisiones.md](../project/cuaderno_decisiones.md).

### 14. Pakistan: menos falsos positivos

**Expone:** Eliezer. **Tiempo:** 11:35–12:30.

**Texto sugerido**

En este chip de Pakistán, el baseline cubre una región muy amplia con falsos positivos, que aparecen en rojo. Después del ajuste, gran parte de esa región desaparece y la IoU sube de 0.150 a 0.399. Sin embargo, también aparecen falsos negativos en azul: hay agua real que el modelo deja de detectar. El ejemplo ilustra el intercambio que vimos en las métricas agregadas. Los paneles incluyen la imagen RGB, la etiqueta y los puntos, por lo que podemos relacionar el resultado con la entrada y la asistencia recibida.

**Al señalar la diapositiva:** Verde = acierto de agua; rojo = falso positivo; azul = agua omitida.

**Orden de lectura de la figura:** RGB, etiqueta manual, puntos simulados, decoder congelado y decoder ajustado. En la etiqueta, blanco es agua y gris es inválido. En las predicciones, verde es TP, rojo es FP y azul es FN.

**Evidencia:** [Pakistan_1027214.png](../project/evidence/e2/colab_run_20261030T013250Z/run/examples/Pakistan_1027214.png), [per_chip_val.csv](../project/evidence/e2/colab_run_20261030T013250Z/run/per_chip_val.csv), [cuaderno_decisiones.md](../project/cuaderno_decisiones.md).

### 15. India: el ajuste pierde agua real

**Expone:** Eliezer. **Tiempo:** 12:30–13:25.

**Texto sugerido**

En India el baseline ya cubría bastante bien el agua. Al ajustar, desaparecen aciertos verdes y aparecen omisiones azules. La IoU baja de 0.755 a 0.550. En los 33 chips con IoU individual definida, 28 mejoran y cinco empeoran. Los siete sin agua tienen predicción vacía por construcción. Elegimos estos dos ejemplos para explicar conductas opuestas, y el repositorio conserva diez, uno por evento según una regla previa. El peor deterioro es Nigeria_984831, que pasa de 0.919 a 0.401, pero no tiene una imagen en ese conjunto de ejemplos.

**Al señalar la diapositiva:** Verde = acierto de agua; rojo = falso positivo; azul = agua omitida.

**Evidencia:** [India_1018317.png](../project/evidence/e2/colab_run_20261030T013250Z/run/examples/India_1018317.png), [per_chip_val.csv](../project/evidence/e2/colab_run_20261030T013250Z/run/per_chip_val.csv), [cuaderno_decisiones.md](../project/cuaderno_decisiones.md).

### 16. Cuaderno de decisiones en GitHub

**Expone:** Eliezer. **Tiempo:** 13:25–14:25.

**Texto sugerido**

El cuaderno registra las decisiones y sus motivos, no solo los resultados finales. Primero documentamos los splits y el preprocesamiento aprendido únicamente con train. Luego fijamos la referencia asistida y el evaluador. Antes de entrenar registramos la hipótesis, la intervención mínima y el criterio de refutación. Después añadimos las métricas, los casos que empeoran y las explicaciones alternativas. El cierre del tres de noviembre conserva el alcance realmente ejecutado. El registro también declara asistencia de Claude y Codex para código, auditoría y documentación. Nosotros debemos comprender y defender las decisiones y las cifras.

**Recorrido opcional de 20–30 s:** abrir el cuaderno, localizar la entrada del 29 de octubre de intervención E2, mostrar la hipótesis anterior al entrenamiento y bajar a los resultados del 30 de octubre. Si no hay conexión, usar el archivo local. El corte documental revisado es el commit `e6929935fed731e6e2d369270b9b4528df2eb2e7`; los enlaces de las diapositivas apuntan a `main` y pueden evolucionar.

**Evidencia:** [cuaderno_decisiones.md](../project/cuaderno_decisiones.md).

### 17. Trazabilidad y reproducción

**Expone:** Eliezer. **Tiempo:** 14:25–15:20.

**Texto sugerido**

El repositorio permite seguir el recorrido desde los dos notebooks hasta la configuración, el código de la intervención y la evidencia del run. El código experimental está identificado por el commit f399047. Conservamos el informe, la historia de entrenamiento, las métricas por chip, los puntos y las predicciones. Los checkpoints están en Drive con hashes documentados. La recuperación repitió las métricas y las máscaras sin diferencias. El límite es que esa recuperación ocurrió en un proceso nuevo de la misma sesión de Colab: no acredita reiniciar todo el runtime. La semilla y los avisos de no determinismo también quedan registrados.

**Al señalar la diapositiva:** Si hay tiempo, abrir el índice de evidencia y mostrar report.json.

**Cambio de expositor:** «Roberto, cerremos con lo que permite concluir este experimento y lo que todavía queda fuera de su alcance.»

**Evidencia:** [README.md](../project/evidence/e2/colab_run_20261030T013250Z/README.md), [report.json](../project/evidence/e2/colab_run_20261030T013250Z/run/report.json), [cuaderno_decisiones.md](../project/cuaderno_decisiones.md).

### 18. Conclusión y límites del entregable

**Expone:** Roberto. **Tiempo:** 15:20–16:20.

**Texto sugerido**

La disección muestra que el encoder concentra el cómputo contado y que el decoder es un punto de intervención pequeño. El ajuste preliminar mejora la IoU en esta validación, reduciendo falsas detecciones a costa de perder agua real. La degradación posterior muestra que la mejora es frágil. Con una sola semilla, pocos chips y selección de época sobre la misma validación, no afirmamos generalización. Tampoco medimos todavía el costo de sustituir los prompts asistidos por automáticos: DINOv3 no intervino en este ajuste. Como continuidad proponemos una fuente de prompts disponible para la demo y evaluación separada. La consulta de licencia del dataset sigue pendiente según el registro del proyecto.

**Cierre oral:** «Con esto cerramos la disección y la intervención preliminar. Podemos revisar las métricas, las imágenes o las decisiones del repositorio según sus preguntas».

**Evidencia:** [resultados_fase1.md](../project/resultados_fase1.md), [cuaderno_decisiones.md](../project/cuaderno_decisiones.md), [accesos_y_licencias.md](../project/accesos_y_licencias.md).

### 19. Respaldo: métricas y comparabilidad

**Expone:** Ambos. **Tiempo:** Respaldo, sin tiempo fijo.

**Texto sugerido**

Usar si preguntan por las cifras exactas o por la línea base. La IoU agrupada suma TP, FP y FN antes de dividir. No coincide con el promedio de las IoU de cada chip. El valor de 0.1580 pertenece al baseline en 89 chips, mientras que 0.1704 corresponde al control en los 40 chips de E2. La comparación de intervención usa estos mismos 40 chips en ambos brazos.

**Evidencia:** [report.json](../project/evidence/e2/colab_run_20261030T013250Z/run/report.json), [resultados_fase1.md](../project/resultados_fase1.md).

### 20. Respaldo: fuentes de la exposición

**Expone:** Ambos. **Tiempo:** Respaldo, sin tiempo fijo.

**Texto sugerido**

Fuentes principales para localizar la evidencia. Cada diapositiva incluye las rutas correspondientes en sus notas. Los enlaces a GitHub siguen la rama main, mientras que el run conserva el commit experimental. Las imágenes son derivados de Sen1Floods11 ya conservados en el repositorio. La consulta de licencia sigue pendiente y no debe confundirse disponibilidad con permiso adicional de redistribución.

**Evidencia:** [teardown_sam2.md](../project/teardown_sam2.md), [resultados_fase1.md](../project/resultados_fase1.md), [cuaderno_decisiones.md](../project/cuaderno_decisiones.md), [README.md](../project/evidence/e2/colab_run_20261030T013250Z/README.md), [tarea_final_deep_learning.md](../course/tarea_final_deep_learning.md).

## Preguntas probables para ambos

**¿Ya segmenta sin prompts humanos?**

E2 usa puntos simulados desde etiquetas. Esto controla la entrada al decoder, pero requiere información que no existe en una imagen nueva sin etiquetar. No se implementó ni evaluó el flujo automático DINOv3–SAM en este experimento.

**¿Dónde está DINOv3?**

Existe una disección previa de DINOv3 en la etapa 4. No generó puntos ni participó en el ajuste E2. No debemos atribuirle las métricas del decoder.

**¿Por qué ajustar el decoder?**

Tiene 4 215 109 parámetros, el 10.82% del modelo, y 1.23% de los FLOPs contados. Es una intervención acotada para estudiar adaptación manteniendo fijos los encoders. Eso no implica una reducción medida del tiempo total de inferencia.

**¿Cómo verificaron que solo cambió ese bloque?**

El informe registra hashes iguales en los bloques congelados, hash distinto del decoder y auditoría de gradientes del primer paso sin gradientes fuera del decoder. Que un parámetro tenga requires_grad no garantiza que reciba gradiente con la pérdida usada.

**¿Por qué la IoU sube si el recall baja?**

IoU = TP/(TP+FP+FN). El ajuste reduce fuertemente FP y también reduce TP. El balance produce una IoU mayor en el agregado. Precisión sube de 0.1715 a 0.3988 y recall baja de 0.9638 a 0.5752. No significa detectar más agua.

**¿Qué significa una mejora de 13.77 puntos porcentuales?**

(0.3080519 − 0.1703785) × 100 = 13.7673 puntos porcentuales. Es una diferencia absoluta entre proporciones, distinta de un incremento relativo de 13.77%.

**¿La hipótesis quedó demostrada?**

La mejor época supera al control y no cumple el criterio prefijado de refutación en este subconjunto. Una semilla, pocos chips y selección sobre la misma validación no permiten afirmar una ley general ni significancia estadística.

**¿La caída posterior demuestra sobreajuste?**

Es compatible con sobreajuste, pero también con una tasa inadecuada, desalineación del puntaje de IoU o ruido de validación. No hicimos la ablación que separe esas causas. La última época queda por debajo del congelado.

**¿Cómo se calcula la IoU agrupada?**

Primero sumamos TP, FP y FN de todos los píxeles válidos y después calculamos la razón. Una media de IoU por chip da el mismo peso a cada chip, y una media por evento da el mismo peso a cada evento. Responden preguntas distintas.

**¿Por qué hay 40 chips y solo 33 IoU individuales?**

Siete chips no tienen agua y reciben predicción vacía por construcción. Su unión agua real/predicha es cero, por lo que la IoU individual es indefinida, no uno. No prueban que un sistema automático evite falsos positivos en escenas secas.

**¿Los 296.05 GFLOPs cubren todo?**

No. Son operaciones con fórmula registrada en FlopCounterMode para forward, batch 1 y dos puntos fijos. Las operaciones no cubiertas no tienen costo real cero. Los tiempos por bloque son replays aislados y no se suman.

**¿152.6 ms es el tiempo de segmentar todo un chip de E2?**

Es la ruta tensorial estática del perfilado con dos puntos fijos y entradas precargadas. La política de E2 puede usar muchos componentes por chip. El perfil excluye E/S y transferencias, por lo que no es latencia extremo a extremo de esa política ni de un sistema automático.

**¿El ERF explica dónde ve agua el modelo?**

Mide sensibilidad local de una activación del encoder a la entrada normalizada. Depende de un canal, una posición y un chip. No es un mapa semántico de agua ni una prueba causal de la máscara.

**¿El régimen official es una prueba de generalización geográfica?**

No. Conserva el split oficial y registra vecindad entre splits. La intervención se evalúa en validación official. Test, Bolivia y Paraguay reservado no tienen una evaluación final de E2. Los chips Paraguay de official no son el conjunto Paraguay reservado del otro régimen.

**¿La recuperación prueba que un reinicio de Colab reproduce todo?**

En este run E2 se verificó un proceso nuevo en la misma sesión: ΔIoU = 0, confusiones idénticas y cero píxeles de desacuerdo. No hay evidencia de reinicio completo del runtime para esa recuperación. Hay antecedentes de otras etapas que no deben trasladarse al run E2.

**¿Qué hicieron los asistentes de IA?**

El cuaderno documenta asistencia de Claude y Codex en implementación, pruebas, auditoría y redacción. Los experimentos conservados se ejecutaron en Colab por el equipo. Codex preparó esta presentación y este guion a partir de evidencia existente, sin reentrenar. Ambos expositores deben comprender cualquier parte.

**¿Cuál sería el siguiente experimento?**

Como propuesta, separar selección y evaluación, contrastar otra tasa de aprendizaje o una política de selección de máscara, y definir prompts disponibles sin etiquetas si se busca automatización. Son ampliaciones futuras, no resultados obtenidos.

**¿Cuál es la situación de la licencia?**

El registro del proyecto mantiene pendiente la respuesta del publicador de Sen1Floods11. El cierre académico de fase 1 no cambia los términos del dataset. Las imágenes utilizadas son derivados conservados en la evidencia del proyecto.

## Glosario breve

- **Chip:** recorte de imagen satelital; aquí, 512 × 512 píxeles.
- **Prompt:** señal de entrada que guía la segmentación, en este caso puntos positivos y negativos.
- **Embedding / features:** representación numérica intermedia de la imagen o del prompt.
- **FPN:** integración de mapas de características de distintas resoluciones.
- **Logit:** valor antes de convertirlo en probabilidad. Umbral de logit 0 equivale a probabilidad 0.5.
- **TP / FP / FN:** agua acertada / falsa detección de agua / agua omitida.
- **BCE:** entropía cruzada binaria por píxel. **Dice loss:** penalización por falta de solapamiento.
- **ERF:** campo receptivo efectivo estimado mediante gradientes.
- **MiB:** mebibyte, 1 048 576 bytes. **GFLOPs:** miles de millones de operaciones de punto flotante.

## Si disponen de menos tiempo

Para una versión de unos 10–12 minutos, resumir las diapositivas 2–3 y 10–11, dedicar 30 segundos a cada figura del teardown y mantener completas las explicaciones de 12–15 y 18. No omitir que los puntos provienen de etiquetas, que el recall cae y que la mejor época se eligió sobre la misma validación. No abrir GitHub durante la exposición breve.

## Procedencia de este material

El guion y las diapositivas sintetizan los documentos del proyecto, las celdas y salidas de E2_01/E2_02, la configuración de intervención, el código del decoder y los artefactos del run `20261030T013250Z_2a795a89d5_f9fbd8b0`. No incorporan nuevas mediciones. El código experimental es `f399047464b89a3ed18972f8dc7f1b1364d83a44`. Las fuentes concretas aparecen bajo cada diapositiva y en las notas del PowerPoint.
