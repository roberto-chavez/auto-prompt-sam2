# FACULTAD DE CIENCIAS E INGENIERÍA
## INGENIERÍA INFORMÁTICA

### TAREA ACADÉMICA FINAL DE CURSO
**Proyecto de investigación y aplicación en Deep Learning**

| Información General | Detalle |
| :--- | :--- |
| **Curso** | Deep Learning |
| **Clave** | 1INF52 |
| **Programa** | Ingeniería Informática |
| **Créditos** | 03 |
| **Semestre** | 2026-2 |
| **Docente** | Dr. César A. Beltrán Castañón |
| **Modalidad** | Grupal |

---

## 1. Presentación y objetivo

La tarea académica final permite aplicar el conocimiento del curso al diseño, implementación y evaluación de algoritmos de aprendizaje profundo sobre un problema del estado del arte. El objetivo es abrir el núcleo de una arquitectura, intervenirlo de forma controlada y explicar su comportamiento. La evaluación se centra en: elegir, ablacionar, diagnosticar y defender decisiones de diseño.

---

## 2. El proyecto: ciclo de tres fases

El proyecto parte de un artículo/modelo de referencia del estado del arte y de un conjunto de datos de reto, y recorre tres fases:

* **Fase A: Disección instrumentada del modelo base**  
  No basta con ejecutar el modelo: hay que abrirlo. Se exige un *architecture teardown*: conteo de parámetros y FLOPs por bloque, visualización de características intermedias, mapa del campo receptivo, y medición de latencia/memoria. El objetivo es entender por qué el modelo tiene la forma que tiene.

* **Fase B: Intervención en el núcleo**  
  El corazón del reto: modificar una parte del núcleo y medir el efecto con ablaciones controladas, cada una con una hipótesis previa. Ejemplos: cambiar el *backbone* de un detector (ResNet $\rightarrow$ DINOv2), variar el *patch size* de un ViT, sustituir la conexión residual por concatenación densa, cambiar la función de pérdida (Dice / cross-entropy / focal), o podar y cuantizar midiendo el compromiso precisión-tamaño.

* **Fase C: Estrés y honestidad**  
  Se evalúa el modelo en un dominio de distribución distinta a la de su preentrenamiento. Aquí el modelo «que funcionaba» suele romperse; el equipo debe diagnosticar por qué, conectando con *transfer learning* y *domain adaptation*, y reportar explícitamente lo que no funcionó.

---

## 3. Alcance y expectativas

El objetivo es demostrar ingeniería sólida y comprensión aplicada. El equipo debe recorrer las tres fases con la profundidad siguiente:

* Reproducir fielmente la línea base de la referencia elegida.
* Realizar una intervención de núcleo bien justificada, con su ablación controlada (una hipótesis, un cambio, una medición).
* Evaluar bajo *shift* de dominio (Fase C) y presentar un análisis de fallos honesto.
* Entregar una demo funcional y reproducible y el cuaderno de decisiones.

| Sí se exige | No se exige (pero suma) |
| :--- | :--- |
| Reproducción correcta + una intervención con ablación. | Novedad publicable o comparación exhaustiva con el SOTA. |
| Análisis de fallos y honestidad en Fase C. | Significancia estadística con múltiples semillas. |
| Defensa oral individual de las decisiones. | Estado del arte (*related work*) extenso. |

---

## 4. Recursos: modelos y datasets

Como referencia general de implementaciones abiertas se mantiene el directorio **Papers With Code** ([paperswithcode.com/sota](https://paperswithcode.com/sota)). Gracias a la colaboración con Meta AI, los equipos disponen de acceso a sus modelos ([huggingface.co/facebook](https://huggingface.co/facebook)) y a sus conjuntos de datos ([ai.meta.com/ai-for-good/datasets](https://ai.meta.com/ai-for-good/datasets)). Se recomienda emparejar un modelo core con un dataset de reto que lo saque de su dominio de preentrenamiento:

| Modelo core (Meta) | Dataset de reto | Foco del núcleo |
| :--- | :--- | :--- |
| **SAM / SAM 2** | SA-1B dominio médico o satelital | Segmentación promptable: adaptar sin prompts humanos; analizar el *mask decoder*. |
| **DINOv2 / DINOv3** | Canopy Height Maps satélite | Features auto-supervisadas: *linear probing* vs *fine-tuning* fuera de ImageNet. |
| **DETR / RF-DETR** | Detección (COCO) dominio propio | Predicción de conjunto (Hungarian matching) vs anchors; *backbone swap*. |
| **Detección de deepfakes** | DFDC (Deepfake Detection Challenge) | Generalización *cross-manipulation*; robustez. |
| **Backbones varios** | Datasets AI-for-Good (población, movilidad) | *Shift* de dominio en problemas socialmente relevantes; uso responsable. |

*Cada equipo debe verificar y respetar la licencia y los términos de uso de modelos y datos. Los conjuntos socialmente sensibles exigen especial cuidado con la privacidad y el sesgo.*

---

## 5. Banco de proyectos-reto

Los temas se formulan como preguntas de investigación. Cada tema puede ser elegido por un máximo de 2 equipos, que deberán partir de un artículo/modelo distinto:

* **¿Sirven las features de DINOv2 fuera de ImageNet?** *Linear probing* sobre datos satelitales frente a una CNN entrenada desde cero.
* **El costo del prompt en SAM.** Segmentación automática (sin prompt) en un dominio nuevo.
* **Predicción de conjunto vs anchors.** Comparar DETR y YOLO en un dataset propio.
* **Backbone swap.** DETR con backbone ResNet vs DINOv2.
* **Robustez de detectores de deepfakes.** Entrenar en una manipulación de DFDC y evaluar en otra.
* **Compresión con honestidad.** Cuantizar/podar un ViT o SAM y trazar la curva precisión-tamaño.

*Se admiten propuestas fuera de este banco previa aprobación docente, siempre que respeten el ciclo de tres fases.*

---

## 6. Entregables y cronograma

| Entregable | Fecha límite | Puntaje |
| :--- | :--- | :--- |
| **E1** Propuesta de tema | Miércoles 07 de octubre de 2026 | 2 puntos |
| **E2** Teardown del modelo base + intervención preliminar + cuaderno de decisiones (GitHub). | Miércoles 21 de octubre de 2026 | 8 puntos |
| **E3** Entrega de trabajo | Miércoles 11 de noviembre de 2026 | 2 puntos |
| **E4** Presentación en WAIMLAP: póster | 18-20 de noviembre de 2026 | 4 puntos |
| **E3** Entrega de informe final (4-8 pág., IEEE/LNCS) revisado con la retroalimentación del workshop. | Miércoles 11 de noviembre de 2026 | 4 puntos |
| **Total** | | **20 puntos** |

---

## 7. Política de uso de asistentes de IA

* El uso de agentes de código y asistentes de IA está permitido y se fomenta como herramienta de trabajo.
* El cuaderno de decisiones (*design log*) debe registrar cada decisión arquitectónica y su justificación, e indicar dónde se usó asistencia de IA.
* Antes de cada ablación, el equipo escribe qué espera que ocurra y por qué; luego contrasta con el resultado. Se evalúa el modelo mental, no la ejecución.
* La defensa oral individual es la salvaguarda central: cada integrante responde preguntas sobre cualquier decisión del código y del diseño. No poder explicar el propio trabajo penaliza la nota individual.
* **Delegar la ejecución es válido; delegar la comprensión no lo es.**

---

## 8. Formato de entrega

1. **Repositorio en GitHub:** notebooks reproducibles, README con instrucciones, semillas fijadas y entorno documentado.
2. **Documento:** resumen extendido (4-8 pág.) según la plantilla de WAIMLAP.
3. **Presentación en el workshop.**

*Toda figura, dataset o modelo de terceros debe estar correctamente citado y con su licencia respetada.*

---

## 9. Uso responsable de datos

Cuando se usen conjuntos socialmente sensibles (población, movilidad), el resumen debe incluir una declaración breve sobre privacidad, posibles sesgos y uso responsable de los datos.