# FACULTAD DE CIENCIAS E INGENIERÍA - INGENIERÍA INFORMÁTICA
**Curso:** Deep Learning (1INF52) Semestre 2026-2  
**Docente:** Dr. César A. Beltrán Castañón  
**Entregable E1:** Propuesta de tema — 7 de octubre de 2026  
**Integrantes:** 

## Título
**El costo del prompt: segmentación de agua de inundación en Sentinel-2 sin intervención humana, con prompts generados desde DINOv3**

---

## 1. Pregunta de investigación
¿Cuánta calidad de segmentación se pierde al reemplazar el prompt humano de SAM 2.1 por prompts automáticos (texto, puntos derivados de features de DINOv3, o ningún prompt)? ¿Cómo cambia esa pérdida en eventos de inundación de Bolivia y Paraguay, fuera del dominio de imágenes naturales con el que SAM fue entrenado?

---

## 2. Tema del banco, modelo y referencias

| Criterio | Detalle |
| :--- | :--- |
| **Tema del banco** | "El costo del prompt en SAM. Segmentación automática (sin prompt) en un dominio nuevo." |
| **Modelo core (Meta)** | SAM 2.1, familia `facebook/sam2.1-hiera-` (variante tiny o small por restricción de cómputo). |
| **Modelos de apoyo** | DINOv3 ViT-L preentrenado en imágenes satelitales (`facebook/dinov3-vitl16-pretrain-sat493m`) para generar los puntos-prompt. SAM 3 (`facebook/sam3`) solo como fila de comparación con prompt de texto, sin entrenamiento. |
| **Artículo de partida** | SAM 2: Segment Anything in Images and Videos (Ravi et al., 2024). |
| **Referencias adicionales** | DINOv3 (arXiv 2508.10104). Sen1Floods11 (Bonafilia et al., CVPRW 2020). Trabajos previos que combinan DINO con SAM (Matcher, PerSAM): el equipo no reclama novedad en el método; la contribución es el análisis sistemático del costo del prompt y su comportamiento bajo shift de dominio. |

---

## 3. Dataset
Sen1Floods11 (bucket `gs://sen1floods11`). Se usará solo la parte hand-labeled: 446 chips de 512x512 píxeles a 10 m con etiquetas agua/no-agua e imágenes Sentinel-2 L1C. **Protocolo:** split oficial (252 entrenamiento / 89 validación / 90 prueba) con los 15 chips de Bolivia reservados como conjunto de transferencia; resultados reportados por evento (Bolivia, Paraguay, Colombia y el resto). **Métrica principal:** IoU de la clase agua.

---

## 4. Plan por fases

| Fase | Contenido |
| :--- | :--- |
| **A. Disección del modelo base** | Reproducción de SAM 2.1 con prompt humano simulado (puntos tomados de la etiqueta). Conteo de parámetros y FLOPs por bloque (image encoder, prompt encoder, mask decoder), visualización de features intermedias, mapa del campo receptivo, latencia y memoria. |
| **B. Intervención en el núcleo** | Interfaz prompt encoder mask decoder de SAM 2.1: reemplazo del prompt humano por prompts automáticos y ajuste del mask decoder. Tres ablaciones controladas (tabla siguiente). |
| **C. Estrés y honestidad** | Evaluación en los eventos de Bolivia y Paraguay frente al resto. Análisis de fallos: agua turbia, vegetación inundada, sombras de nubes y relieve; conexión con transfer learning y domain adaptation. Se reportará explícitamente lo que no funcione. |

### Ablaciones de la Fase B
*(Una hipótesis, un cambio, una medición; las hipótesis se contrastarán con el resultado en el cuaderno de decisiones):*

| ID | Cambio | Hipótesis previa | Medición |
| :---: | :--- | :--- | :--- |
| **A1** | Tipo de prompt: texto (SAM 3), puntos auto-generados desde DINOv3, caja del chip completo, sin prompt (malla automática). | Los puntos desde DINOv3 superarán a la malla sin prompt porque aportan información semántica de dónde hay agua; el texto quedará por debajo por tratarse de imágenes no naturales. | IoU agua, global y por evento. |
| **A2** | Número de puntos auto-generados: $k = 1, 3, 5, 10$. | El IoU mejora hasta saturar y luego los puntos adicionales añaden falsos positivos. | IoU agua vs $k$; latencia. |
| **A3** | Mask decoder congelado vs afinado (solo el decoder, con los chips de entrenamiento). | Afinar mejora el IoU en eventos vistos, pero la ganancia será menor en Bolivia por el shift. | IoU agua en prueba y en Bolivia; parámetros entrenados. |

---

## 5. Demo y resultados visuales
* Máscaras de agua superpuestas con los puntos-prompt dibujados, y comparación lado a lado de las cuatro estrategias de prompt.
* Mapa de features intermedias y de similitud de DINOv3 que explica dónde se generan los puntos.
* Demo reproducible (Gradio/Hugging Face Space): se sube un chip Sentinel-2 y se obtiene la máscara de agua. Se puede agregar un chip de Perú descargado de Google Earth Engine, mostrado de forma solo cualitativa por no tener etiquetas.

---

## 6. Riesgos y mitigación

| Riesgo | Mitigación |
| :--- | :--- |
| Sen1Floods11 etiqueta superficie de agua, incluida el agua permanente; no distingue inundación de río. | Declararlo en el informe; no afirmar discriminación inundación vs agua permanente. |
| Los chips tienen 13 bandas; SAM espera RGB. | Documentar la conversión (RGB o falso color) en el cuaderno de decisiones y fijarla en el repositorio. |
| Pocos chips en el conjunto de transferencia (15 de Bolivia). | Reportar por chip y por evento, sin sobreafirmar; no se exige significancia con múltiples semillas. |
| DINOv3 SAT-493M se entrenó con imágenes de mayor resolución que Sentinel-2 (10 m). | Declararlo como confusor en el análisis de fallos. |
| Acceso a pesos con términos de uso (DINOv3, SAM). | Solicitar acceso en Hugging Face desde la semana 1 y verificar licencias. |

---

## 7. Cumplimiento del enunciado y cronograma
* Reproducción fiel de la línea base: SAM 2.1 con prompt humano simulado (Fase A).
* Una intervención de núcleo con ablación controlada: interfaz prompt encoder mask decoder, con A1-A3 (Fase B).
* Shift de dominio y análisis de fallos honesto: imágenes naturales $\rightarrow$ Sentinel-2, eventos de Bolivia y Paraguay (Fase C).
* Demo funcional, repositorio reproducible y cuaderno de decisiones con semillas fijas, entorno documentado y registro de dónde se usó asistencia de IA. Toda figura, dataset o modelo de terceros se citará con su licencia.

| Entregable | Fecha | Plan del equipo |
| :--- | :--- | :--- |
| **E1 Propuesta de tema** | miércoles 7 de octubre de 2026 | Esta propuesta. |
| **E2 Teardown + intervención preliminar + cuaderno (GitHub)** | miércoles 21 de octubre de 2026 | Semana 1: accesos, datos y Fase A.<br>Semana 2: A1 preliminar y cuaderno de decisiones. |
| **E3 Entrega de trabajo e informe final (4-8 pág., IEEE/LNCS)** | miércoles 11 de noviembre de 2026 | A2, A3, Fase C y redacción. |
| **E4 Póster en WAIMLAP** | 18-20 de noviembre de 2026 | Póster y demo. |

---

## 8. Pendientes de verificación (semana 1)
Licencia de Sen1Floods11, ID exacto de la variante de SAM 2.1 y términos de acceso a DINOv3 y SAM en Hugging Face.

> **Nota:** Como el tema *"El costo del prompt en SAM"* admite máximo 2 equipos con artículos distintos, se solicita confirmar disponibilidad del tema con el docente.