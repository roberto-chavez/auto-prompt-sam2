# Hoja de ruta de la fase 1: experimentación y entrenamiento en Google Colab

## 1. Propósito, alcance y fuentes

Esta guía organiza la implementación futura de la fase 1 de `auto-prompt-sam2`. El resultado será un protocolo reproducible, notebooks ejecutables en Google Colab, resultados experimentales y artefactos de inferencia conservados para la fase posterior. **Este documento no acredita ejecuciones ni resultados y no implementa código.** La construcción de interfaces, la integración con Gradio y el despliegue en Hugging Face Spaces quedan fuera del alcance.

La pregunta principal es cuánto cambia la IoU de agua al reemplazar puntos humanos simulados con puntos automáticos derivados de DINOv3, y cómo cambia esa diferencia en eventos geográficos excluidos del aprendizaje. La tarea es segmentación de **agua superficial**, incluida agua permanente; no identifica por sí sola el agua atribuible a una inundación.

Fuentes revisadas, con rutas relativas a este documento:

| Fuente | Función en esta guía |
| --- | --- |
| [README](../../README.md), especialmente «Fases de ejecución», «Metodología propuesta» y «Datos y protocolo de evaluación» | Alcance vigente, Colab, separación entre fases 1 y 2, persistencia y entregables. |
| [Bases del curso](../course/tarea_final_deep_learning.md), secciones 2, 3, 6 y 7 | Disección A, intervención B, estrés C, cuaderno de decisiones, hipótesis previas y declaración de asistencia de IA. |
| [Resumen WAIMLAP](../WAIMLAP/Abstract_CostOfThePrompt.md), secciones 2 y 3 | Protocolo experimental detallado, particiones, preprocesamiento, prompts, entrenamiento y métricas. |
| [Póster WAIMLAP](../WAIMLAP/Poster_CostOfThePrompt.pptx), diapositiva 1 | Síntesis del diseño actual y sus limitaciones; no contiene resultados. |
| [Propuesta inicial](draft/propuesta_inicial_deep_learning.md), secciones 2–4 y 8 | Antecedentes, candidatos iniciales para A2 y pendientes de acceso. |
| [Diapositivas E1](draft/El%20Costo%20del%20Prompt%20-%20INF52%20PUCP_slides.pdf), páginas 2–9 | Planteamiento inicial, arquitectura, ablaciones y riesgos. |

Se emplean tres etiquetas:

- **[D] Documentado:** decisión o requisito explícito de las fuentes. Los recuentos se deben verificar posteriormente contra los archivos oficiales.
- **[R] Recomendación:** concreción técnica propuesta por esta hoja de ruta. Debe registrarse en el cuaderno antes de ejecutarse; no se presenta como decisión previa del equipo.
- **[P] Pendiente:** dato o elección que debe resolverse y quedar registrado antes de la etapa dependiente. No autoriza a inventar valores ni resultados.

Para discrepancias, esta guía sigue el alcance del README y el detalle metodológico del resumen WAIMLAP, corroborado por el póster. **[R]** Esta prioridad se adopta para implementar un protocolo coherente; cualquier modificación posterior debe tener una justificación explícita. SAM 3 con texto y la posibilidad de falso color pertenecen al planteamiento inicial: no se incluyen en las ablaciones actuales, que usan SAM 2.1 y RGB. Las afirmaciones sobre errores en las diapositivas iniciales se tratarán como riesgos o hipótesis, no como resultados observados.

Las fases académicas A, B y C se cubren dentro de esta única fase de ejecución. Las fechas de E1–E4 son antecedentes administrativos, no pruebas de finalización. **[P]** Confirmar con el docente la duplicación de E3 y la fecha del informe revisado con retroalimentación de un workshop posterior; esta inconsistencia no modifica el orden técnico de la guía.

## 2. Contrato experimental común

### 2.1. Particiones y protección frente a fugas

**[D]** Utilizar exclusivamente los 446 chips con anotación manual de Sen1Floods11, Sentinel-2 L1C, 512 × 512 píxeles, 13 bandas y resolución de 10 m. Etiquetas: agua `1`, no agua `0`, inválido `-1`. No añadir ejemplos con etiquetas débiles al conjunto principal.

| Régimen | Entrenamiento | Validación | Prueba de eventos vistos | Transferencia reservada |
| --- | ---: | ---: | ---: | --- |
| `official` [D] | 252 | 89 | 90 | Bolivia: 15 |
| `leave_paraguay_out` [D, recuentos derivados] | 213 = 252 − 39 | 75 = 89 − 14 | 76 = 90 − 14 | Paraguay: 67 = 39 + 14 + 14; Bolivia permanece apartada |

Los 67 chips de Paraguay se reúnen desde las tres particiones oficiales para la evaluación reservada de ese régimen. No se reasignan a entrenamiento ni validación. Los 76 chips restantes son su referencia de eventos vistos. **[R]** Reportar Bolivia principalmente con el régimen oficial; si se desea evaluarla también con el régimen sin Paraguay, declararlo antes del cierre experimental y tratarlo como una comparación adicional independiente.

Reglas que deben implementar todos los notebooks:

1. **[D]** Entrenar probe y decoder solo con entrenamiento. Elegir umbrales, hiperparámetros y checkpoints solo con validación. No unir train y validación para un reentrenamiento final.
2. **[D]** No usar prueba ni Bolivia para seleccionar configuraciones; su evaluación ocurre una sola vez al final. **[R]** Aplicar el mismo cierre a los 67 chips reservados de Paraguay. «Una vez» significa una campaña final con todas las filas predefinidas, no una búsqueda adaptativa de configuraciones.
3. **[R]** Hasta la etapa 8, limitar los conjuntos reservados a inventario, integridad estructural y comprobaciones de pertenencia. No visualizar sus imágenes para diseñar heurísticas, extraer distribuciones, calcular métricas ni revisar sus etiquetas para tomar decisiones.
4. **[D/R]** En `leave_paraguay_out`, recalcular el recorte radiométrico y cualquier estadística aprendida usando solo sus 213 chips, entrenar su propio probe y decoder y calibrar únicamente con sus 75 chips. Reutilizar código y pesos preentrenados originales, nunca pesos, normalizadores o decisiones ajustadas con Paraguay. Mantener el régimen oficial aislado para que sus resultados de Paraguay no influyan en el diseño del régimen excluido.
5. **[R]** Preparar desde el inicio ambos regímenes y un espacio de búsqueda común prefijado. No elegir ese espacio después de observar resultados del régimen oficial. Ejecutar las etapas 2–7 por régimen antes de abrir cualquier conjunto reservado.
6. **[D]** La referencia asistida puede usar etiquetas para producir puntos; se identifica claramente como no desplegable. **[R]** Su acceso a etiquetas debe estar separado del flujo automático y, en evaluación final, no puede cambiar pesos, umbrales o selección de máscaras. La selección multimáscara utiliza IoU **predicha**, nunca IoU real.
7. **[R]** La inferencia automática acepta imagen y metadatos de entrada, sin máscara real. La validez para pérdidas y métricas puede usar `label != -1`; el generador automático solo puede usar validez deducida de la entrada, como nodata o valores no finitos. No utilizar el patrón de píxeles inválidos de la etiqueta como información oculta para producir prompts.
8. **[R]** Verificar IDs únicos, archivos duplicados por hash, pertenencia a eventos e intersecciones entre particiones. Registrar vecindad o solapamiento espacial cuando existan metadatos. No reemplazar el split oficial por un split aleatorio para resolver estos problemas. Ante duplicados incompatibles, bloquear el uso afectado y documentar la incidencia.

**[D]** Las particiones oficiales por chip pueden contener regiones vecinas de un mismo evento. Sus resultados no prueban independencia geográfica; Bolivia y Paraguay excluido sustentan esa evaluación.

### 2.2. Modelos, preprocesamiento y métricas

- **[D]** SAM 2.1 tiny como primera opción; DINOv3 ViT-L/16 satelital congelado, checkpoint propuesto `facebook/dinov3-vitl16-pretrain-sat493m`. **[P]** Fijar ID, revisión y archivo exacto de SAM, revisiones de DINOv3, licencias, acceso y compatibilidad real del entorno. Cambiar de variante es una modificación de protocolo que exige nuevas comparaciones consistentes.
- **[D]** RGB B4/B3/B2, reflectancia dividida por 10 000, recorte superior fijado una vez a partir de entrenamiento y reescalado a [0, 1]. Normalización propia de cada modelo: estadísticas satelitales para DINOv3 y las usadas por el código oficial de SAM 2.1. SAM recibe tamaño 1024. **[P]** Verificar orden de bandas del archivo, estadísticas exactas, tamaño de DINOv3, interpolaciones y ruta de normalización para evitar aplicarla dos veces.
- **[D]** Probe de regresión logística sobre parches DINOv3. Etiqueta de parche inicial: agua si al menos 50% de sus píxeles válidos son agua, no agua si como máximo 5%, resto ignorado. Los umbrales pueden seleccionarse con validación. **[R]** Ignorar parches sin píxeles válidos; registrar cualquier mínimo adicional de cobertura válida y fijarlo sin consultar conjuntos reservados.
- **[D]** Ajustar exclusivamente el mask decoder con BCE + Dice, dejando congelados DINOv3 y los encoders de imagen y prompts de SAM.
- **[D]** Métrica principal: IoU de agua acumulando píxeles válidos. Complementar con media de IoU por evento, mediana e IQR por chip, precisión, recall, Dice/F1 y mIoU de dos clases.

**[R]** Implementar una única biblioteca de métricas con acumuladores TP, FP, FN y TN. Calcular IoU agua = TP/(TP+FP+FN), Dice = 2TP/(2TP+FP+FN), precisión = TP/(TP+FP), recall = TP/(TP+FN) e IoU no agua = TN/(TN+FP+FN). mIoU es la media de las dos IoU definidas; si una clase no tiene denominador, registrar la convención y no imputar silenciosamente un valor perfecto. Los denominadores nulos se guardan como no definidos junto con su motivo. El resultado agregado se calcula sumando confusiones, no promediando IoU de chips. Para la media por evento, agregar primero sus píxeles y después promediar eventos con igual peso, informando exclusiones.

**[D]** Chips sin agua real válida: excluirlos de IoU individual incluso si hay predicciones positivas, mantenerlos en las métricas agregadas y reportar área de falsos positivos. **[R]** Conservar FP en píxeles y área física calculada con la georreferenciación; a 10 × 10 m corresponden 100 m² por píxel. Chips completamente inválidos no aportan confusiones y deben contarse aparte. Validar métricas sobre la malla original 512 × 512 y nunca interpolar etiquetas categóricas con métodos que creen clases nuevas.

**[R]** Definir el costo del prompt como `IoU_referencia_asistida − IoU_automática`, en puntos porcentuales al multiplicar por 100, con mismo régimen, chips, decoder y presupuesto de puntos. El contraste primario usa decoder congelado en ambas ramas; la mejora del decoder se presenta separada mediante A3. Un costo negativo es posible y se reporta sin corregirlo. La referencia asistida es una referencia experimental, no una cota matemática garantizada.

### 2.3. Persistencia, trazabilidad y reanudación en Colab

**[D]** Ningún artefacto necesario para reproducir o reanudar debe depender únicamente de `/content`. **[R]** Usar una raíz persistente configurable, por ejemplo Google Drive, y copiar a disco local solo archivos activos. Las rutas y nombres siguientes son propuestas de implementación, no archivos que ya existan:

```text
notebooks/01_entorno_colab.ipynb ... 09_analisis_y_cierre.ipynb
src/auto_prompt_sam2/                 # Funciones compartidas, sin lógica duplicada en celdas
configs/                            # Configuraciones versionadas y matriz experimental
tests/                              # Validaciones de contratos científicos
docs/project/cuaderno_decisiones.md
<PERSIST_ROOT>/
  raw/                              # Datos originales, inventario y hashes
  base_models/                      # Pesos autorizados o referencias de adquisición
  regimes/<regime>/
    manifests/                      # IDs, eventos, roles y hashes
    preprocessing/                  # Estadísticas y geometría
    cache/                          # Features/embeddings y metadatos de procedencia
    runs/<run_id>/                   # Config, estado, métricas, logs, checkpoints
    final_evaluation/<campaign_id>/  # Predicciones y resultados finales inmutables
  release/<release_id>/              # Artefactos elegidos para reutilización posterior
```

Todo `run_id` conservará configuración resuelta, régimen y hashes de splits, semilla, commit y parche de cambios locales si existieran, versiones instaladas, revisiones y hashes de pesos, hardware, CUDA, precisión, tamaño de entrada, tiempos, estado y dependencias de etapas anteriores. Registrar si el determinismo estricto no es compatible con alguna operación. Una semilla es el alcance documentado; semillas adicionales serían una extensión explícita.

**[R]** Cada caché tendrá clave con hash de datos, preprocesamiento, modelo/revisión, capa, resolución, precisión y régimen. No considerar válido un archivo solo porque existe. Mantener shards por chip o grupo pequeño y un índice de finalización; en particular, un scaler aprendido invalida features transformadas si cambia. Datos brutos y pesos preentrenados inmutables pueden compartirse; separar artefactos ajustados por régimen.

**[R]** Guardar primero archivos temporales, validar contenido/hash y publicar después un marcador de completitud. No asumir que un montaje remoto ofrece escrituras atómicas. Mantener la última versión completa hasta verificar la siguiente. Reanudar solo cuando config, splits y modelos coincidan; de lo contrario iniciar una ejecución nueva, sin sobrescribir resultados.

Para entrenar, guardar `last` para reanudación y `best` seleccionado por validación, además de estados del optimizador, scheduler, escalador AMP, época/paso, mejor métrica, contador de parada temprana, RNG de Python/NumPy/PyTorch CPU/GPU y estado del muestreador. **[R]** Reanudar en límites de época como opción inicial; si se permite mitad de época, conservar cursor, orden de lotes y acumulación de gradientes. Si el solver del probe no admite reanudación exacta, reiniciar solamente ese ajuste desde features persistidas y declararlo.

Registrar cada decisión y experimento en el cuaderno: fecha, responsable, hipótesis previa, configuración, cambios, ejecución, resultados, fallos, interpretación y uso de IA. Ninguna etapa se considera concluida por haber creado un notebook vacío o por alcanzar una IoU esperada.

## 3. Secuencia y dependencias

| Etapa | Resultado que habilita la siguiente | Cobertura académica |
| --- | --- | --- |
| 1. Entorno y persistencia | Entorno reconstruible y acceso verificado | Reproducibilidad transversal |
| 2. Datos y protocolo | Dos manifiestos aislados y preprocesamiento | Control experimental |
| 3. Línea base asistida | Inferencia SAM y métricas verificadas | A |
| 4. Disección instrumentada | Arquitectura y costo computacional medidos | A |
| 5. Prompts automáticos | Probe y generador congelados por régimen | B, cambio de entradas |
| 6. Entrenamiento del decoder | Intervención de núcleo reproducible | B |
| 7. Ablaciones y selección | Configuraciones finales selladas | B |
| 8. Evaluación final y geográfica | Predicciones y métricas finales completas | C |
| 9. Análisis y cierre de artefactos | Evidencia interpretable y paquete reutilizable | C y cierre de fase 1 |

Las etapas 2–7 trabajan con train/validación de cada régimen. La etapa 8 abre los conjuntos reservados únicamente tras completar ambos regímenes. Los prompts siguientes están redactados para una entrega futura a un agente con acceso al repositorio: contienen su ámbito y contratos esenciales y remiten a esta guía para el detalle compartido. No requieren la conversación original. Las rutas previstas pueden adaptarse a una implementación anterior equivalente, documentando el mapeo y evitando duplicaciones.

Después de concluir cada etapa, actualizar el [estado actual de la fase 1](estado_actual_fase1.md) con el progreso realizado, los entregables, las validaciones y los pendientes. Marcarla como concluida solo cuando se cumplan sus criterios de cierre; si queda parcial o bloqueada, registrar ese estado y la evidencia disponible.

## 4. Etapa 1: entorno de Colab, acceso y almacenamiento persistente

**Objetivo.** Establecer una sesión reconstruible y una política de persistencia antes de adquirir o procesar datos.

**Prerrequisitos.** Leer las fuentes de la sección 1. Disponer de una cuenta de Colab, una raíz persistente y acceso autorizado a modelos; resolver los permisos pendientes sin incluir secretos en el repositorio.

**Descripción técnica y tareas.**

1. **[R]** Crear bootstrap idempotente que monte la raíz persistente, identifique GPU/VRAM y espacio libre, instale versiones compatibles y registre un archivo de dependencias fijadas después de verificarlas. Distinguir requisitos mínimos de versiones efectivamente probadas.
2. **[D/P]** Verificar licencias de Sen1Floods11, SAM 2.1 y DINOv3, autorización de pesos gated, IDs exactos y revisiones. Usar secretos de Colab o variables de entorno; no registrar tokens ni redistribuir pesos sin permiso.
3. **[R]** Probar importaciones y disponibilidad CUDA sin ejecutar experimentos. Definir una semilla explícita, política de precisión y convenciones de `run_id`, hashes, logs y estados `planned/running/interrupted/complete/failed`.
4. **[R]** Implementar escritura persistente verificable, lectura desde una sesión nueva y control de compatibilidad para reanudar. Reservar disco para datos, cachés, checkpoints y predicciones. Si no hay GPU suficiente o falta acceso, producir un diagnóstico y detener el paso dependiente; no cambiar modelos silenciosamente.

**Archivos previstos y entregables.** `notebooks/01_entorno_colab.ipynb`, `requirements-colab.txt` o equivalente fijado, `configs/environment.yaml`, módulos `runtime.py` y `artifacts.py`, `docs/project/cuaderno_decisiones.md`; informe persistente de entorno, accesos y licencias.

**Validaciones y cierre.** Desde una sesión limpia se instala el entorno, se recupera un artefacto de prueba tras reinicio y se detecta un archivo incompleto. Los secretos no aparecen en salidas. Quedan registrados hardware, versiones, semilla, raíz persistente y revisiones elegidas. No exigir una GPU concreta sin medir disponibilidad.

**Prompt para el agente.**

```text
Implementa exclusivamente la etapa 1 de la fase 1 de auto-prompt-sam2 en Google Colab. Lee README.md, docs/course/tarea_final_deep_learning.md, docs/WAIMLAP/Abstract_CostOfThePrompt.md y docs/project/hoja_de_ruta_fase1.md. Respeta las instrucciones del repositorio y los cambios existentes.
Prepara notebooks/01_entorno_colab.ipynb, dependencias fijadas tras comprobar compatibilidad, configs/environment.yaml y utilidades reutilizables de runtime y persistencia. Verifica acceso y licencias de Sen1Floods11, SAM 2.1 tiny y DINOv3 facebook/dinov3-vitl16-pretrain-sat493m, sin descargar todavía datos ni pesos. Registra los IDs/revisiones confirmados; si falta autorización, deja un bloqueo preciso, sin sustituir modelos.
Define una raíz persistente configurable, hashes, run_id, semilla, manifiesto de entorno, escritura con marcador de completitud y recuperación tras desconexión. Usa secretos de Colab sin mostrarlos. Documenta hardware y memoria disponibles y valida desde una sesión limpia con un artefacto pequeño de prueba.
Crea o amplía el cuaderno de decisiones con justificación y uso de IA. No prepares los splits, no ejecutes inferencia o entrenamiento y no construyas la demo. Entrega los archivos, evidencia de las validaciones y pendientes; distingue comprobaciones ejecutadas de instrucciones aún no probadas. Detente al cerrar esta etapa.
Al concluir esta etapa, actualiza docs/project/estado_actual_fase1.md con la fecha, el estado, el progreso realizado, los entregables, las validaciones y los pendientes o bloqueos.
```

## 5. Etapa 2: datos, particiones y preprocesamiento

**Objetivo.** Convertir los datos originales en entradas trazables manteniendo aislados los dos regímenes experimentales.

**Prerrequisitos.** Etapa 1 cerrada; acceso y almacenamiento listos; archivos oficiales de particiones identificados. **[P]** Confirmar estructura real de GeoTIFF, orden de bandas y metadatos de nodata.

**Descripción técnica y tareas.**

1. **[D/R]** Adquirir en esta futura etapa solo la modalidad Sentinel-2 y etiquetas manuales necesarias, con origen y hashes. Guardar originales sin modificar y validar correspondencia de chip, imagen y etiqueta; no inferir IDs únicamente por orden de archivos.
2. Construir manifiestos `official` y `leave_paraguay_out` según la sección 2.1, con ID, evento, split original, rol efectivo, rutas, hashes y geometría. Hacer que cualquier función de entrenamiento requiera explícitamente régimen y rol.
3. Verificar estructura de los 446 chips sin inspección exploratoria de reservados. Realizar visualizaciones y estadísticas de calidad solo sobre entrenamiento/validación permitidos; revisar CRS, resolución y alineación espacial.
4. **[R/P]** Fijar una regla de selección del límite superior de recorte antes de medirla; por ejemplo, un percentil prefijado de valores válidos del entrenamiento. Guardar regla, valor y IDs fuente por régimen. Nunca recalcularlo por imagen de prueba ni con validación.
5. Implementar RGB y las dos ramas de transformación del modelo, con mapas explícitos entre coordenadas originales, parches DINOv3 y entrada SAM. Registrar padding, resize, estadísticas, rango/dtype y transformaciones inversas. Proteger contra normalización doble en los wrappers oficiales.
6. Definir tratamiento de nodata de entrada y máscara de validez de etiquetas por separado. Preparar casos sintéticos de coordenadas, etiquetas `-1/0/1` y chips sin agua; no generar cachés de conjuntos reservados todavía.

**Archivos previstos y entregables.** `notebooks/02_datos_y_splits.ipynb`, `src/auto_prompt_sam2/data.py`, `preprocessing.py`, `configs/data.yaml`, manifiestos y estadísticas persistentes por régimen, pruebas de particiones y geometría, informe de auditoría de datos.

**Validaciones y cierre.** Recuentos 252/89/90/15 y 213/75/76/67 con Bolivia apartada; unión de IDs coherente e intersecciones vacías dentro de cada régimen; Paraguay ausente de toda fuente de ajuste del régimen excluido. Transformación inversa de puntos con error dentro de una tolerancia fijada, etiquetas sin clases nuevas y exclusión verificable de inválidos. El preprocesamiento se reproduce desde archivos guardados tras reiniciar Colab.

**Prompt para el agente.**

```text
Implementa solo la etapa 2 de docs/project/hoja_de_ruta_fase1.md en auto-prompt-sam2. Lee esa guía, README.md y el resumen WAIMLAP. Reutiliza el entorno, persistencia y registro de decisiones de la etapa 1; verifica sus entregables y no reconstruyas trabajo equivalente.
Prepara notebooks/02_datos_y_splits.ipynb y módulos compartidos de datos/preprocesamiento. Adquiere los 446 chips manuales Sentinel-2 L1C de Sen1Floods11 y sus splits oficiales con procedencia y hashes. Construye official: train 252, val 89, test 90, Bolivia 15; y leave_paraguay_out: train 213, val 75, test visto 76, Paraguay reservado 67. Bolivia sigue apartada. Verifica IDs, duplicados, eventos, correspondencias y geometría; no hagas análisis exploratorio de reservados.
Implementa RGB B4/B3/B2 dividido por 10000, recorte superior aprendido solo del train permitido por régimen, reescalado y normalización específica de SAM 2.1 y DINOv3. Confirma estadísticas y orden de bandas en las fuentes oficiales y registra las decisiones pendientes resueltas. Guarda transforms de coordenadas y máscaras de validez; no uses inválidos de etiquetas para condicionar la inferencia automática.
Valida conteos, disjunción, exclusión de Paraguay, etiquetas -1/0/1, transformaciones e igualdad del preprocesamiento tras recargarlo. Conserva manifiestos y estadísticas fuera de /content. No ejecutes modelos, entrenamientos ni evaluaciones finales. Registra IA y pendientes, entrega evidencia y detente al cerrar esta etapa.
Al concluir esta etapa, actualiza docs/project/estado_actual_fase1.md con la fecha, el estado, el progreso realizado, los entregables, las validaciones y los pendientes o bloqueos.
```

## 6. Etapa 3: línea base asistida y evaluador común

**Objetivo.** Reproducir SAM 2.1 congelado con puntos derivados de etiquetas y establecer el evaluador que utilizarán todas las variantes.

**Prerrequisitos.** Etapas 1–2 cerradas; acceso a pesos de SAM; contrato geométrico y preprocesamiento verificados.

**Descripción técnica y tareas.**

1. **[D]** Cargar SAM 2.1 tiny y congelarlo completo. Registrar procedencia de pesos y comprobar que inferencia no modifica parámetros.
2. **[R/P]** Precisar la política de puntos simulados: conectividad, positivos dentro del agua válida, negativos fuera, presupuesto por componente, separación mínima, desempates y semilla por chip. Para el contraste principal, alinear presupuesto y política geométrica con los automáticos; declarar la ventaja de usar componentes reales. No ocultar que el acceso a etiquetas también informa sobre ausencia de agua.
3. Ejecutar una llamada por componente, seleccionar la máscara de mayor IoU predicha y unir resultados binarios en la malla original. **[R]** Para chips sin componentes de agua reales, la referencia asistida devuelve máscara vacía; documentar este beneficio de la asistencia. Guardar puntos, scores y reglas de selección.
4. Implementar las métricas de la sección 2.2 con agregación por chip, evento y conjunto. Mostrar resultados de desarrollo en validación, con configuraciones fijadas; no calcular todavía prueba o Bolivia.
5. **[R]** Aislar la función que lee etiquetas para simular prompts de la función de segmentación que solo recibe imagen y prompts. Preparar salida común con máscara, puntos, componentes, IoU predicha y metadatos de ejecución.

**Archivos previstos y entregables.** `notebooks/03_linea_base_asistida.ipynb`, módulos `sam_adapter.py`, `reference_prompts.py`, `metrics.py`; configuración de baseline por régimen, predicciones de validación y reporte de pruebas del evaluador.

**Validaciones y cierre.** Pruebas manualmente calculables para predicción perfecta, error total, solo inválidos, ausencia de agua y falsos positivos. Confirmar coordenadas `(x, y)`, etiquetas positivas/negativas y selección por score predicho. Repetir un chip permitido con igual semilla produce los mismos puntos y resultados dentro de tolerancia documentada. Ningún peso cambia y el registro de acceso a conjuntos reservados permanece vacío.

**Prompt para el agente.**

```text
Implementa exclusivamente la etapa 3 de docs/project/hoja_de_ruta_fase1.md. Lee README.md y docs/WAIMLAP/Abstract_CostOfThePrompt.md y reutiliza los contratos de datos, transforms y persistencia de las etapas 1–2. No modifiques splits ni recalcules estadísticas fuera del train correspondiente.
Construye notebooks/03_linea_base_asistida.ipynb y utilidades compartidas de SAM 2.1 tiny congelado, puntos simulados desde etiquetas y métricas. Los positivos deben caer en agua válida y los negativos fuera; registra semilla, presupuestos y conectividad. Usa SAM por componente, selección multimáscara por IoU predicha y unión en 512x512. Etiqueta la rama como referencia asistida no desplegable y documenta el caso sin agua.
Implementa IoU agua acumulada, media por evento, mediana/IQR por chip, precisión, recall, Dice y mIoU. Ignora -1 en pérdidas/métricas; chips sin agua se excluyen de IoU individual, permanecen en agregados y reportan área FP. Prueba denominadores nulos, chips inválidos, geometría y selección por score predicho. Separa la lectura de etiquetas del adaptador de inferencia.
Ejecuta únicamente comprobaciones y baseline de desarrollo en los roles permitidos de ambos regímenes, nunca test, Bolivia o Paraguay reservado. Conserva config, puntos, máscaras y confusiones, verifica congelamiento y recarga. Registra decisiones y asistencia de IA. No implementes DINOv3, entrenamiento ni demo; informa validaciones y pendientes y detente.
Al concluir esta etapa, actualiza docs/project/estado_actual_fase1.md con la fecha, el estado, el progreso realizado, los entregables, las validaciones y los pendientes o bloqueos.
```

## 7. Etapa 4: disección instrumentada de SAM 2.1 y DINOv3

**Objetivo.** Explicar arquitectura, características y costo de cómputo con evidencia instrumentada que satisfaga el requisito académico A.

**Prerrequisitos.** Baseline y geometría correctas; acceso confirmado a DINOv3; conjunto fijo de chips de entrenamiento para inspección.

**Descripción técnica y tareas.**

1. **[D]** Contar parámetros totales y entrenables por encoder de imagen, encoder de prompts y mask decoder de SAM, y por bloques de DINOv3. Registrar por separado cualquier módulo residual, sin atribuirle costo cero por no aparecer en la ruta de imagen estática.
2. **[D]** Usar contador de FLOPs de PyTorch con tamaño de entrada y cantidad de prompts fijos. **[R]** Reportar operaciones no cubiertas, convención FLOPs/MACs y diferencias entre módulos presentes y realmente ejecutados. No convertir un conteo incompleto en una cifra exacta.
3. **[D/R]** Capturar features de capas identificadas y visualizarlas sobre chips permitidos; si se utiliza PCA u otra proyección aprendida, ajustarla solo con entrenamiento y persistirla. Registrar posición de tokens especiales y cómo se obtiene la cuadrícula de parches.
4. **[D/R]** Elaborar mapas de campo receptivo efectivo mediante gradientes respecto a la entrada o perturbaciones controladas. Distinguirlos del campo receptivo teórico; atención global no implica influencia uniforme. Para esta medición pueden requerirse gradientes de entrada, sin entrenar pesos.
5. **[D]** Medir latencia mediana con calentamiento, repeticiones, batch 1 y sincronización GPU, y memoria máxima. **[R]** Fijar y guardar número de warm-ups y repeticiones, precisión, hardware, cachés y carga de modelos incluida/excluida. Medir por bloque y flujo disponible; el costo completo del método automático se completará en la etapa 7.

**Archivos previstos y entregables.** `notebooks/04_diseccion_modelos.ipynb`, `profiling.py`, `features.py`, `configs/profiling.yaml`; tablas de parámetros/FLOPs/tiempos/memoria, figuras de features y campo receptivo con procedencia, explicación técnica en el cuaderno.

**Validaciones y cierre.** La suma por bloques concuerda con el total bajo una convención explícita. Hooks se eliminan al terminar, no hay gradientes acumulados ni pesos modificados. Las mediciones se reconstruyen desde config y tienen hardware y cobertura documentados. Quedan cubiertos los cinco elementos exigidos: parámetros, FLOPs, features, campo receptivo y latencia/memoria.

**Prompt para el agente.**

```text
Implementa solo la etapa 4 de docs/project/hoja_de_ruta_fase1.md. Consulta las bases del curso y el resumen WAIMLAP. Reutiliza SAM, datos, transforms, métricas y persistencia de las etapas 1–3. Carga la revisión autorizada de DINOv3 ViT-L/16 satelital, sin entrenarlo.
Crea notebooks/04_diseccion_modelos.ipynb y utilidades reutilizables de features/profiling. Instrumenta parámetros y FLOPs por bloque de SAM y DINOv3, features intermedias y mapas de campo receptivo efectivo. Usa solo chips de entrenamiento permitidos por régimen; cualquier proyección aprendida se ajusta exclusivamente allí. Documenta capas, tokens, shapes y transformación a parches.
Mide mediana de latencia tras calentamiento con batch 1 y GPU sincronizada y memoria GPU máxima, fijando repeticiones, hardware, precisión, tamaño y prompts. Declara operaciones no cubiertas por el contador y qué costos incluyen los tiempos. Verifica suma de parámetros, eliminación de hooks y pesos sin cambios.
Guarda tablas, figuras, config e interpretación en almacenamiento persistente y registra hipótesis, límites y uso de IA. No generes prompts automáticos, entrenes el probe/decoder ni evalúes conjuntos reservados. No construyas la demo. Entrega evidencia y pendientes y detente en esta etapa.
Al concluir esta etapa, actualiza docs/project/estado_actual_fase1.md con la fecha, el estado, el progreso realizado, los entregables, las validaciones y los pendientes o bloqueos.
```

## 8. Etapa 5: probe de agua y generación automática de prompts

**Objetivo.** Obtener un flujo automático reproducible que genere puntos sin consultar etiquetas en inferencia.

**Prerrequisitos.** Etapas 1–4 cerradas; features por parche y coordenadas verificadas; espacio de búsqueda registrado antes de comparar resultados entre regímenes.

**Descripción técnica y tareas.**

1. **[D/R]** Extraer features con DINOv3 congelado para train/validación, persistir shards y comprobar sus claves. Construir targets de parches usando proporción de agua sobre píxeles válidos, con umbrales iniciales 0.50/0.05. Registrar ignorados y balance de clases. No dividir al azar parches del mismo chip entre train y validación.
2. **[D/R]** Ajustar regresión logística con train. Elegir regularización, balanceo, muestreo y umbrales de target con validación dentro de una búsqueda prefijada. Cualquier scaler o muestreo depende únicamente de train; guardar coeficientes, intercepto, orden de clases, solver y estado de convergencia. La selección final del flujo se basa en IoU de segmentación de validación; diagnósticos del probe quedan como auxiliares.
3. **[D/R]** Transformar probabilidades de parches a mapa espacial con interpolación explícita, umbralizar, extraer componentes y seleccionar máximos positivos con límite `k+` y separación `d`. Elegir `k−` puntos de baja probabilidad con ámbito espacial y umbral negativo documentados. Resolver empates de forma determinista. Guardar conectividad, tamaño mínimo de componente, bordes, distancia en píxeles originales y correspondencia de puntos.
4. **[R/P]** Fijar casos degenerados antes de evaluar: ausencia de componentes devuelve máscara vacía; si faltan candidatos negativos o positivos, usar los disponibles y registrar el presupuesto real. No inventar puntos con ayuda de la etiqueta. Una alternativa de respaldo sería otra configuración explícita, calibrada solo con validación.
5. **[D]** SAM por componente, multimáscara, mayor IoU predicha y unión. Preparar también malla automática con 32 puntos por eje (32 × 32 posiciones), filtros de IoU predicha y estabilidad, y aceptación como agua por probabilidad DINOv3 media de cada máscara. Fijar el tratamiento de máscaras vacías o regiones sin validez de entrada.
6. **[D/P]** La variante NDWI está contemplada: precisar B3 verde/B8 NIR tras verificar bandas, regla de pseudoetiquetado, denominador cero y umbrales. **[R]** Entrenar su probe únicamente con pseudoetiquetas de train y evaluarla como extensión separada, manteniendo el resto fijo. Si se calibra con validación manual o se ajusta después el decoder con etiquetas, denominarla «probe con pseudoetiquetas NDWI», no sistema completamente libre de etiquetas. Su inclusión y presupuesto deben quedar decididos antes de la etapa 7; no condiciona la comparación principal supervisada.

**Archivos previstos y entregables.** `notebooks/05_prompts_automaticos.ipynb`, `probe.py`, `automatic_prompts.py`, `mask_selection.py`, `configs/probe.yaml`, `configs/prompts.yaml`; features, probe y scaler por régimen, mapas, puntos y predicciones de validación, configuración NDWI o decisión explícita de aplazarla.

**Validaciones y cierre.** El generador recibe solo imágenes/metadatos y funciona con el cargador de etiquetas deshabilitado. Pruebas de puntos en rango, separación mínima, ausencia de componentes, budgets insuficientes, multimáscara y clasificación semántica de malla. Probe y decoder base mantienen revisiones identificadas; caches y probabilidades coinciden tras recarga. Cada parámetro aprendido identifica los IDs train y cada parámetro elegido identifica la validación usada.

**Prompt para el agente.**

```text
Implementa exclusivamente la etapa 5 de docs/project/hoja_de_ruta_fase1.md. Lee el resumen WAIMLAP y reutiliza las etapas 1–4: adaptador SAM, extracción DINOv3, geometría, métricas y persistencia. Trabaja por separado en official y leave_paraguay_out; no uses datos reservados ni artefactos ajustados del otro régimen.
Crea notebooks/05_prompts_automaticos.ipynb y módulos de probe/prompts/selección. Extrae features DINOv3 congeladas de train y validación y entrena regresión logística solo con train: targets iniciales agua >=50% de píxeles válidos y no agua <=5%, ignorando parches ambiguos o sin validez. Ajusta scaler solo con train y selecciona hiperparámetros solo con validación dentro de una búsqueda predefinida. Guarda pesos, scaler, configuración y linaje de datos.
Genera probabilidades, componentes, hasta k+ máximos separados por d y k− negativos de baja probabilidad. Define geometría, umbrales, conectividad, desempates y casos vacíos. Ejecuta SAM por componente, selecciona por IoU predicha y une máscaras. Implementa la rama malla 32x32 con filtros de calidad y aceptación como agua por probabilidad media DINOv3. La inferencia automática no puede leer etiquetas, tampoco su máscara de inválidos.
Registra antes del cierre si se incluye la variante del probe con pseudoetiquetas NDWI; si se incluye, usa solo train y documenta toda dependencia restante de etiquetas manuales. Prueba geometría, ausencia de componentes, conteo de puntos y reconstrucción desde disco. No entrenes el decoder, no ejecutes la campaña final ni construyas la demo. Guarda evidencia, decisiones, IA y pendientes y detente.
Al concluir esta etapa, actualiza docs/project/estado_actual_fase1.md con la fecha, el estado, el progreso realizado, los entregables, las validaciones y los pendientes o bloqueos.
```

## 9. Etapa 6: entrenamiento exclusivo del mask decoder

**Objetivo.** Realizar la intervención de núcleo del requisito B y producir checkpoints completos para reanudar y comparar con el modelo congelado.

**Prerrequisitos.** Probe y política de prompts de etapa 5 congelados para cada régimen; baseline disponible; presupuesto de entrenamiento y función objetivo registrados.

**Descripción técnica y tareas.**

1. **[D]** Inicializar desde SAM base y habilitar gradientes solo en su mask decoder. Mantener DINOv3 y ambos encoders SAM congelados y en el modo adecuado. **[R]** Enumerar parámetros entrenables por nombre; no usar sin revisión un predictor decorado globalmente con `inference_mode` para la ruta de entrenamiento.
2. Reutilizar prompts automáticos del propio régimen sobre train; las etiquetas supervisan la pérdida, no la localización ni la elección del prompt. El probe permanece fijo durante A3. Puede cachearse el encoder de imagen si su entrada no cambia; augmentations que cambian geometría requieren transformar conjuntamente imagen, puntos y etiquetas y regenerar los embeddings correspondientes.
3. **[D/P]** BCE + Dice sobre píxeles válidos. **[R]** Como opción inicial, llevar logits a resolución original, seleccionar la salida por IoU predicha y formar una unión suave mediante máximo de probabilidades por componente para calcular pérdida de chip. Esto mantiene gradiente hacia máscaras seleccionadas; no umbralizar antes de la pérdida. Registrar estabilidad numérica, reducción, pesos BCE/Dice y suavizado. Confirmar esta elección antes de entrenar, pues el documento fuente no prescribe cómo supervisar múltiples componentes.
4. **[R]** Chips sin componentes automáticos mantienen predicción vacía y cuentan en evaluación; registrar explícitamente que no generan actualización en esa ruta, y cuántos pasos/chips quedaron sin gradiente. No sustituir sus prompts por puntos reales. Si esta cobertura es insuficiente, resolver una política de respaldo usando train/validación antes del cierre y repetir comparaciones afectadas.
5. **[P/R]** Fijar optimizador, learning rate, weight decay, batch efectivo, acumulación, precisión, clipping de gradientes, máximo de épocas/pasos y parada temprana. Seleccionar `best` por IoU agua acumulada de validación; fijar desempate determinista. Guardar curva BCE/Dice/IoU, cobertura de chips y mejor época, no solo pérdida mínima.
6. Implementar `last` y `best` con el estado de reanudación de la sección 2.3. Reducir batch o activar acumulación por falta de memoria exige registrar la configuración efectiva; no descongelar otros bloques ni reducir resolución silenciosamente.

**Archivos previstos y entregables.** `notebooks/06_entrenamiento_decoder.ipynb`, `training.py`, `losses.py`, `configs/training.yaml`; checkpoints completos y pesos del decoder exportables por régimen, curvas y reporte de parámetros entrenados.

**Validaciones y cierre.** Un lote sintético o de train demuestra gradiente finito en el decoder y ausencia de cambios en encoders y DINOv3. Cambiar etiquetas únicamente en posiciones inválidas no cambia la pérdida. Una interrupción controlada y reanudación recupera estado y progreso con la tolerancia declarada. Existen `last`, `best`, logs completos y causa de finalización. No se exige mejora sobre la baseline para aceptar una ejecución válida.

**Prompt para el agente.**

```text
Implementa solo la etapa 6 de docs/project/hoja_de_ruta_fase1.md. Reutiliza datos, SAM, probe, generador, métricas y persistencia de etapas 1–5 y verifica sus manifiestos. Entrena independientemente cada régimen desde los pesos base SAM 2.1, con su probe y preprocesamiento propios.
Crea notebooks/06_entrenamiento_decoder.ipynb y módulos de entrenamiento/pérdidas. Descongela exclusivamente el mask decoder: congela encoders de imagen/prompts y DINOv3. Usa prompts automáticos fijos y etiquetas solo en BCE + Dice en píxeles válidos. Resuelve y registra antes del entrenamiento la supervisión multicomponente y el caso sin componentes según esta guía; no uses prompts asistidos como reemplazo oculto ni una salida binaria sin gradiente para la pérdida.
Fija presupuesto, optimizador, learning rate, precisión y batch efectivo; selecciona best por IoU agua de validación. Guarda last y best, estados de optimizador/scheduler/AMP/RNG, progreso, config, hashes y curvas en almacenamiento persistente. No mezcles artefactos official con leave_paraguay_out.
Verifica que solo cambie el decoder, que inválidos no afecten la pérdida y que se pueda reanudar tras una interrupción controlada. Usa solo train/validación permitidos, sin evaluación final. Registra hipótesis, resultados aunque no mejoren, limitaciones y uso de IA. No implementes ablaciones adicionales ni demo; entrega evidencia y pendientes y detente.
Al concluir esta etapa, actualiza docs/project/estado_actual_fase1.md con la fecha, el estado, el progreso realizado, los entregables, las validaciones y los pendientes o bloqueos.
```

## 10. Etapa 7: ablaciones controladas, eficiencia y selección

**Objetivo.** Contrastar A1–A3 con un solo cambio por comparación y fijar todas las filas de la campaña final antes de abrir conjuntos reservados.

**Prerrequisitos.** Etapas 1–6 completas en ambos regímenes; hipótesis previas en el cuaderno; inventario de configuraciones candidatas y presupuesto cerrado.

| Ablación | Cambio y controles | Hipótesis documentada y concreción |
| --- | --- | --- |
| A1 | Fuente: referencia asistida, puntos DINOv3, caja del chip completo, malla + clasificación DINOv3. Mantener SAM y decoder congelados, datos y semilla. | [D] DINOv3 quedaría por debajo de referencia asistida y por encima de malla. [R] Caja cubre extensión fija, sin derivarla de etiqueta. Configuraciones específicas de cada rama se calibran antes y después se congelan; declarar que malla y puntos no igualan número de llamadas ni presupuesto espacial. |
| A2 | Variar `k+` con `k−` fijo; después variar `k−` con `k+` fijo. Resto idéntico y decoder congelado. | [D] IoU aumenta hasta saturar y negativos pueden reducir FP. [R] Usar como candidatos `k+ ∈ {1,3,5,10}`, heredados del borrador, y `k− ∈ {0,1,3,5}` como propuesta nueva; confirmar presupuesto antes de ejecutar. |
| A3 | Decoder congelado frente a ajustado, con idéntico probe, puntos, umbrales, chips y semilla. | [D] La mejora sería mayor en eventos vistos que en Bolivia. La validación permite seleccionar checkpoint, pero la parte de Bolivia se contrasta solo en etapa 8. |

**Descripción técnica y tareas.**

1. Crear una matriz declarativa con ID de comparación, hipótesis fechada, variable modificada, configuración de referencia y controles. Separar búsqueda de hiperparámetros de ablación: en una ablación no recalibrar umbrales para favorecer cada fila. Si se presenta además el mejor sistema de cada rama, etiquetarlo como comparación de sistemas calibrados.
2. Reutilizar predicciones existentes cuando sus hashes coincidan, sin reentrenar el probe por variar `k`. Comparar A3 con prompts idénticos; no recalibrar el generador para el decoder ajustado dentro de esa ablación.
3. **[R]** Presentar en validación costo del prompt, IoU por evento/chip y métricas auxiliares, cantidad real de puntos/componentes y llamadas SAM. Completar latencia y memoria del flujo entero: lectura/preprocesamiento, DINOv3, probe, prompts, SAM y posprocesamiento. Separar inferencia con cachés de inferencia desde imagen sin cachés y carga inicial de modelos.
4. Elegir la configuración automática principal exclusivamente por IoU agua acumulada de validación. **[R]** Desempatar con menor latencia bajo condiciones iguales y después por ID estable. Conservar también las referencias de A1–A3, aunque no sean ganadoras, para medir las hipótesis finales.
5. Crear un cierre por régimen que incluya configs, hashes, checkpoints, filas a evaluar, métricas y reglas de casos especiales. Congelar también el alcance NDWI si se incluyó. No decidir qué variantes reportar después de conocer sus resultados finales.

**Archivos previstos y entregables.** `notebooks/07_ablaciones_y_seleccion.ipynb`, `experiments.py`, `configs/ablations.yaml`; matriz, tablas/curvas de validación, perfiles de eficiencia y `selection.json`/`evaluation_lock.json` por régimen.

**Validaciones y cierre.** Un diff de configuraciones verifica un único factor conceptual por contraste. A3 conserva hashes del probe y prompts. Todas las filas tienen resultados o fallo justificado, sin ocultar negativos. La selección se puede recalcular desde métricas de validación guardadas. Ambos cierres existen y no contienen métricas de reservados.

**Prompt para el agente.**

```text
Implementa exclusivamente la etapa 7 de docs/project/hoja_de_ruta_fase1.md. Lee las hipótesis del resumen WAIMLAP y reutiliza los artefactos de etapas 1–6 sin cambiar splits, preprocesamiento ni modelos de forma implícita. Trabaja solo con validación de cada régimen; ambos deben cerrarse antes de evaluar reservados.
Crea notebooks/07_ablaciones_y_seleccion.ipynb y una matriz declarativa. A1 compara referencia asistida, puntos DINOv3, caja completa y malla con scoring, usando decoder congelado. A2 varía positivos y negativos en barridos separados con controles fijos; registra como recomendaciones los candidatos no fijados en el protocolo. A3 compara decoder congelado/ajustado con el mismo probe, puntos y umbrales. Cada comparación debe tener hipótesis previa, un factor y diff auditable. Separa calibración de ablación.
Reutiliza cachés válidas y reporta métricas, costo del prompt y eficiencia del flujo completo con hardware/precisión/warm-up, batch 1, sincronización y memoria máxima. Selecciona la configuración automática por IoU agua de validación con desempate prefijado y conserva las filas necesarias para contrastar A1–A3, incluidos fallos y resultados negativos.
Guarda selection.json y evaluation_lock.json con hashes y todas las filas de evaluación final para official y leave_paraguay_out. Registra la decisión sobre NDWI. No abras test, Bolivia ni Paraguay reservado; no construyas la demo. Entrega validaciones, cuaderno actualizado y pendientes y detente.
Al concluir esta etapa, actualiza docs/project/estado_actual_fase1.md con la fecha, el estado, el progreso realizado, los entregables, las validaciones y los pendientes o bloqueos.
```

## 11. Etapa 8: evaluación final y generalización geográfica

**Objetivo.** Medir las configuraciones cerradas en datos reservados y distinguir adaptación del decoder de transferencia a eventos no vistos.

**Prerrequisitos.** Cierres experimentales de etapa 7 completos para ambos regímenes, verificaciones de ausencia de fugas aprobadas mediante evidencia y capacidad de almacenar todas las predicciones finales.

**Descripción técnica y tareas.**

1. Reconstruir cada pipeline desde config y checkpoints persistentes, verificar hashes y activar modo de evaluación sin optimizadores ni calibración. Abrir solo las filas y conjuntos listados en el cierre.
2. Evaluar `official` en sus 90 chips de prueba y 15 de Bolivia; evaluar el régimen excluido en sus 76 chips de eventos vistos y 67 de Paraguay. No sumar conjuntos de regímenes distintos como si procedieran del mismo entrenamiento. La fila asistida usa etiquetas solo para su procedimiento prefijado de puntos y para métricas.
3. Guardar máscara final, mapa de probabilidad del probe, puntos y etiquetas de puntos, componentes, scores SAM, confusiones por chip, validez para evaluación, tiempos, evento, régimen y hashes. Los valores esperados de conteos por campaña permiten detectar omisiones sin descartar chips difíciles.
4. Calcular métricas comunes y costo del prompt por conjunto/evento. A3 mide congelado frente a ajustado en los mismos chips; el contraste geográfico compara resultados vistos con Bolivia o Paraguay excluido dentro de su régimen. **[R]** Tratar la mejora por ajuste como evidencia de adaptación al dataset, no como estimación causal pura de todo el cambio natural→satelital.
5. Si hay interrupción, reanudar únicamente unidades pendientes según el manifiesto. **[R]** Recalcular tablas desde predicciones conservadas no constituye nueva selección. Un error técnico que invalide la campaña debe conservarse, explicarse y requerir una corrección versionada; nunca ajustar heurísticas por resultados del test y presentarlo como la misma evaluación cerrada.

**Archivos previstos y entregables.** `notebooks/08_evaluacion_final_geografica.ipynb`, `evaluation.py`; campaña persistente por régimen, tablas finales por chip/evento/conjunto y reporte de integridad y procedencia.

**Validaciones y cierre.** Todas las filas previstas tienen exactamente sus IDs esperados o una incidencia explícita. Los hashes de pesos/config no cambian antes y después. Inferencia automática funciona sin acceso a etiquetas; el evaluador las consume después. Las métricas se reconstruyen únicamente desde predicciones/confusiones persistidas y reproducen las tablas. No se selecciona ningún nuevo checkpoint por rendimiento en Bolivia o Paraguay.

**Prompt para el agente.**

```text
Implementa exclusivamente la etapa 8 de docs/project/hoja_de_ruta_fase1.md. Verifica los cierres evaluation_lock.json de ambos regímenes producidos por etapas 1–7. Si falta cualquiera o sus hashes no coinciden, detén la evaluación y explica el bloqueo. No calibres ni entrenes en esta etapa.
Crea notebooks/08_evaluacion_final_geografica.ipynb reutilizando el evaluador y pipelines existentes. Reconstruye desde almacenamiento persistente y ejecuta una campaña final de las filas prefijadas: official en test 90 y Bolivia 15; leave_paraguay_out en test visto 76 y Paraguay reservado 67. Usa artefactos ajustados propios de cada régimen. Bolivia adicional con el régimen excluido solo si estaba declarada en el cierre.
Separa la referencia asistida, que simula puntos desde etiquetas, del flujo automático que no puede leerlas. Guarda predicciones, probabilidades, puntos, componentes, scores, confusiones, tiempos, IDs y hashes. Reporta IoU acumulada, por evento y mediana/IQR por chip, precisión, recall, Dice, mIoU y área FP de chips sin agua. Contrasta costo del prompt, A3 y generalización dentro de cada régimen sin mezclar poblaciones.
Implementa reanudación de unidades pendientes con idéntica configuración y audita cobertura e inmutabilidad. No cambies el modelo al ver resultados; conserva cualquier fallo técnico y documenta su efecto. Registra IA y limitaciones. No hagas la selección final por test ni construyas demo. Entrega evidencia y detente.
Al concluir esta etapa, actualiza docs/project/estado_actual_fase1.md con la fecha, el estado, el progreso realizado, los entregables, las validaciones y los pendientes o bloqueos.
```

## 12. Etapa 9: análisis de resultados, fallos y cierre de artefactos

**Objetivo.** Convertir la evidencia final en conclusiones defendibles y conservar un paquete suficiente para reconstruir después la inferencia seleccionada.

**Prerrequisitos.** Campañas completas de etapa 8, métricas reconstruibles y configuración principal elegida previamente en validación.

**Descripción técnica y tareas.**

1. Contrastar H1–H3 con diferencias absolutas y puntos porcentuales, sin ocultar hipótesis refutadas. Separar efecto de fuente de prompts, cantidad y ajuste. Reportar soporte por evento y chip, incluyendo Bolivia completa, sin significancia estadística basada en una sola semilla.
2. Analizar agua turbia, vegetación inundada, nubes y sombras de nubes/relieve. **[R]** Usar una rúbrica cualitativa explícita, permitir múltiples categorías e indicar incertidumbre: L1C no aporta la máscara de calidad asumida por otros productos y estas categorías no son etiquetas verificadas por defecto. Incluir fallos del probe, ausencia de positivos, componentes fragmentados, selección multimáscara y límites del decoder.
3. Elegir ejemplos mediante una regla transparente, por ejemplo casos mejores/medianos/peores por IoU definida y chips sin agua con mayor FP, junto con ejemplos por categoría. Las observaciones de reservados solo alimentan el análisis y futuras hipótesis; no cambian el sistema cerrado.
4. Discutir agua permanente, RGB sin NIR/SWIR, resolución de preentrenamiento DINOv3 frente a Sentinel-2, dependencia espacial del split oficial, tamaño de Bolivia y heterogeneidad de hardware Colab. No atribuir diferencias entre eventos únicamente a geografía sin reconocer composición de escenas.
5. Elaborar el paquete de la sección 13 a partir del sistema elegido en validación. Si el decoder ajustado no ganó, conservarlo para A3 y declarar que la configuración principal usa decoder base. Preservar también el modelo sin Paraguay como evidencia, sin sustituir el modelo principal después de consultar sus resultados finales.
6. **[R]** Comprobar en una sesión limpia de Colab la carga del paquete y reproducción de una entrada de validación guardada, sin etiquetas ni reentrenamiento. Este ensayo verifica la entrega experimental; no construye aplicación, interfaz, servicio ni despliegue.

**Archivos previstos y entregables.** `notebooks/09_analisis_y_cierre.ipynb`, `docs/project/resultados_fase1.md`, cuaderno completado, tablas y figuras citadas; manifiesto del paquete, instrucciones de reconstrucción y reporte de verificación en sesión limpia. Son materiales de la fase 1, no la redacción obligatoria de un informe académico completo o un nuevo póster.

**Validaciones y cierre.** Cada tabla/figura identifica campaña y configuración; diferencias se recalculan desde métricas persistidas. Ejemplos se vinculan a IDs y reglas de selección. No se presentan resultados faltantes como cero ni expectativas como hallazgos. El paquete reconstruye la inferencia sin variables ocultas del notebook y su origen coincide con la selección de validación. Quedan declarados límites, fallos y pendientes que afecten su reutilización.

**Prompt para el agente.**

```text
Implementa exclusivamente la etapa 9 de docs/project/hoja_de_ruta_fase1.md. Lee las hipótesis del resumen WAIMLAP y las exigencias académicas. Reutiliza campañas, configs, checkpoints y selección de validación de etapas 1–8; no vuelvas a entrenar, calibrar ni seleccionar a partir de resultados finales.
Crea notebooks/09_analisis_y_cierre.ipynb y docs/project/resultados_fase1.md. Reconstruye tablas y figuras desde artefactos guardados; contrasta A1–A3 y costo del prompt, separando official/Bolivia y leave_paraguay_out/Paraguay. Reporta por evento y chip, chips sin agua, fallos y resultados negativos. Analiza agua turbia, vegetación inundada, nubes y sombras con categorías cualitativas explícitas e incertidumbre. Usa reglas transparentes para elegir ejemplos y no afirmes significancia con una semilla y 15 chips de Bolivia.
Completa el cuaderno con decisiones y asistencia de IA. Conserva el paquete de la sección 13: versiones, manifiestos, transforms, probe/scaler, parámetros de prompts/posprocesamiento, decoder elegido y alternativo, métricas, procedencia y dependencias. La selección principal debe seguir siendo la hecha en validación. Comprueba en una sesión limpia que una entrada de validación reproduce la inferencia sin etiquetas usando estos artefactos.
No construyas interfaz, Gradio, Space, servicio ni despliegue. No descargues datos nuevos para análisis cualitativo adicional. Entrega el análisis, índice de artefactos, evidencia de reconstrucción y asuntos pendientes, distinguiendo lo comprobado de lo no ejecutado. Detente al cerrar la fase 1.
Al concluir esta etapa, actualiza docs/project/estado_actual_fase1.md con la fecha, el estado, el progreso realizado, los entregables, las validaciones y los pendientes o bloqueos.
```

## 13. Artefactos que deben conservarse para la fase posterior

**[D]** La fase posterior debe poder reconstruir inferencia sin entrenamiento y sin etiquetas. **[R]** Entregar un paquete versionado por configuración seleccionada, con rutas relativas al paquete o identificadores portables, no rutas personales de Drive codificadas en el código. Separar este paquete de inferencia del archivo científico completo.

| Grupo | Contenido obligatorio previsto | Comprobación |
| --- | --- | --- |
| Identidad y procedencia | `release_manifest.json`: régimen, run/campaign IDs, selección de validación, commit, checksums, fecha, semilla y referencias de licencias | Todo artefacto resuelve a una ejecución conocida; no contiene tokens. |
| Entorno y modelos base | Dependencias fijadas; versión Python/PyTorch/CUDA; IDs, revisiones, configuración de arquitectura y hashes de SAM 2.1 y DINOv3; instrucciones de adquisición autorizada | Se identifica sin ambigüedad qué pesos base necesita el decoder parcial. No se redistribuyen pesos restringidos por defecto. |
| Datos y geometría | Manifiestos de splits, orden de bandas, unidades, dtype/rango, nodata, CRS/resolución esperados, tamaños, resize/padding, transformaciones de coordenadas | Reproduce la preparación de una entrada sin consultar etiquetas. |
| Preprocesamiento | Límite de recorte por régimen y método train usado para obtenerlo; estadísticas de normalización de cada modelo, interpolaciones y rutas de entrada | Los valores están guardados; no se estiman durante inferencia ni se normaliza dos veces. |
| Probe | Pesos/intercepto, clases, scaler si existe, capa/features/tokens usados, dimensión, solver/regularización y metadatos de targets 0.50/0.05 o seleccionados | Probabilidades coinciden tras recarga y su origen train es trazable. La variante NDWI tiene identificador propio. |
| Prompts y posprocesamiento | Umbrales positivo/negativo, `k+`, `k−`, distancia `d`, conectividad, mínimos de componente, interpolación, desempates, casos vacíos, selección por IoU predicha, umbral de máscara y unión | Inferencia determinista o tolerancia documentada, independiente de etiquetas. |
| Malla y referencias | Densidad de malla, filtros de IoU/estabilidad y umbral de probabilidad media; config asistida y caja para reproducir comparaciones | La referencia asistida está marcada como no utilizable sin etiquetas; no entra en el flujo automático principal. |
| Checkpoints | Probe, decoder `best`, decoder base identificado, configuración seleccionada y alternativa ajustada para A3; archivo científico con `last` y estado completo de reanudación | Compatibilidad de shapes/keys y hash con modelo base; `best` elegido con validación. |
| Calidad experimental | Métricas por chip/evento/conjunto, TP/FP/FN/TN, IoU/F1/precisión/recall/mIoU, mediana/IQR, exclusiones y FP sin agua; predicciones y prompts finales | Tablas reconstruibles sin nueva inferencia; conjuntos y regímenes no mezclados. |
| Recursos y diagnóstico | Parámetros, FLOPs y cobertura, latencia por bloque y flujo completo, memoria máxima, hardware, precisión, condiciones y perfiles con/sin caché | Permite decidir recursos futuros basándose en mediciones, sin prometer rendimiento en otro hardware. |
| Ejemplo de reconstrucción | Entrada permitida de validación o referencia verificable a ella, hash, salida esperada y tolerancias; instrucciones de carga y orden de llamadas de los módulos existentes | Sesión nueva de Colab reproduce el resultado sin entrenamiento ni etiquetas. |
| Evidencia académica | Cuaderno de decisiones/IA, hipótesis y resultados, fallos, figuras con fuentes y licencias, informe de fase 1 | Cada conclusión enlaza sus artefactos y reconoce límites. |

Guardar las probabilidades y predicciones necesarias para auditar; los embeddings voluminosos pueden quedar en el archivo persistente de experimentos y no en el paquete mínimo de inferencia. **[R]** Mantener copias verificadas del manifiesto y de los checkpoints esenciales en un segundo destino autorizado si el presupuesto lo permite.

## 14. Asuntos pendientes y criterio global de aceptación

| Pendiente | Resolver antes de | Evidencia de resolución |
| --- | --- | --- |
| IDs/revisiones exactos, acceso gated, licencias y compatibilidad Colab | Uso de pesos/datos en etapas 2–4 | Registro de fuentes oficiales consultadas durante la implementación, versiones y permisos. |
| Orden de bandas, normalizaciones, tamaño DINOv3, nodata y regla de recorte | Cierre de etapa 2 | Config efectiva por régimen y validaciones geométricas/radiométricas. |
| Muestreo de referencia, presupuesto, conectividad, máscaras vacías y criterios de multimáscara | Cierre de etapas 3 y 5 | Configs y pruebas de casos límite. |
| Regularización del probe, filtros de parches, thresholds, mínimos de componente, ámbito de negativos y grillas de búsqueda | Comparaciones de etapa 5 | Búsqueda prefijada y linaje train/validación aislado por régimen. |
| Alcance de NDWI y denominación correcta de su dependencia de etiquetas | Cierre de etapa 5 | Decisión de inclusión/aplazamiento, config y presupuesto separado. |
| Supervisión multicomponente, casos sin gradiente, pesos de BCE/Dice y presupuesto de entrenamiento | Entrenamiento de etapa 6 | Justificación, pruebas de gradientes y config de entrenamiento. |
| Barridos A2, condiciones de eficiencia, filas finales y posible Bolivia adicional | Cierre de etapa 7 | Matriz de ablaciones y cierres inmutables para ambos regímenes. |
| Fechas académicas y datos institucionales incompletos de WAIMLAP | Preparación de entregables académicos posteriores | Confirmación del equipo/docente; no bloquea pruebas técnicas independientes. |

La fase 1 estará terminada cuando:

- [ ] Los nueve cierres de etapa tengan evidencia y ambos regímenes cumplan la separación de datos.
- [ ] La baseline, la disección completa, la intervención del decoder, A1–A3 y la evaluación geográfica estén documentadas, incluidos fallos y ausencia de mejoras.
- [ ] No se haya ajustado ninguna decisión usando prueba, Bolivia o Paraguay reservado; cualquier desviación esté identificada y limite explícitamente las conclusiones.
- [ ] Los resultados puedan reconstruirse desde predicciones guardadas y la inferencia principal desde el paquete, en una sesión limpia de Colab.
- [ ] Configuraciones, métricas, checkpoints y estados de reanudación tengan almacenamiento persistente e integridad verificada.
- [ ] Quede claro qué está documentado, qué recomendación se adoptó y qué asunto sigue pendiente, sin presentar como completadas tareas no ejecutadas.

No forman parte de este cierre la construcción de la demo, su despliegue ni la adquisición de un chip adicional de Perú. El límite de la fase 1 es la entrega verificable de resultados y artefactos experimentales reutilizables.
