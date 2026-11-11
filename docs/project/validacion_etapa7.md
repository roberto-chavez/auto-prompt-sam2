# Etapa 7: ablaciones y selección en validación

Fecha: **2026-11-01 (America/Lima)**. Estado: **En curso: implementación preparada; Colab y revisión externa pendientes**. Las seis etapas anteriores conservan su cierre. No se atribuyen resultados nuevos ni se generan cierres experimentales ficticios en la PC.

## Protocolo previo y alcance

Se leyó la sección 2 del [resumen WAIMLAP](../WAIMLAP/Abstract_CostOfThePrompt.md) y la etapa 7 de la [hoja de ruta](hoja_de_ruta_fase1.md). H1: puntos DINOv3 por debajo de la referencia asistida y por encima de malla. H2: positivos mejoran hasta saturar y negativos reducen FP. H3: el decoder ajustado mejora más eventos vistos que Bolivia; **aquí solo se mide validación**, sin probar todavía su parte geográfica. Una semilla no permite afirmaciones de significancia. No hay anotaciones de vegetación húmeda/sombras que permitan atribuirles causalmente los FP.

La PC se limita a código, sintaxis, empaquetado y pruebas sintéticas CPU sin modelos, gradientes ni entrenamiento. Todos los modelos, las ablaciones y los perfiles se ejecutan exclusivamente por el usuario en Colab GPU con Drive montado. No se abre test, Bolivia ni Paraguay reservado, ni se implementan etapas 8–9 o demo. La lectura de inventarios e informes históricos no constituye una nueva evaluación de train o reservados.

## Matriz cerrada antes de ejecutar

[ablations.yaml](../../configs/ablations.yaml), JSON válido como YAML, declara las hipótesis fechadas, candidatos, controles, presupuesto, calibración previa, reglas de selección y NDWI. [experiments.py](../../src/auto_prompt_sam2/experiments.py) resuelve 13 filas y 10 contrastes por régimen. Los controles se serializan íntegramente y su diff debe cambiar exactamente un campo conceptual.

| Filas | Factor y controles | Procedencia |
| --- | --- | --- |
| `a1_points`, `a1_assisted`, `a1_whole_box`, `a1_grid` | Fuente del prompt; SAM base congelado, mismos chips/semilla, probe y umbrales previos. Caja `[0,0,511,511]` nativa fija, sin etiquetas. Malla 32×32 y scoring DINOv3 previos. | Puntos/malla reutilizan etapa 5 por hashes; caja y referencia emparejada nuevas en Colab. |
| `historical_assisted` | Referencia descriptiva de etapa 3, **fuera del contraste controlado y del costo primario**. | 1+/1−, separación 8 px, política histórica y máscaras exactas de etapa 3. |
| `a2_positive_{1,5,10}` más `a1_points` para 3 | Solo k+ ∈ {1,3,5,10}; k−=3. Decoder base, probe, umbrales, distancia 16 px y demás política fijos. | **[R]** Candidatos heredados del borrador; no decisión documentada en WAIMLAP. |
| `a2_negative_{0,1,5}` más `a1_points` para 3 | Solo k− ∈ {0,1,3,5}; k+=3. No se toma el ganador del barrido positivo como nuevo control. | **[R]** Candidatos propuestos, presupuesto común fijado en esta implementación. |
| `a3_frozen`, `a3_tuned` | Solo checkpoint decoder base/best de etapa 6. Mismo probe, puntos exactos, umbrales y **misma ruta directa de etapa 6 en ambas filas**. | Decoder propio del régimen; best official época 10, excluido época 8. Máscaras best existentes reutilizadas; reconstrucción real comprobada durante perfiles y recover. |

La referencia asistida emparejada utiliza 3+/3− por componente y 16 px, iguales a los presupuestos configurados automáticos; cambia la fuente espacial de componentes y puntos. Conserva el muestreo determinista de etapa 3. La referencia histórica no cambia. El costo primario es `IoU(a1_assisted) − IoU(a1_points)` (también pp); **se conservan valores negativos**. Presupuestos por componente iguales no implican igual número de componentes ni igual total de puntos; se reportan los realmente obtenidos, faltantes y llamadas. Malla/caja tampoco igualan presupuesto espacial ni costo de llamadas. Una referencia asistida no es cota matemática garantizada.

La ruta `predictor` se conserva en A1/A2 para reutilizar etapa 5. A3 usa la ruta directa de entrenamiento, con su preprocesamiento/resize originales; cambiar simultáneamente de ruta al comparar congelado y ajustado introduciría un segundo factor. Por eso `a3_frozen` es una fila adicional explícita. No se modifica ningún módulo previo.

## Dependencias, calibración y persistencia

[stage7_dependencies.json](../../configs/stage7_dependencies.json) fija identidades, generaciones y SHA de validation, decoder_export, puntos fijos, parámetros, máscaras y métricas best de ambos regímenes revisados. Se reutilizan los pines de etapas 1–5 de `stage6_dependencies.json`. Los lectores validan bytes y hashes de dependencias, splits/preprocesamiento, revisiones/pesos y biblioteca/configs previos. Un archivo existente no es una caché válida por sí mismo. Ante divergencia se detiene: no se reconstruye un split, recalcula normalizador, entrena probe o cambia modelo.

La calibración queda en `calibration.json`, separada de los contrastes: hereda el probe, umbral positivo y umbral de malla de etapa 5 y best por validación de etapa 6. No se busca ningún umbral adicional en etapa 7 ni se reajusta el probe por k. Se registra NDWI como **aplazado y excluido de etapa 7 y de esta campaña final**: sin entrenamiento/calibración previa, sin afirmación de método sin etiquetas.

Resultados persistentes:

```text
<PERSIST_ROOT>/regimes/<regime>/stage7/<identity_sha256>/
  <artifact>/<generation>/payload + COMPLETE.json
  evidence/
    resolved_matrix.json, calibration.json, inference_bundle.json
    validation_records.json, environment.json, model_loading.json
    result_<row_id>.json, validation_tables.json
    selection.json, evaluation_lock.json, validation.json
    recovery_validation.json, both_regimes.json
  evidence.zip
<PERSIST_ROOT>/stage7/both_regimes/<generation>/...
```

Las generaciones contienen todas las predicciones por chip, métricas, puntos/probabilidades cuando correspondan, origen y hashes de cachés. Para máscaras best reutilizadas de etapa 6 no existían scores individuales SAM guardados: se declara `scores_unavailable_reason`; no se inventan. El perfil registra scores nuevos de la ruta directa sobre el chip prefijado. Los pesos grandes permanecen en sus generaciones verificadas de Drive y no se incluyen en el transporte.

`resume` reutiliza unidades completas y los perfiles anteriores únicamente con las mismas condiciones; conserva fallos publicados. Una corrección de código/protocolo cambia la identidad y exige una ejecución nueva, sin sobrescribir evidencia. `recover` requiere otro boot ID Colab, verifica y recalcula selección/cierre desde disco y reconstruye desde imagen, sin cachés de features/embeddings, el primer chip val de **todas** las filas exitosas. No rellena huecos ni ejecuta entrenamiento.

## Métricas, eficiencia y elección

Se reutiliza el evaluador común: confusiones acumuladas, IoU agua, media por evento, mediana/IQR por chip, precisión, recall, Dice y mIoU; inválidos excluidos, chips sin agua conservados en pooled y FP/área. Cada máscara persistida se vuelve a contrastar con etiquetas de **val** después de inferir. La API automática no acepta etiquetas; hay tripwires en todo el flujo automático. Se exportan también tablas de contrastes y coordenadas de ambas curvas de A2, sin omitir fallos o deltas negativos.

Eficiencia: FP32 sin autocast, batch **una imagen** (la malla mantiene sus 64 prompts por lote internos), 5 warm-ups y 20 repeticiones; sincronización CUDA antes/después, tiempo host y picos allocated/reserved tras reiniciar contadores. Condiciones GPU/CPU, versiones, precisión y boot ID quedan guardados. Se miden por separado lectura/verificación SHA, preprocesamiento DINO/SAM, DINO, probe/mapa, prompts, SAM y posprocesamiento. La comparación asistida incluye leer etiquetas para simular puntos, declarado como asistencia.

Se separan features verificadas en RAM y ejecución desde imagen sin cachés de features/embeddings. **Ambas recalculan embeddings SAM**, pero el sistema operativo/Drive y allocator están calientes. Los tiempos incluyen instrumentación ligera/sincronizaciones de etapas, no métricas, escritura de artefactos ni carga inicial de modelos. SAM y DINO permanecen residentes para condiciones iguales; la memoria base incluye ambos aunque caja/asistencia no utilicen DINO. Carga offline más auditoría de hashes se mide aparte. Los perfiles usan **solo el primer chip val por ID**, común a todas las filas del régimen: no son una estimación de latencia de toda la población. Las llamadas de cachés antiguas se derivan de componentes persistidos o de la agenda fija 1024/64 de la malla; los perfiles nuevos cuentan forwards reales con hooks retirados al terminar.

Se elige exclusivamente la fila automática con mayor IoU agua acumulada de **toda** su validación. Empate exacto: menor mediana del flujo desde imagen en condiciones iguales, luego ID lexicográfico; no se introduce tolerancia para transformar diferencias de IoU en empates. Se conservan las 13 configuraciones para evaluación final, incluidas fallidas con política de intentar exactamente esa configuración o registrar fallo. No hay poda adaptativa por resultados reservados. Ambos `selection.json` y `evaluation_lock.json` están enlazados por hashes; un registro conjunto exige los dos regímenes. Los locks computacionales **siguen pendientes de revisión externa**, con `stage8_authorized=false`.

## Ejecución y entrega

1. En PC, ejecutar `.venv-validation/Scripts/python.exe scripts/validate_stage7_local.py`. Genera el ZIP de fuentes `artifacts/auto-prompt-sam2-stage7.zip`, SHA e informes CPU. No instalar entornos de modelos locales.
2. Subir ese ZIP a MyDrive. Abrir [07_ablaciones_y_seleccion.ipynb](../../notebooks/07_ablaciones_y_seleccion.ipynb) en Colab GPU. Configurar la raíz histórica íntegra y el SHA impreso por el exportador. Mantener todos los artefactos de Drive de etapas previas; no basta con sus ZIP ligeros.
3. Ejecutar entorno, revisar matriz/presupuesto y ejecutar con `RUN_MODE='prepare'`. Hace solo validación en ambos regímenes. Si se corta, detener el escritor anterior y usar `resume` con idéntico paquete/raíz/hardware. Un fallo publicado permanece visible.
4. Guardar las salidas prepare y cambiar a un runtime GPU nuevo. Repetir entorno con el mismo SHA y ejecutar `RUN_MODE='recover'`, offline para modelos. Conservar ambos conjuntos de logs y notebooks con salidas.
5. Compartir notebook(s) ejecutado(s), los **dos** `evidence.zip` terminados y sus SHA, los JSON selection/lock/validation/recovery/both_regimes y logs/failure si existen. El script imprime rutas exactas. No ejecutar reservados.
6. Revisión CPU preparada: `python scripts/review_stage7_evidence.py <official.zip> <leave_paraguay_out.zip> --data-root <raiz_persistente_local_etapa2> --output <review.json>`. Verifica CRC, generaciones, fuentes, diffs, métricas, perfiles, máscaras/etiquetas val, selección y ambos locks. Sin `--data-root`, informa auditoría de etiquetas pendiente. Nunca ejecuta notebooks ni deserializa pesos. La revisión humana del notebook, hardware, fallos, resultados y límites completa el cierre; el script solo verifica consistencia portable.

## Validaciones y pendientes reales

La evidencia local queda en [evidence/stage7](evidence/stage7/). Su alcance es sintaxis, notebook sin ejecutar, pruebas sintéticas CPU, pines de etapa 6 y transporte íntegro de fuentes; no acredita IoU, tiempos GPU ni nuevos cierres. El estado numérico de las comprobaciones se registra en `local_validation.json` y `bundle_validation.json`.

Comprobación local: **91 pruebas CPU sintéticas (16 de etapa 7), pip check y sintaxis correctos**; se verifica además el transporte por CRC, conjunto de archivos y bytes originales. [matrix_preview.json](evidence/stage7/matrix_preview.json) resuelve los 13 candidatos/10 diffs con los bundles revisados sin ejecutar modelos ni abrir rásteres: es una previsualización de metadatos, no un lock. La nueva prueba de transporte/revisión reconstruye selección/lock desde máscaras y métricas sintéticas y detecta una fuente alterada. No se instanciaron modelos reales, entrenó, calculó gradientes ni abrió datos reales/reservados en estas comprobaciones.

Pendientes: ejecutar prepare/recover por el usuario en Colab, recibir notebook/JSON/ZIP/logs, auditar ambas evidencias e interpretar A1–A3 (incluidos negativos/fallos). Hasta entonces **etapa 7 En curso, 6/9 concluidas**. Si no hay GPU/artefactos de Drive, registrar el bloqueo y no sustituir la carga en PC ni alterar precisión/modelo/protocolo. Salvedad de licencia y revisión académica humana previas conservadas.
