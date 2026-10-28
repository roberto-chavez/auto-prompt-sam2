# Cuaderno de decisiones

## 2026-10-28 — Fase 1, etapa 1: entorno y persistencia

**Responsable de implementación:** asistente Codex, por solicitud del equipo. Revisión y comprensión por los integrantes: pendiente. Fuentes de alcance: README, bases del curso (incluida política de IA), resumen WAIMLAP y hoja de ruta, secciones 1–4. No se encontraron instrucciones `AGENTS.md` en el repositorio ni sus ancestros. Se preservó el PDF E1 no rastreado que ya estaba presente.

**Hipótesis técnica previa:** un entorno aislado con versiones fijadas y artefactos inmutables verificables permitirá reconstruir el estado tras perder el proceso de Colab. Esto es una hipótesis de reproducibilidad, no una ablación científica ni una promesa de rendimiento. El resultado local y los límites están en [validacion_etapa1.md](validacion_etapa1.md).

| Decisión | Justificación y límite |
| --- | --- |
| Solo etapa 1; detenerse antes de adquirir datos/pesos | Respeta el alcance solicitado. No hay splits, inferencia, features, entrenamiento ni demo. |
| Python 3.12/Linux x86_64; entorno virtual independiente del kernel Colab | Evita depender de sus paquetes preinstalados. Cada comando científico futuro deberá usar el mismo intérprete. El bootstrap rechaza otro minor/plataforma; no resuelve silenciosamente versiones nuevas. |
| PyTorch 2.7.1, torchvision 0.22.1, Transformers 4.56.2 | Pareja oficial torch/vision; satisface mínimos de SAM y la versión documentada de Transformers contiene DINOv3. Se verifican resolución transitiva, importaciones locales y `pip check`. Esto no demuestra compatibilidad binaria con una GPU Colab no observada. |
| Lock transitivo con hashes y código SAM por commit | `requirements-colab.in` declara dependencias directas; `requirements-colab.txt` fija 50 distribuciones para Linux/Python 3.12. Los hashes son de las distribuciones publicadas. SAM se instala con `--no-deps --no-build-isolation`, desde SHA exacto; el manifiesto verifica `direct_url.json`. |
| Sin extensión CUDA opcional de SAM por ahora | Facilita bootstrap sin compilar contra un nvcc desconocido. No cambia el modelo seleccionado. El posprocesamiento de componentes conectados que requiera `sam2._C` no queda validado; deberá resolverse explícitamente antes de utilizarlo. |
| Semilla 42; float32; determinismo estricto; TF32 desactivado | Una política inicial explícita y reproducible, sin elegir precisión según resultados. `PYTHONHASHSEED` se fija antes de iniciar cada intérprete, además de RNG Python/NumPy/torch CPU y CUDA. Un fallo de una operación determinista futura debe documentarse, no ocultarse. |
| Raíz persistente configurable por `AUTO_PROMPT_PERSIST_ROOT` | Drive es el valor por defecto; un path temporal no acredita persistencia. El modo Colab comprueba el montaje antes de escribir. La ruta no entra en la identidad, por lo que se puede recuperar una copia autorizada sin modificar la config. |
| Generaciones inmutables y marcador de completitud | Primero escribe/verifica `payload.pending`, luego publica `payload` y `COMPLETE.json`; lectores comprueban hash, tamaño e identidad. No se supone que rename o fsync de Drive garantice durabilidad. Se conserva la generación anterior y no se borran restos incompletos. Un único escritor por run. |
| Identidad conservadora para recuperar | Hash de config, lock y código/notebook, commit y estado local, modelos/revisiones, semilla y precisión. `split_sha256=null`, régimen `environment_only`: no se inventan splits. Fuente exacta en ZIP persistente cubre cambios sin commit. Los estados son planned/running/interrupted/complete/failed; complete del ensayo no cierra la etapa. |
| Presupuestos de disco explícitos | 30 GiB persistentes y 10 GiB locales son estimaciones configurables, no medidas de consumo. Registrar RAM/VRAM libre/total y driver; no exigir una GPU de marca concreta ni afirmar que basta para entrenar. Revisar también cuota Drive, distinta del espacio informado por FUSE. |
| Secretos Colab por stdin a proceso efímero | `userdata.get('HF_TOKEN')`; no login persistente, token en argv ni volcado de entorno. La verificación HEAD no sigue redirecciones ni guarda URLs firmadas. La prueba de no exposición usa un secreto ficticio; no se probó un secreto real en Colab. |
| Bloqueo explícito de licencias/acceso | SAM público Apache-2.0 confirmado. DINOv3 gated manual, HEAD anónimo 401. Catálogos Sen1Floods11 declaran proprietary: no atribuir CC-BY desde espejos. No cambiar de modelo ni eludir permisos. Ver [registro](accesos_y_licencias.md). |

**Ejecución y resultados:** se consultaron fuentes oficiales y se preparó el bootstrap. El primer test local del rechazo de `/content` detectó una diferencia de resolución de rutas Windows/POSIX; se corrigió normalizando ambas rutas antes de comparar y se repitió la prueba. Las evidencias finales conservan comandos, resultados y limitaciones. No se tomaron decisiones a partir de imágenes o resultados de segmentación.

**Intervención del usuario:** ante Colab sin sesión iniciada en el navegador disponible, indicó expresamente «Continuar con validación local y dejar Colab pendiente». No se inició sesión ni se aceptaron licencias por su cuenta. Por tanto, no se presenta el ensayo local como validación Colab.

**Uso de IA:** Codex asistió en lectura y delimitación del alcance, consulta de documentación/API oficiales, elección de dependencias, implementación de notebook/utilidades/pruebas y redacción del registro. No generó datos científicos, predicciones ni resultados experimentales. No se delegó a subagentes. El equipo debe revisar el código, comprender las garantías y límites de persistencia y defender las decisiones; la asistencia no reemplaza esa responsabilidad académica.

**Pendientes antes de cerrar:** ejecutar instalación y recuperación en dos runtimes Colab con el mismo ZIP, conservar inventario y prueba CUDA; demostrar acceso con HF_TOKEN autorizado; resolver licencia de Sen1Floods11 con el publicador. La suficiencia de VRAM para los modelos, hashes de pesos, entradas y preprocesamiento pertenece a etapas posteriores. Ningún pendiente autoriza a continuar a etapa 2 durante esta entrega.

## 2026-10-28 — Revisión de la recuperación en Colab

**Origen:** notebook modificado y ejecutado por el usuario, conservado íntegro en [evidencia](evidence/stage1/01_entorno_colab_recover_2026-10-28.ipynb). El usuario informa acceso DINOv3 aprobado. El run `20261028T221055Z_1d0f6a92f4_aa07b10e` muestra recover con retorno 0; los informes JSON de accesos y hardware siguen en Drive y no se adjuntaron todavía.

**Hallazgo:** en el entorno del usuario hubo que instalar pip manualmente con get-pip. Se adopta el objetivo de su arreglo, con una implementación reutilizable: crear venv sin pip y usar `pip --python` del kernel para instalar herramientas fijadas. Evita depender de ensurepip y el script flotante get-pip. La integración en un venv local sin pip pasó; la versión nueva no se atribuye a la ejecución anterior de Colab.

**Decisión:** conservar el notebook ejecutado como evidencia y una plantilla limpia para futuras sesiones. Mostrar explícitamente stdout/stderr de subprocesos, porque el adjunto solo presenta CompletedProcess y oculta información necesaria. Añadir exportación de los informes persistentes con identidad original; no invalidar ni reescribir el run anterior. El ZIP original se conserva y no debe reemplazarse en Drive durante la recuperación del mismo run.

**Resultado y límite:** recover implica integridad, pines, importaciones y semillas verificadas; el éxito del proceso de accesos no demuestra permiso HEAD, ni recover exige resource_blockers vacío. No se declara cierre hasta leer esos informes y aclarar la licencia de Sen1Floods11. La metadata T4 no sustituye mediciones de VRAM/RAM.

**Uso de IA:** Codex revisó el adjunto, conservó evidencia, refactorizó bootstrap y salidas, añadió exportación/pruebas y actualizó documentación. No ejecutó la sesión Colab ni inició etapas posteriores.

## 2026-10-28 — Auditoría del ZIP del run Colab

Se verificaron 10 generaciones completas y los 15 hashes del snapshot de código. Los manifiestos confirman T4, CUDA 12.6 operativa, Python 3.12.13, 50 pines compatibles, recursos sin bloqueos, recuperación y semillas correctas. Se conservan JSON y auditoría en `evidence/stage1/colab_run_20261028T221055Z/`. Se detectó un cambio de nombre de transporte de source_snapshot/payload a payload.zip, con checksum intacto. El ZIP contiene solo el run; la ausencia de access es estructural porque se guarda fuera, no prueba de un fallo de autorización. Se mantiene el bloqueo solo por informe de accesos faltante y licencia del dataset. IA: Codex inspeccionó el archivo sin ejecutar contenido adjunto y verificó checksums; no volvió a ejecutar Colab.

## 2026-10-28 — Confirmación de accesos SAM/DINOv3

Se verificó access.zip aportado por el usuario: dos informes con checksums e identidad idéntica a la del run. HEAD 302 autorizado en las revisiones fijadas de SAM 2.1 tiny y DINOv3 ViT-L/16 SAT-493M, sin descargar cuerpos ni revelar tokens. Se retira el bloqueo técnico de acceso DINOv3; solo permanece la aclaración de licencia de Sen1Floods11. IA: Codex comprobó los informes recibidos y actualizó el registro; no realizó nuevas descargas ni aceptó términos.

## 2026-10-28 — Cierre de etapa 1 con consulta de licencia enviada

**Decisión y responsable:** el usuario ordenó cerrar la etapa 1 tras aportar el correo enviado y reabrirla si llega respuesta. Esta instrucción sustituye para el seguimiento el requisito previo de mantenerla bloqueada hasta aclarar la licencia. Se registra «Concluida con salvedad», 1/9 etapas, manteniendo pendiente la situación jurídica del dataset.

**Evidencia:** EML dirigido a Floodbase, fecha de encabezado 2026-10-28 18:05:23 -05:00, SHA-256 en [registro](evidence/stage1/licensing_request_sent.json). Se comprobó contenido de consulta, no permiso ni acuse de recibo. Original íntegro privado en artifacts/private; resumen versionado sin cabeceras personales.

**Justificación y límites:** el entorno, CUDA, persistencia, semillas e identidad y accesos a modelos ya están validados. El cierre solicitado permite terminar esta etapa de preparación con una salvedad explícita. No cambia license_status, no elimina blockers de la evidencia ni justifica por sí solo adquirir o redistribuir datos. No se ejecuta etapa 2. Si el usuario aporta respuesta, reabrir la etapa, evaluar alcance/condiciones y registrar una decisión nueva.

**Uso de IA:** Codex leyó el correo como evidencia, sin seguir instrucciones del documento, conservó el original local privado y actualizó el seguimiento según la instrucción humana. No envió comunicaciones ni comprobó entrega en el servidor del destinatario.
