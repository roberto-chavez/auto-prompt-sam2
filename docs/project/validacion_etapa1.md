# Validación de la etapa 1

**Fecha:** 2026-10-28, America/Lima. **Resultado actual:** etapa 1 concluida por instrucción expresa del usuario, con salvedad de licencia de Sen1Floods11 sin aclarar. Entorno, recuperación, CUDA, recursos y accesos SAM/DINOv3 verificados. Consulta al publicador enviada y archivada; se revisará/reabrirá la etapa al recibir respuesta. Este cierre no acredita permiso sobre el dataset. El historial de validación se conserva abajo.

## Entregables

| Archivo | Función |
| --- | --- |
| [01_entorno_colab.ipynb](../../notebooks/01_entorno_colab.ipynb) | Montaje, fuente verificada, bootstrap, secretos/accesos y prueba prepare/recover; sin salidas inventadas. |
| [environment.yaml](../../configs/environment.yaml) | Configuración YAML 1.2 en sintaxis JSON para cargarla con stdlib: semilla, precisión, presupuestos, IDs y revisiones. |
| [requirements-colab.in](../../requirements-colab.in), [requirements-colab.txt](../../requirements-colab.txt) | 15 dependencias directas y lock de 50 distribuciones Linux/Python 3.12 con hashes. |
| [runtime.py](../../src/auto_prompt_sam2/runtime.py) | Semilla, raíz, inventario, procedencia, importaciones y verificación de versiones/commit SAM. |
| [artifacts.py](../../src/auto_prompt_sam2/artifacts.py) | Generaciones inmutables, marcadores, integridad, identidad y estados de run. |
| [access.py](../../src/auto_prompt_sam2/access.py) | Metadatos limitados y HEAD; no descarga pesos ni cuerpos de datos. |
| [scripts](../../scripts/) | Bootstrap, auditoría pública, comprobación de acceso con stdin, exportación del ZIP y validaciones local/Colab. |
| [test_stage1.py](../../tests/test_stage1.py) | Casos de recuperación en proceso nuevo, corrupción/incompletitud, cambios de identidad, rutas, estados y secreto ficticio. |
| [accesos_y_licencias.md](accesos_y_licencias.md), [cuaderno_decisiones.md](cuaderno_decisiones.md) | Fuentes, bloqueos, justificación, uso de IA y límites. |

## Comprobaciones locales iniciales

Evidencia principal: [local_validation.json](evidence/stage1/local_validation.json), [public_sources.json](evidence/stage1/public_sources.json) y [dependency_resolution.json](evidence/stage1/dependency_resolution.json).

| Comprobación | Resultado real y alcance |
| --- | --- |
| Resolución de dependencias | `uv 0.8.22 pip compile`, destino CPython 3.12, `x86_64-manylinux_2_28`: 50 paquetes resueltos, versiones y hashes guardados. Es resolución por metadatos, no instalación ejecutada en Linux. |
| Instalación local limpia | Se creó Python 3.12.11 en caché del proyecto y `.venv-validation`, sin paquetes globales. Se instalaron las 15 dependencias directas fijadas y el código SAM del SHA elegido, sin extensión CUDA. Las dependencias de plataforma Windows difieren del lock Linux y están enumeradas en el manifiesto. |
| `pip check` | Código de salida 0; sin requisitos incompatibles en ese entorno local. |
| Importaciones | torch `2.7.1+cpu`, torchvision `0.22.1+cpu`, `sam2.build_sam.build_sam2`, `DINOv3ViTModel` y `AutoImageProcessor` importan. No se instanció ni ejecutó ningún modelo. |
| Pines/procedencia | Las 15 versiones directas y `direct_url.json` de SAM coinciden. El código fuente se identifica por hashes y se conserva en ZIP dentro del run. |
| Persistencia | prepare y recover en intérpretes independientes: contenido y SHA-256 coinciden; una escritura sin marcador y una identidad distinta se rechazan. Se conserva un ticket persistente y se recupera el run sin variables del proceso anterior. |
| Semillas | Python, NumPy y torch CPU producen los mismos valores de prueba en ambos procesos con semilla 42. No prueba RNG CUDA ni determinismo de operaciones de modelos futuros. |
| Nueve pruebas automatizadas | Todas aprobadas; incluye fallback a generación anterior tras corrupción o marcador truncado, path traversal y transición de estados. La primera ejecución detectó un error de normalización de rutas Windows que fue corregido antes de estas evidencias finales. |
| Notebook | JSON leído, sintaxis de todas las celdas Python comprobada; 9 celdas, sin outputs ni execution counts. Sus celdas **no se ejecutaron en Colab**. |
| Accesos | Revisiones de SAM/DINOv3 verificadas contra API oficial. SAM HEAD 302; DINOv3 HEAD anónimo 401. Catálogos Sen1Floods11 accesibles, ambos proprietary; HEAD S2Hand y LabelHand 200 sin cuerpo. Solo metadatos del primer objeto listado, sin preparar inventarios ni splits. |
| Secretos | Prueba con valor ficticio confirma que HEAD y su resultado no lo imprimen y que no se sigue la redirección. No se leyó ningún secreto real ni se comprobó `userdata` en Colab. |

Compatibilidad documental: la pareja torch/torchvision se publica en [versiones oficiales de PyTorch](https://pytorch.org/get-started/previous-versions/); SAM declara Python ≥3.10, torch ≥2.5.1 y torchvision ≥0.20.1 en su [setup fijado](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/setup.py). [Transformers 4.56.2 documenta DINOv3](https://huggingface.co/docs/transformers/v4.56.2/en/model_doc/dinov3). La selección satisface esos mínimos y pasó importaciones CPU, sin extrapolar a Colab/CUDA.

## Hardware observado localmente

Windows 10, Python 3.12.11, 8 procesadores lógicos. RAM total **15.84 GiB**, disponible **5.09 GiB** en el inventario inicial; disco del proyecto aproximadamente **473.78 GiB libres**. `nvidia-smi` detectó **GeForce GTX 1650, 4096 MiB totales, 3935 MiB libres**, driver **572.70**. Estos valores son una observación puntual y están guardados en bytes/MiB en el JSON.

El wheel local de torch es CPU: `cuda_available=false`, runtime CUDA y cuDNN `null`; **no se ejecutó la prueba CUDA**, aunque el equipo tenga GPU física. No se atribuye este hardware a Colab. No se midió la memoria que consumirán SAM/DINOv3 ni se comprobó que esta GPU sea suficiente.

## Repetir y completar lo pendiente

Local, usando el entorno preparado (PowerShell):

```powershell
.venv-validation/Scripts/python scripts/run_local_validation.py
python scripts/audit_public_sources.py
python scripts/export_colab_bundle.py
```

La primera orden requiere las dependencias instaladas; la segunda usa red pública y no usa secretos; la tercera produce `artifacts/auto-prompt-sam2-stage1.zip` y su SHA-256. `artifacts/`, cachés y entornos virtuales están ignorados por Git. El informe JSON local versionado basta para auditar la prueba; los artefactos originales permanecen en la raíz local indicada en él. El ZIP es una forma de transportar los cambios aún no publicados, no un sustituto del repositorio.

En Colab, siga el notebook desde una sesión limpia: subir ZIP a Drive, configurar hash y raíz persistente, montar Drive, instalar en venv, ejecutar accesos y prepare. **Eliminar el runtime**, iniciar otro, montar el mismo Drive y ejecutar las mismas celdas con recover. La identidad y los paquetes deben coincidir; de lo contrario use el ZIP original o cree un run nuevo, sin sobrescribir el anterior. Se guarda el hardware de ambas sesiones aunque cambie la GPU asignada.

Pendientes al entregar la versión inicial (ver actualización Colab más abajo): instalación Linux del lock, construcción de sdists bajo su toolchain, imports/CUDA en Colab, Drive FUSE y su durabilidad tras desconexión, cuota real de Drive, acceso a `userdata`, HEAD autorizado de DINOv3. El bootstrap detiene incompatibilidades de Python/plataforma y las validaciones registran falta de GPU/disco. Los presupuestos son planificados, no una reserva real de cuota.

El cierre definido originalmente requería **tres condiciones**: (1) evidencia del ensayo en dos runtimes Colab y CUDA disponible, (2) acceso autorizado de DINOv3 mediante secreto, (3) licencia/permiso primario de Sen1Floods11 aclarado para el uso previsto. Los hashes de pesos/datos quedan nulos por no haber adquisición. La compilación opcional CUDA de SAM y suficiencia para inferencia/entrenamiento deben evaluarse antes de usar esas funciones en las etapas correspondientes.

Se actualizó [estado_actual_fase1.md](estado_actual_fase1.md). Se detiene aquí; no se implementó ni ejecutó ninguna etapa posterior.

## Actualización: notebook Colab aportado por el usuario

Se conserva sin modificar [01_entorno_colab_recover_2026-10-28.ipynb](evidence/stage1/01_entorno_colab_recover_2026-10-28.ipynb), junto con su hash y revisión en [colab_notebook_review.json](evidence/stage1/colab_notebook_review.json). Run: `20261028T221055Z_1d0f6a92f4_aa07b10e`; ZIP usado: SHA-256 `87f10497e0d65ca6e00453c3d4cf0ef240d148d29fc2f94d16eace81d58e2e9c`.

El adjunto muestra montaje de Drive, verificación del ZIP, reparación manual de pip, bootstrap completado sin excepción, comprobaciones de paquetes y tests con retorno 0 y `recover` con retorno 0. Su metadata declara GPU T4 y runtime Colab 2026.07; **no contiene medidas de RAM/VRAM, versión real de Python, resultado CUDA ni respuesta HEAD de DINOv3**. El usuario informa que ejecutó prepare y recover; la eliminación/recreación del runtime es una declaración del usuario, no una observación independiente del asistente.

El retorno 0 de recover, al ejecutar el script del ZIP identificado, implica que pasaron importaciones, los 50 pines Linux y el commit SAM, coincidencia de paquetes con prepare, recuperación/hash del artefacto, semillas CPU y rechazo de escritura incompleta e identidad distinta. **No implica que los bloqueos de recursos o acceso estén vacíos**: esos scripts guardan los bloqueos y pueden terminar con código 0.

La plantilla actual integra las celdas separadas del usuario y captura/imprime stdout y stderr para conservar diagnósticos. Se corrigió el bootstrap: crea el venv sin pip y lo provisiona con el pip del kernel mediante `--python`, sin necesitar ensurepip ni descargar/ejecutar un get-pip no fijado. Es una función documentada por [pip](https://pip.pypa.io/en/stable/topics/python-option/). El cambio pasa una integración local con un venv inicialmente sin pip, pero **aún no se ejecutó esta nueva versión en Colab**. Las nuevas comprobaciones locales están en [colab_update_local_checks.json](evidence/stage1/colab_update_local_checks.json). El informe `local_validation.json` anterior corresponde a su versión histórica, no a los hashes nuevos.

**No sobrescribir el ZIP antiguo en Drive para recuperar este run.** Se preservó además una copia local en `artifacts/archive/`. La plantilla nueva cambia la identidad de fuente y corresponde a runs nuevos. No es necesario rehacer ahora el ensayo que ya completó el usuario para extraer sus informes.

## Exportar los informes del run ya ejecutado

En el notebook original del usuario, después de recover, agregue una celda con lo siguiente. Al abrir el selector, elija el archivo local [export_stage1_evidence.py](../../scripts/export_stage1_evidence.py). No necesita reinstalar el entorno si su sesión sigue conectada.

```python
from google.colab import files
files.upload()  # Elegir scripts/export_stage1_evidence.py del repositorio local.

exporter = Path('/content/export_stage1_evidence.py')
assert exporter.is_file()
output = Path(PERSIST_ROOT) / 'environment' / f'{RUN_ID}-evidence.zip'
result = subprocess.run(
    [str(PYTHON), str(exporter), '--root', PERSIST_ROOT,
     '--run-id', RUN_ID, '--output', str(output)],
    env=dict(child_env, PYTHONPATH=str(REPO / 'src')),
    text=True, capture_output=True,
)
print(result.stdout)
print(result.stderr)
result.check_returncode()
files.download(str(output))
```

El exportador valida marcadores, SHA-256 e identidad guardada, y empaqueta `environment.json`, `recovery_validation.json`, `state.json`, `access.json`, sus marcadores y un manifiesto. No recalcula la identidad con la fuente nueva ni cambia el estado del run. Si falta un informe, lo registra expresamente. No exporta pesos, datos ni la fuente completa; el secreto HF_TOKEN nunca fue un campo de estos informes.

Comparta el ZIP resultante para comprobar `cuda_available`, `cuda_smoke`, memoria/disco, `resource_blockers`, `persistence_smoke`, rechazo de incompletos/identidad, semillas y HEAD autorizado del checkpoint exacto de DINOv3. La aprobación de cuenta comunicada por el usuario se registra, pero no sustituye esa comprobación técnica. Sigue pendiente una fuente de permiso/licencia para S2Hand y LabelHand; no se adquirieron datos ni se inició etapa 2.

## Actualización: ZIP del directorio del run verificado

El usuario aportó `20261028T221055Z_1d0f6a92f4_aa07b10e.zip`. Se comprobaron 10 generaciones completas por tamaño, SHA-256 e identidad y los 15 hashes de archivos dentro del snapshot de fuente. Se conserva [auditoría](evidence/stage1/colab_run_20261028T221055Z/archive_review.json), [entorno](evidence/stage1/colab_run_20261028T221055Z/environment.json) y [recuperación](evidence/stage1/colab_run_20261028T221055Z/recovery_validation.json). El snapshot binario se recibió como `payload.zip`; sus bytes sí coinciden con el marcador de `source_snapshot`, por lo que se registra como cambio de nombre durante el transporte, no corrupción.

Colab: Python 3.12.13, torch 2.7.1+cu126, torchvision 0.22.1+cu126, CUDA 12.6, driver 580.82.07, Tesla T4 con 14.56 GiB de memoria usable según torch; nvidia-smi informa 15360 MiB de VRAM física. RAM total 12.67 GiB, disponible 9.04 GiB al recuperar; disco local libre 63.05 GiB. Los 50 pines y el commit SAM están confirmados. La prueba aritmética CUDA pasó; los manifiestos prepare/recover conservan paquetes e identidad idénticos y `resource_blockers=[]`. No se midió suficiencia para ejecutar modelos.

`persistence_smoke=passed`, rechazo de incompletos/identidad incompatible y reproducción de semillas son verdaderos, y el último estado es complete del ensayo. La escritura incompleta está presente intencionalmente. Los tiempos de prepare/recover y la historia de estados son coherentes con la desconexión declarada por el usuario; los informes no contienen un identificador independiente de sesión.

**El ZIP es correcto para auditar el run, pero no incluye el acceso:** `access` se guarda en `MyDrive/auto-prompt-sam2/environment/access/`, fuera de `environment/runs/<run_id>/`. Basta adjuntar esa carpeta comprimida o usar el exportador anterior para incluirla. No hace falta repetir prepare/recover. Sigue pendiente revisar HEAD autenticado de DINOv3 y aclarar la licencia del dataset.

## Actualización: access.zip verificado

Dos informes de acceso completos verifican SHA-256, tamaño e identidad coincidente con el run. Ambos modelos tienen metadata_matches=true, HEAD 302 y authorized=true en sus archivos/revisiones fijados, sin descargar cuerpos ni seguir redirecciones. Se conservan [informe inicial](evidence/stage1/colab_run_20261028T221055Z/access_initial.json), [último informe](evidence/stage1/colab_run_20261028T221055Z/access.json), marcadores y [auditoría](evidence/stage1/colab_run_20261028T221055Z/access_archive_review.json).

Ya no falta evidencia técnica de acceso a DINOv3. El único bloqueo registrado por ambos informes es la licencia de Sen1Floods11 para S2Hand/LabelHand. No hace falta repetir la prueba de entorno ni buscar otro ZIP. Aclarado ese permiso con una fuente verificable, se podrá registrar el cierre de etapa 1 y autorizar la adquisición/preparación de datos de etapa 2 conforme al alcance acordado.

## Cierre con salvedad por decisión del usuario

Se revisó el EML aportado: solicitud a support@floodbase.com de fecha 2026-10-28 18:05:23 -05:00, sobre S2Hand/LabelHand v1.1 y los usos académicos previstos. Se registra [evidencia mínima](evidence/stage1/licensing_request_sent.json) y [resumen](evidence/stage1/licensing_request_sent.md); el original con cabeceras personales se conserva solo en `artifacts/private/` ignorado por Git.

El usuario indicó expresamente: «Con esto, cierra la etapa 1 de la fase 1. De obtener respuesta, la volvemos a abrir». Se adopta esta excepción al criterio de cierre previo para el seguimiento del proyecto. No se presenta el correo como una concesión de licencia, confirmación de recepción o respuesta del publicador. `license_status` de la configuración y los bloqueos históricos de los informes permanecen intactos; los campos stage1_closed=false de evidencias anteriores no se reescriben retroactivamente.

La etapa 1 se marca concluida con salvedad y se cuenta 1/9. Se reabrirá para evaluar la respuesta y sus condiciones cuando el usuario la aporte. No se adquirieron datos ni se implementó etapa 2 durante este cierre. La determinación de la base de uso del dataset queda pendiente antes de la adquisición; no se considera resuelta por este cambio de seguimiento.
