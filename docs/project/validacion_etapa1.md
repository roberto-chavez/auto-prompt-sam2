# Validación de la etapa 1

**Fecha:** 2026-10-28, America/Lima. **Resultado:** implementación disponible y validación local aprobada; cierre de etapa **Bloqueado**. El usuario autorizó continuar localmente y dejar Colab pendiente. No hay resultados de segmentación.

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

## Comprobaciones ejecutadas

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

No probado todavía: instalación Linux del lock, construcción de sdists bajo su toolchain, imports/CUDA en Colab, Drive FUSE y su durabilidad tras desconexión, cuota real de Drive, acceso a `userdata`, HEAD autorizado de DINOv3. El bootstrap detiene incompatibilidades de Python/plataforma y las validaciones registran falta de GPU/disco. Los presupuestos son planificados, no una reserva real de cuota.

Para cerrar faltan **tres condiciones**: (1) evidencia del ensayo en dos runtimes Colab y CUDA disponible, (2) acceso autorizado de DINOv3 mediante secreto, (3) licencia/permiso primario de Sen1Floods11 aclarado para el uso previsto. Los hashes de pesos/datos quedan nulos por no haber adquisición. La compilación opcional CUDA de SAM y suficiencia para inferencia/entrenamiento deben evaluarse antes de usar esas funciones en las etapas correspondientes.

Se actualizó [estado_actual_fase1.md](estado_actual_fase1.md). Se detiene aquí; no se implementó ni ejecutó ninguna etapa posterior.
