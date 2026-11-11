# Preparar y publicar los cambios de fase 2 en GitHub

**Fecha:** 11 de noviembre de 2026. Esta guía prepara una publicación del código y la documentación de la demo. Publicar en GitHub no despliega automáticamente un Space ni activa la inferencia. El desarrollo actual no ha creado commits ni ejecutado `push`.

## Alcance de la publicación

- Demo Gradio con carga GeoTIFF/PNG/JPEG, puntos manuales, controles de edición y comparación preparada para pesos futuros.
- Modo `preview` local predeterminado; inferencia reservada para Colab/Spaces con GPU.
- Configuración de pesos, preflight sin modelos, dependencias propias, Dockerfile y pruebas de fase 2.
- README, guía de uso/despliegue, validación, contexto y cuaderno actualizados. Se preservan los resultados experimentales de fase 1.

No hay nuevos resultados de segmentación, entrenamiento, validación GPU o despliegue. Los **17 tests locales** y la revisión de interfaz tienen el alcance descrito en [validacion_fase2.md](validacion_fase2.md). La guía de [fase 2](fase2_demo.md) identifica los checkpoints pendientes y sus hashes.

## Comprobar desde la raíz del repositorio

Con el entorno de preparación ya instalado según el [README](../../README.md):

```powershell
.\.venv-demo\Scripts\python.exe scripts/validate_demo_local.py
git diff --check
git status --short
```

El validador escribe `artifacts/demo-local-validation.json` localmente. Los datos de prueba, capturas y entornos no forman parte de la publicación; tampoco hace falta copiar `descargar.png` del usuario ni `demo-synthetic.tif`. Para probar desde otro equipo se puede cargar una imagen propia compatible.

## Seleccionar los archivos del cambio

Revisar primero el estado de Git. Esta lista corresponde a la implementación y documentación de fase 2 desarrolladas en esta revisión; no abarca cualquier otro trabajo que pueda existir en el árbol local. En particular, `docs/course/entregable2.pptx` y `docs/course/guion.md` ya estaban sin seguimiento y no se incluyen en esta lista.

En PowerShell, desde la raíz:

```powershell
$phase2Files = @(
    '.dockerignore'
    'Dockerfile'
    'app.py'
    'configs/demo.json'
    'requirements-demo.txt'
    'requirements-demo-inference.txt'
    'src/auto_prompt_sam2/demo_backend.py'
    'src/auto_prompt_sam2/demo_config.py'
    'src/auto_prompt_sam2/demo_images.py'
    'src/auto_prompt_sam2/demo_ui.py'
    'scripts/check_demo_artifacts.py'
    'scripts/validate_demo_local.py'
    'tests/demo/test_demo.py'
    'README.md'
    'docs/project/README.md'
    'docs/project/fase2_demo.md'
    'docs/project/validacion_fase2.md'
    'docs/project/publicacion_github.md'
    'docs/project/estado_actual_fase1.md'
    'docs/project/hoja_de_ruta_fase1.md'
    'docs/project/cuaderno_decisiones.md'
    'docs/WAIMLAP/README.md'
)
git add -- $phase2Files
git diff --cached --stat
git diff --cached --check
git diff --cached
```

Si ya había archivos en staging, los comandos de revisión también los mostrarán: comprobar el conjunto completo antes de confirmar. Evitar `git add .` para esta entrega. Los directorios `weights/`, `data/`, `artifacts/`, `.cache/`, `.venv*` y los archivos `.env` están ignorados. Un checkpoint sin extensión llamado `payload` debe guardarse dentro de `weights/`; el nombre `payload` por sí solo no está ignorado globalmente.

Verificar que el diff no contenga tokens, pesos, rasters nuevos, capturas privadas ni cambios en notebooks/evidencias de fase 1. Revisar especialmente que `configs/demo.json` conserve `preview`, descargas deshabilitadas y rutas de pesos vacías como valores predeterminados. Los secretos se configurarán en el runtime remoto.

## Commit y subida, cuando el equipo publique

Después de revisar la selección, una propuesta de mensaje es:

```powershell
git commit -m "feat: preparar demo Gradio con prompts manuales y despliegue Spaces"
git branch --show-current
git remote -v
```

Revisar que la rama y el remoto sean los que el equipo desea publicar. Si se usa un flujo con pull request, publicar la rama de trabajo y abrir el PR correspondiente. Para subir la rama actual al remoto `origin`, cuando esa sea la opción elegida:

```powershell
git push -u origin HEAD
```

No hace falta modificar el commit experimental `f399047464b89a3ed18972f8dc7f1b1364d83a44` registrado en la evidencia ni volver a ejecutar E2. La implementación de la demo tendrá su propio commit. No se debe sobrescribir el archivo `e2/code_commit.txt` del run cerrado en Drive.

## Texto sugerido para el PR

**Título:** Demo Gradio con puntos manuales y preparación para Hugging Face Spaces.

**Descripción:** Añade carga de imágenes satelitales, puntos positivos/negativos y visualización de segmentación preparada para comparar SAM 2.1 tiny congelado con el mejor decoder E2. El modo local permite preparar imágenes y puntos; la inferencia requiere GPU remota y pesos verificados. Incluye configuración, preflight, Dockerfile, pruebas y guía con los archivos exactos pendientes de Colab. Preserva los resultados de fase 1.

**Validación:** 17 pruebas locales aprobadas sin importar modelos; comprobaciones de dependencias, sintaxis, HTTP, imágenes/puntos y conservación de fase 1. Carga y controles revisados en navegador; PNG RGBA del usuario comprobado mediante controlador. Pendientes: checkpoint real, inferencia GPU, construcción del contenedor y despliegue.
