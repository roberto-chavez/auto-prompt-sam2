"""Generate the two E2 Colab notebooks (no outputs, never executed) into notebooks/.

They clone THIS repository from GitHub inside Colab (branch _E.V_), pin the exact commit in the
user's Drive (e2/code_commit.txt) and run everything from that checkout. No ZIP bundles and no
hashes embedded in notebooks, hence no circular dependency between notebooks, commits and hashes.
Usage: python scripts/build_e2_notebooks.py
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "notebooks"

SETUP = '''from pathlib import Path
import json, os, subprocess, sys
from google.colab import drive

drive.mount('/content/drive')
GITHUB_URL = 'https://github.com/roberto-chavez/auto-prompt-sam2.git'
BRANCH = '_E.V_'
PINNED_COMMIT = ''    # '' = usa el commit ya fijado en tu Drive; en la 1.ª ejecución fija la punta actual de la rama
UPDATE_CODE = False   # True SOLO para adoptar código nuevo de la rama (entonces rehaz E2 desde 'prepare')
PERSIST_ROOT = '/content/drive/MyDrive/auto-prompt-sam2'   # datos, pesos, cachés y checkpoints: SOLO aquí
REPO = Path('/content/auto-prompt-sam2')
PYTHON = Path('/content/auto-prompt-venv/bin/python')
Path(PERSIST_ROOT).mkdir(parents=True, exist_ok=True)

def sh(args, cwd=None):
    r = subprocess.run(args, cwd=cwd, text=True, capture_output=True)
    if r.returncode:
        raise RuntimeError(f'{args[:3]} falló:\\n{r.stdout}\\n{r.stderr}')
    return r.stdout.strip()

# --- Código desde GitHub, en una revisión fijada y reproducible --------------------------------------
PIN_FILE = Path(PERSIST_ROOT) / 'e2/code_commit.txt'
existed = REPO.exists()
if not existed:
    sh(['git', 'clone', '--branch', BRANCH, GITHUB_URL, str(REPO)])
elif UPDATE_CODE:
    sh(['git', 'fetch', 'origin', BRANCH], cwd=REPO)
want = PINNED_COMMIT.strip() or ('' if UPDATE_CODE or not PIN_FILE.exists() else PIN_FILE.read_text().strip())
if UPDATE_CODE and existed and not PINNED_COMMIT.strip():
    want = sh(['git', 'rev-parse', 'FETCH_HEAD'], cwd=REPO)
if want:
    sh(['git', 'checkout', '--detach', want], cwd=REPO)
COMMIT = sh(['git', 'rev-parse', 'HEAD'], cwd=REPO)
assert not sh(['git', 'status', '--porcelain'], cwd=REPO), 'El código clonado tiene cambios locales'
if PINNED_COMMIT.strip() or UPDATE_CODE or not PIN_FILE.exists():
    PIN_FILE.parent.mkdir(parents=True, exist_ok=True)
    PIN_FILE.write_text(COMMIT)
print('Código:', GITHUB_URL, '| rama', BRANCH, '| commit', COMMIT)
print('Commit fijado en Drive:', PIN_FILE.read_text().strip())

# --- Entorno ----------------------------------------------------------------------------------------
child_env = dict(os.environ, PYTHONPATH=str(REPO / 'src'),
    AUTO_PROMPT_PERSIST_ROOT=PERSIST_ROOT, PYTHONHASHSEED='42',
    CUBLAS_WORKSPACE_CONFIG=':4096:8', HF_HUB_DISABLE_TELEMETRY='1',
    AUTO_PROMPT_EXECUTION_CONTEXT='google_colab_user_notebook')
try:   # HF_TOKEN se lee de los secretos de Colab (activa "Acceso desde el notebook"); nunca se imprime ni se guarda
    from google.colab import userdata
    child_env['HF_TOKEN'] = userdata.get('HF_TOKEN')
    print('HF_TOKEN leído de los secretos de Colab (valor no mostrado).')
except Exception as error:
    print('HF_TOKEN no disponible (' + type(error).__name__ + '): SAM es público; solo DINOv3 lo necesita.')
LOG_DIR = Path(PERSIST_ROOT) / 'e2/notebook_logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)

def command(args, log_name):
    """Ejecuta un proceso mostrando su salida en vivo y guardándola en Drive."""
    with (LOG_DIR / log_name).open('w', encoding='utf-8') as log:
        process = subprocess.Popen(args, cwd=REPO, env=child_env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
        for line in process.stdout:
            print(line, end='')
            log.write(line)
            log.flush()
        result = process.wait()
    if result:
        raise RuntimeError(f'Proceso terminado con código {result}; revisa la salida y el log {log_name}')

if not PYTHON.is_file():
    command([sys.executable, str(REPO / 'scripts/bootstrap_colab.py'), '--venv', str(PYTHON.parent.parent)], 'bootstrap.log')
command([str(PYTHON), '-m', 'pip', 'install', '--no-deps', '-r', str(REPO / 'requirements-stage2.txt')], 'stage2_pins.log')
command([str(PYTHON), '-m', 'pip', 'check'], 'pip_check.log')
print('Entorno listo.')
'''

GPU = '''# Comprueba que Colab te dio una GPU (Entorno de ejecución > Cambiar tipo > T4 GPU).
command(['nvidia-smi'], 'nvidia_smi.log')
command([str(PYTHON), '-c', 'import torch; assert torch.cuda.is_available(), "NO HAY GPU: cambia el tipo de entorno a GPU"; print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__)'], 'gpu_check.log')
'''

STAGE2 = '''# Descarga Sen1Floods11 v1.1 (446 pares S2Hand/LabelHand) a TU Drive, audita, crea los splits y fija el preprocesamiento.
# Es la parte más lenta. Es reanudable: si Colab se corta, vuelve a ejecutar las celdas de entorno y esta.
command([str(PYTHON), str(REPO / 'scripts/prepare_stage2.py'), '--root', PERSIST_ROOT,
         '--evidence', str(Path(PERSIST_ROOT) / 'stage2/evidence')], 'stage2_prepare.log')'''

STAGE2_CHECK = '''# Segunda pasada sin red: relee todo desde tu Drive y comprueba que coincide.
command([str(PYTHON), str(REPO / 'scripts/prepare_stage2.py'), '--root', PERSIST_ROOT, '--offline',
         '--evidence', str(Path(PERSIST_ROOT) / 'stage2/evidence')], 'stage2_offline.log')
report = json.loads((Path(PERSIST_ROOT) / 'stage2/evidence/validation.json').read_text())
print(json.dumps({k: {'counts': v['counts'], 'parameters_reused': v['parameters_reused'], 'reload': v['reload']}
                  for k, v in report['regimes'].items()}, indent=2))
assert report['models_executed'] is False and report['reserved_pixel_caches'] == []'''

STAGE3 = '''# Reutiliza scripts/run_stage3.py: SAM 2.1 tiny congelado, puntos simulados desde etiquetas (referencia asistida), evaluador común.
MODE3 = 'prepare'
command([str(PYTHON), str(REPO / 'scripts/run_stage3.py'), MODE3, '--root', PERSIST_ROOT], 'stage3_' + MODE3 + '.log')'''

STAGE3_SHOW = '''RUNS3 = sorted(p for p in (Path(PERSIST_ROOT) / 'stage3/evidence').iterdir() if p.is_dir() and (p / 'validation.json').is_file())
assert RUNS3, 'No hay run completo de la etapa 3; revisa e2/notebook_logs/stage3_prepare.log'
R3 = json.loads((RUNS3[-1] / 'validation.json').read_text())
print('run_id:', R3['run_id'], '| estado:', R3['status'])
for regime, value in R3['regimes'].items():
    v = value['summary']['val']['pooled']['values']
    print(regime, '-> val agrupado: IoU agua', v['water_iou'], '| precisión', v['precision'], '| recall', v['recall'], '| Dice', v['dice'])'''

STAGE4 = '''# Teardown del modelo base (SAM 2.1 tiny): parámetros/FLOPs por bloque, features intermedias, ERF, latencia y memoria.
# Solo SAM: DINOv3 es opcional y no bloquea E2.
command([str(PYTHON), str(REPO / 'scripts/run_stage4.py'), 'prepare', '--models', 'sam', '--root', PERSIST_ROOT], 'stage4_sam_prepare.log')'''

STAGE4_SHOW = '''from IPython.display import display, Image, HTML
import csv, html
RUNS4 = sorted(p for p in (Path(PERSIST_ROOT) / 'stage4/evidence').iterdir() if p.is_dir() and (p / 'validation.json').is_file())
assert RUNS4, 'No hay run completo de la etapa 4; revisa e2/notebook_logs/stage4_sam_prepare.log'
RUN4 = RUNS4[-1]
R4 = json.loads((RUN4 / 'validation.json').read_text())
print('run_id:', R4['run_id'], '| modelos:', R4['identity'].get('models_run'))
print(json.dumps(R4['interpretation']['observed'], indent=1)[:3000])
for table in sorted(RUN4.glob('sam_*_parameters.csv'))[:1] + sorted(RUN4.glob('sam_*_flops.csv'))[:1] + sorted(RUN4.glob('sam_*_timings.csv'))[:1]:
    rows = list(csv.reader(table.open(encoding='utf-8', newline='')))
    print(table.name)
    display(HTML('<table>' + ''.join('<tr>' + ''.join('<td>' + html.escape(v) + '</td>' for v in row) + '</tr>' for row in rows) + '</table>'))
for figure in sorted(RUN4.glob('sam_*_0_figure.png'))[:1]:
    print(figure.name)
    display(Image(filename=str(figure)))
print('Evidencia completa:', RUN4.with_suffix('.zip'))'''

DINO = '''RUN_DINO = False   # True = repetir el teardown incluyendo DINOv3 (gated; requiere HF_TOKEN con acceso aprobado). NO es necesario para E2.
if RUN_DINO:
    command([str(PYTHON), str(REPO / 'scripts/run_stage4.py'), 'prepare', '--models', 'sam,dino', '--root', PERSIST_ROOT], 'stage4_sam_dino_prepare.log')
else:
    print('DINOv3 omitido (opcional).')'''

PREREQ = '''# E2 lee lo que dejó TU ejecución del notebook E2_01 en TU Drive. Nada depende de archivos de otra persona.
root = Path(PERSIST_ROOT)
required = {
    'raw/acquisition (descarga Sen1Floods11)': root / 'raw/acquisition',
    'stage2/validation (datos preparados)': root / 'stage2/validation',
    'regimes/official/manifests': root / 'regimes/official/manifests',
    'regimes/official/preprocessing': root / 'regimes/official/preprocessing',
}
for name, path in required.items():
    print(('OK       ' if path.exists() else 'FALTA    ') + name)
assert all(p.exists() for p in required.values()), 'Faltan datos: ejecuta primero E2_01_datos_baseline_teardown.ipynb completo (celda de la etapa 2).'
for label, folder in (('línea base (etapa 3)', root / 'stage3/evidence'), ('teardown (etapa 4)', root / 'stage4/evidence')):
    done = sorted(p.name for p in folder.glob('*/validation.json')) if folder.exists() else []
    print(('OK       ' if done else 'AVISO    ') + label + ': ' + ('run en tu Drive' if done else 'aún no ejecutada en tu Drive (E2 puede correr, pero la evidencia de Fase A estará incompleta)'))'''

RESULTS = '''from IPython.display import display, Image, HTML
import csv, html
EVIDENCE = Path(PERSIST_ROOT) / 'e2/evidence'
RUNS = sorted(p for p in EVIDENCE.iterdir() if p.is_dir() and (p / 'report.json').is_file())
assert RUNS, 'No hay un prepare completo. Conserva e2/notebook_logs/prepare.log.'
RUN_DIR = RUNS[-1]
REPORT = json.loads((RUN_DIR / 'report.json').read_text())
print('run_id:', REPORT['run_id'], '| estado:', REPORT['status'])
print(json.dumps({'hipotesis_previa': REPORT['hypothesis']['statement'], 'brazos': REPORT['arms'],
                  'mejor_epoca': REPORT['best_epoch'], 'verificacion_hipotesis': REPORT['hypothesis_check'],
                  'congelamiento': {k: REPORT['freezing'][k] for k in ('frozen_blocks_unchanged', 'decoder_changed')},
                  'paridad_con_etapa3': REPORT['parity_with_stage3_predictor'],
                  'subconjuntos': {k: REPORT['subsets'][k] for k in ('val_stats', 'train_stats')}},
                 indent=2, ensure_ascii=False))
for name in ('history.csv', 'per_chip_val.csv'):
    rows = list(csv.reader((RUN_DIR / name).open(encoding='utf-8', newline='')))
    print(name)
    display(HTML('<table>' + ''.join('<tr>' + ''.join('<td>' + html.escape(v) + '</td>' for v in row) + '</tr>' for row in rows) + '</table>'))
display(Image(filename=str(RUN_DIR / 'curves.png')))
for figure in sorted((RUN_DIR / 'examples').glob('*.png')):
    print(figure.name)
    display(Image(filename=str(figure)))
print('Limitaciones declaradas:'); [print(' -', item) for item in REPORT['limitations']]'''

DRAFT = '''EVID = Path(PERSIST_ROOT) / 'e2/evidence'
RUN_DIR = sorted(p for p in EVID.iterdir() if p.is_dir() and (p / 'report.json').is_file())[-1]
# Borrador (solo cifras medidas) para pegar en docs/project/cuaderno_decisiones.md; la interpretación queda PENDIENTE.
command([str(PYTHON), str(REPO / 'scripts/e2_decision_log_draft.py'), str(RUN_DIR)], 'decision_log_draft.log')
print('\\nArchivo:', RUN_DIR / 'e2_resultados_para_cuaderno_decisiones.md')'''

ZIPCELL = '''from google.colab import files
import hashlib
# Regenera el ZIP incluyendo el borrador del cuaderno de decisiones y el commit de código usado.
import zipfile
EVID = Path(PERSIST_ROOT) / 'e2/evidence'
RUN_DIR = sorted(p for p in EVID.iterdir() if p.is_dir() and (p / 'report.json').is_file())[-1]
(RUN_DIR / 'code_commit.txt').write_text(COMMIT)
target = RUN_DIR.with_name(RUN_DIR.name + '-final.zip')
with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
    for p in sorted(RUN_DIR.rglob('*')):
        if p.is_file():
            archive.write(p, p.relative_to(RUN_DIR.parent).as_posix())
print(target, hashlib.sha256(target.read_bytes()).hexdigest())
files.download(str(target))'''


def md(text, i):
    return {"cell_type": "markdown", "id": f"md{i:02d}", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text, i):
    return {"cell_type": "code", "id": f"cd{i:02d}", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


def notebook(cells, name):
    return {"cells": cells, "metadata": {"colab": {"name": name, "provenance": [], "gpuType": "T4"}, "accelerator": "GPU",
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


def build():
    nb1 = notebook([
        md("""
# E2 · Notebook 1 — Entorno, datos Sen1Floods11, línea base y teardown

**NO EJECUTADO.** Solo E2. Parte 1 de 2: deja en **tu Drive** (`MyDrive/auto-prompt-sam2`) los datos y la evidencia de la línea base (SAM 2.1 tiny congelado) y del teardown. Reutiliza el código del repositorio (`prepare_stage2.py`, `run_stage3.py`, `run_stage4.py`).

* Runtime: **T4 GPU**. Todo el código se clona desde GitHub (rama `_E.V_`) y su commit queda fijado en tu Drive.
* Los datos, pesos, cachés y checkpoints se guardan solo en tu Drive, nunca en GitHub.
* Referencia asistida por etiquetas (no desplegable). No se usa el flujo automático DINOv3→SAM.
* La evidencia ejecutada de tu compañero en `docs/project/evidence/` es antecedente; **no acredita tu ejecución**.
* Licencia de Sen1Floods11: respuesta del publicador pendiente.
""", 0),
        md("## 1. Entorno\nClona el código, fija el commit, crea el entorno aislado y lee `HF_TOKEN` de los secretos de Colab (sin mostrarlo).", 1),
        code(SETUP, 2), code(GPU, 3),
        md("## 2. Datos (etapa 2): descarga y preparación de Sen1Floods11\nDescarga los 446 pares a tu Drive, audita, crea splits y fija estadísticas con **solo train**. No abre test ni Bolivia para ajustar nada.", 4),
        code(STAGE2, 5), code(STAGE2_CHECK, 6),
        md("## 3. Línea base (etapa 3)\nSAM 2.1 tiny congelado sobre train/val con puntos simulados (1 positivo + 1 negativo por componente de agua). Máscara elegida por IoU **predicho**. Las métricas de esta celda son tuyas y **solo se interpretan tras revisar tus salidas**.", 7),
        code(STAGE3, 8), code(STAGE3_SHOW, 9),
        md("## 4. Teardown del modelo base (etapa 4, solo SAM)\nParámetros y FLOPs por bloque, features intermedias, mapa de campo receptivo, latencia y memoria máxima en la GPU que te asignó Colab.", 10),
        code(STAGE4, 11), code(STAGE4_SHOW, 12),
        md("### (Opcional) Incluir DINOv3\nNo es necesario para E2 y no bloquea nada. Requiere acceso aprobado a DINOv3 en Hugging Face y el secreto `HF_TOKEN`.", 13),
        code(DINO, 14),
        md("""
## Siguiente
Abre `E2_02_intervencion_decoder.ipynb` (misma carpeta `notebooks/` de la rama `_E.V_`). Guarda **este cuaderno ejecutado** (Archivo → Descargar → .ipynb) y no cambies de commit entre notebooks (queda fijado en `e2/code_commit.txt`). Nada de esto marca E2 como acreditada.
""", 15),
    ], "E2_01_datos_baseline_teardown.ipynb")
    nb2 = notebook([
        md("""
# E2 · Notebook 2 — Intervención preliminar: decoder de SAM 2.1 congelado vs ajustado

**NO EJECUTADO.** Solo E2. Compara el *mask decoder* **congelado** (época 0) con el **ajustado** (solo ese bloque; encoder de imagen y de prompts congelados), BCE+Dice sobre un subconjunto pequeño de train, evaluado en un subconjunto de val.

* **Referencia asistida por etiquetas, no desplegable**; DINOv3 no se usa. Preliminar: semilla 42, subconjuntos pequeños, sin test ni Bolivia.
* La hipótesis se guarda en tu Drive **antes** de cargar o entrenar el modelo (celda `prepare`).
* Los resultados reales aparecerán solo cuando tú ejecutes. Hasta revisar tus salidas, métricas e interpretación siguen **pendientes**.
* Requisito: haber ejecutado `E2_01_datos_baseline_teardown.ipynb` en tu cuenta. Runtime: **T4 GPU**.
""", 0),
        md("## 1. Entorno (mismo commit que el notebook 1)", 1),
        code(SETUP, 2), code(GPU, 3),
        md("## 2. Prerrequisitos en tu Drive", 4), code(PREREQ, 5),
        md("## 3. Hipótesis y configuración (antes de entrenar)\nTal como están en `configs/e2_intervention.yaml` del commit fijado. No se cambian tras ver resultados.", 6),
        code("print(json.dumps(json.loads((REPO / 'configs/e2_intervention.yaml').read_text()), indent=2, ensure_ascii=False))", 7),
        md("""
## 4. Ejecutar E2 (`prepare`)
En orden: pruebas, verificación del hash de SAM, **guardado de la hipótesis**, paridad con el predictor de la etapa 3, época 0 (decoder congelado), entrenamiento solo del decoder con auditoría de gradientes, evaluación de la mejor época y evidencia. Imprime el tiempo por época (no estimado de antemano).

Si Colab se corta durante el entrenamiento: reconecta, ejecuta las celdas 1–2 y cambia `MODE` a `'resume'` (nunca dos escritores).
""", 8),
        code("""MODE = 'prepare'   # 'resume' SOLO para continuar un run interrumpido
assert MODE in ('prepare', 'resume')
command([str(PYTHON), str(REPO / 'scripts/run_e2.py'), MODE, '--root', PERSIST_ROOT], MODE + '.log')""", 9),
        md("## 5. Ver resultados medidos\nTodo sale de `report.json`; no hay números escritos a mano.", 10),
        code(RESULTS, 11),
        md("## 6. Borrador del cuaderno de decisiones\nCopia solo las cifras medidas a un `.md`. La **interpretación** debe escribirla el equipo.", 12),
        code(DRAFT, 13),
        md("""
## 7. Recuperar en un runtime nuevo (`recover`)
Descarga primero la evidencia (sección 8). Luego *Entorno de ejecución → Desconectar y eliminar*, reconecta con T4, ejecuta **solo** las celdas 1–2 (entorno y prerrequisitos) y la celda siguiente. **No repitas `prepare`.** Recarga SAM desde tu Drive offline, verifica el hash, recarga el mejor decoder y reevalúa ambos brazos; deben coincidir (IoU agrupado a 1e-6 y confusiones por chip idénticas). Si falla, no cambies tolerancias.
""", 14),
        code("command([str(PYTHON), str(REPO / 'scripts/run_e2.py'), 'recover', '--root', PERSIST_ROOT], 'recover.log')", 15),
        md("## 8. Descargar la evidencia\nEl ZIP incluye informe, tablas, curvas, ejemplos, predicciones, puntos, el borrador y el commit de código. **No** incluye checkpoints, pesos, tokens ni imágenes de entrenamiento.", 16),
        code(ZIPCELL, 17),
        md("""
## 9. Qué devolver y qué NO afirmar
Devuelve: **este notebook ejecutado** (y el E2_01 ejecutado), el ZIP final de evidencia y, si hubo errores, `e2/notebook_logs/*.log`. E2 **no queda acreditada** por crear o correr estos cuadernos: falta revisar resultados, interpretarlos (incluyendo fallos y límites) y completar el cuaderno de decisiones. IA: Claude preparó código, pruebas y documentación; la comprensión y la defensa oral son tuyas.
""", 18),
    ], "E2_02_intervencion_decoder.ipynb")
    return {"E2_01_datos_baseline_teardown.ipynb": nb1, "E2_02_intervencion_decoder.ipynb": nb2}


if __name__ == "__main__":
    for name, nb in build().items():
        (OUT / name).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", "utf-8")
        print("wrote", OUT / name)
