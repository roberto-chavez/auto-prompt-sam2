"""Generate the two E2 Colab notebooks (no outputs, never executed) with the REAL bundle SHA-256.

Order: export bundle -> compute SHA -> build notebooks. Usage:
    python scripts/export_e2_bundle.py && python scripts/build_e2_notebooks.py [--out DIR]
"""
import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO / "src"))
from auto_prompt_sam2.artifacts import sha256_file

SETUP = '''from pathlib import Path
import hashlib, json, os, subprocess, sys, zipfile
from google.colab import drive

drive.mount('/content/drive')
PERSIST_ROOT = '/content/drive/MyDrive/auto-prompt-sam2'            # carpeta de trabajo en TU Drive
BUNDLE = Path('/content/drive/MyDrive/auto-prompt-sam2-e2.zip')    # ZIP que subiste a la raíz de "Mi unidad"
EXPECTED_BUNDLE_SHA256 = '@@SHA@@'                                  # SHA-256 real del ZIP (calculado al generar este notebook)
REPO = Path('/content/auto-prompt-sam2-e2')
PYTHON = Path('/content/auto-prompt-venv/bin/python')

assert BUNDLE.is_file(), f'No encuentro {BUNDLE}. Sube auto-prompt-sam2-e2.zip a la raíz de Mi unidad en Drive.'
actual = hashlib.sha256(BUNDLE.read_bytes()).hexdigest()
assert actual == EXPECTED_BUNDLE_SHA256, f'ZIP distinto al esperado: {actual}'
print('ZIP verificado, SHA-256 =', actual)
REPO.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(BUNDLE) as archive:
    for item in archive.infolist():
        assert (REPO / item.filename).resolve().is_relative_to(REPO.resolve()), 'Ruta insegura en el ZIP'
    archive.extractall(REPO)

child_env = dict(os.environ, PYTHONPATH=str(REPO / 'src'),
    AUTO_PROMPT_PERSIST_ROOT=PERSIST_ROOT, PYTHONHASHSEED='42',
    CUBLAS_WORKSPACE_CONFIG=':4096:8', HF_HUB_DISABLE_TELEMETRY='1',
    AUTO_PROMPT_EXECUTION_CONTEXT='google_colab_user_notebook')
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

PREREQ = '''# E2 NO descarga ni reconstruye datos: lee lo que dejó la etapa 2 del equipo en PERSIST_ROOT.
root = Path(PERSIST_ROOT)
required = {
    'raw/acquisition (adquisición Sen1Floods11)': root / 'raw/acquisition',
    'stage2/validation (informe de la etapa 2)': root / 'stage2/validation',
    'regimes/official/manifests': root / 'regimes/official/manifests',
    'regimes/official/preprocessing': root / 'regimes/official/preprocessing',
}
missing = [name for name, path in required.items() if not path.exists()]
for name, path in required.items():
    print(('OK       ' if path.exists() else 'FALTA    ') + name)
assert not missing, ('Faltan datos de la etapa 2 en tu Drive. NO continúes: pide a tu compañero que comparta la carpeta '
                     '"auto-prompt-sam2" (o ejecuta el notebook 02 del repositorio con su ZIP). Ver PASOS_E2.md, sección 3.')
'''

RESULTS = '''from IPython.display import display, Image, HTML
import csv, html
EVIDENCE = Path(PERSIST_ROOT) / 'e2/evidence'
RUNS = sorted(p for p in EVIDENCE.iterdir() if p.is_dir() and (p / 'report.json').is_file())
assert RUNS, 'No hay un prepare completo. Conserva e2/notebook_logs/prepare.log y envíalo.'
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
print('Limitaciones declaradas:'); [print(' -', item) for item in REPORT['limitations']]
'''


def md(text, i):
    return {"cell_type": "markdown", "id": f"md{i:02d}", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text, i):
    return {"cell_type": "code", "id": f"cd{i:02d}", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


def notebook(cells, name):
    return {"cells": cells, "metadata": {"colab": {"name": name, "provenance": [], "gpuType": "T4"}, "accelerator": "GPU",
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


def build(sha):
    s = lambda t: t.replace("@@SHA@@", sha)
    nb01 = notebook([
        md("""
# 01 · Entorno Colab para E2 (versión preparada)

**NO EJECUTADO.** Este cuaderno solo prepara el entorno (ZIP verificado + venv + pines + GPU). No descarga datos ni pesos, no entrena y **no demuestra nada por sí mismo**. La copia intacta del original está en `originales/01_entorno_colab_original.ipynb`.

Qué se corrigió respecto del original: el hash del ZIP ya no está vacío (es el SHA-256 real del ZIP de E2), la ruta apunta al ZIP de E2 (`auto-prompt-sam2-e2.zip`, que **no contiene notebooks**, así que no hay dependencia circular con su propio hash), se conservan las comprobaciones de integridad y se añade la comprobación explícita de GPU. Las verificaciones de accesos DINOv3/HEAD (etapa 1) ya las cerró el equipo en `main` y no se repiten aquí.

Antes: Entorno de ejecución > Cambiar tipo de entorno > **T4 GPU**. Sube `auto-prompt-sam2-e2.zip` a la raíz de *Mi unidad*.
""", 0),
        code(s(SETUP), 1), code(GPU, 2),
        md("Listo. Si las dos celdas terminaron sin error, abre `02_E2_teardown_intervencion.ipynb` y ejecuta sus celdas en orden. Este cuaderno 01 **no** acredita E2.", 3),
    ], "01_entorno_colab.ipynb")
    nb02 = notebook([
        md("""
# 02 · E2 — Intervención preliminar sobre el decodificador de SAM 2.1 (Sen1Floods11)

**NO EJECUTADO. Solo E2.** Compara SAM 2.1 tiny con el *mask decoder* **congelado** (época 0) frente al **ajustado** (solo ese bloque; encoder de imagen y de prompts congelados), con pérdida BCE+Dice sobre un subconjunto pequeño y explícito de entrenamiento y evaluación en un subconjunto de validación.

* **Referencia asistida por etiquetas, no desplegable:** los puntos (1 positivo + 1 negativo por componente de agua) se simulan a partir de las etiquetas. No es el flujo automático DINOv3 → SAM y **DINOv3 no se usa**.
* **Preliminar:** semilla única (42), subconjuntos pequeños, sin test ni Bolivia. Nada aquí generaliza a otros eventos.
* La hipótesis se guarda en Drive **antes** de entrenar (celda de `prepare`). Los resultados reales aparecerán solo tras ejecutar en tu Colab.
* La línea base (etapa 3) y el teardown (etapa 4) ya existen en el repositorio; E2 los reutiliza (mismo evaluador, mismos prompts, mismo modelo verificado por hash).
""", 0),
        md("## 1. Preparar el entorno (ZIP verificado, venv, pines)\nRequiere GPU T4 y que `auto-prompt-sam2-e2.zip` esté en la raíz de *Mi unidad*.", 1),
        code(s(SETUP), 2), code(GPU, 3),
        md("## 2. Comprobar prerrequisitos de la etapa 2\nSi falta algo, la celda se detiene con un mensaje claro: no se reconstruyen splits ni estadísticas aquí.", 4),
        code(PREREQ, 5),
        md("## 3. Hipótesis y configuración (antes de entrenar)\nSe muestran exactamente los valores de `configs/e2_intervention.yaml`. No se cambian después de ver resultados.", 6),
        code("print(json.dumps(json.loads((REPO / 'configs/e2_intervention.yaml').read_text()), indent=2, ensure_ascii=False))", 7),
        md("""
## 4. Ejecutar E2 (`prepare`)
Hace en orden: pruebas unitarias, verificación del hash de SAM, **guardado de la hipótesis**, comprobación de paridad con el predictor de la etapa 3, época 0 (decoder congelado), entrenamiento solo del decoder (con auditoría de gradientes), evaluación de la mejor época y exportación de evidencia. Imprime el tiempo de cada época: no se estimó antes.

`prepare` crea un run nuevo. Si Colab se desconecta a medio entrenamiento: reabre, ejecuta las celdas 1–2 y cambia `MODE` a `'resume'` (mismo ZIP y mismo Drive; nunca dos escritores a la vez).
""", 8),
        code("""MODE = 'prepare'   # 'resume' SOLO para continuar un run interrumpido
assert MODE in ('prepare', 'resume')
command([str(PYTHON), str(REPO / 'scripts/run_e2.py'), MODE, '--root', PERSIST_ROOT], MODE + '.log')""", 9),
        md("## 5. Ver resultados medidos\nTodo lo mostrado sale de `report.json` y archivos del run; no hay números escritos a mano.", 10),
        code(RESULTS, 11),
        md("""
## 6. Recuperar en un runtime nuevo (`recover`)
1. Anota el ZIP de evidencia impreso arriba y **descárgalo** (celda siguiente).
2. *Entorno de ejecución > Desconectar y eliminar el entorno*. Vuelve a conectar con T4 GPU.
3. Ejecuta **solo** las celdas 1–2 (entorno y prerrequisitos) y la celda de abajo. **No repitas `prepare`.**

`recover` recarga SAM desde Drive en modo *offline*, verifica el hash del checkpoint, recarga el mejor decoder guardado y vuelve a evaluar ambos brazos; deben coincidir con lo guardado (IoU agrupado a 1e-6 y confusiones por chip idénticas). Si falla, **no cambies las tolerancias**: guarda el log.
""", 12),
        code("command([str(PYTHON), str(REPO / 'scripts/run_e2.py'), 'recover', '--root', PERSIST_ROOT], 'recover.log')", 13),
        md("## 7. Descargar la evidencia\nEl ZIP contiene informe, tablas, curvas, ejemplos, predicciones y puntos. **No** incluye checkpoints, pesos, tokens ni imágenes de entrenamiento (la licencia de Sen1Floods11 sigue pendiente).", 14),
        code("""from google.colab import files
ZIPS = sorted((Path(PERSIST_ROOT) / 'e2/evidence').glob('*.zip'))
assert ZIPS, 'No hay ZIP de evidencia'
print(ZIPS[-1], hashlib.sha256(ZIPS[-1].read_bytes()).hexdigest())
files.download(str(ZIPS[-1]))""", 15),
        md("""
## 8. Qué devolver y qué NO afirmar
Devuelve: **este notebook ejecutado** (Archivo > Descargar > .ipynb, con salidas y errores), el ZIP de evidencia y, si hubo errores, `e2/notebook_logs/*.log`.

E2 **no queda acreditada** por crear este cuaderno ni por correrlo: falta revisar los resultados, interpretarlos (incluyendo fallos y límites) y completar el cuaderno de decisiones con tus conclusiones. IA: Claude preparó código, pruebas y documentación; la comprensión y la defensa oral son tuyas.
""", 16),
    ], "02_E2_teardown_intervencion.ipynb")
    return nb01, nb02


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=REPO / "artifacts/notebooks")
    parser.add_argument("--repo-copy", type=Path, default=REPO / "notebooks/05_e2_intervencion_decoder.ipynb")
    args = parser.parse_args()
    bundle = REPO / "artifacts/auto-prompt-sam2-e2.zip"
    sha = sha256_file(bundle)
    nb01, nb02 = build(sha)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, nb in (("01_entorno_colab.ipynb", nb01), ("02_E2_teardown_intervencion.ipynb", nb02)):
        (args.out / name).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", "utf-8")
    args.repo_copy.write_text(json.dumps(nb02, indent=1, ensure_ascii=False) + "\n", "utf-8")
    print("bundle sha256", sha, "->", args.out)


if __name__ == "__main__":
    main()
