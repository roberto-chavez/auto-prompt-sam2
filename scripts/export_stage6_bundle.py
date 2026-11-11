"""Export source-only stage-6 package with Git provenance; no data/base weights."""
from pathlib import Path
import json
import subprocess
import zipfile
from run_stage6 import REPO, source_paths
from auto_prompt_sam2.artifacts import sha256_file

def export(target):
    target=Path(target); target.parent.mkdir(parents=True,exist_ok=True)
    def git(*args):
        result=subprocess.run(['git',*args],cwd=REPO,capture_output=True,text=True,encoding='utf-8')
        return result.stdout if result.returncode==0 else None
    provenance={'commit':git('rev-parse','HEAD'), 'status':git('status','--porcelain'),
                'tracked_patch':git('diff','HEAD','--no-ext-diff'),
                'note':'Source snapshot includes untracked implementation files as well.'}
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in source_paths():
            archive.write(path,path.relative_to(REPO).as_posix())
        archive.writestr('stage6_build_provenance.json',json.dumps(provenance,ensure_ascii=False,indent=2))
    return sha256_file(target)

if __name__=='__main__':
    target=REPO/'artifacts/auto-prompt-sam2-stage6.zip'
    print(target)
    print('sha256:',export(target))
