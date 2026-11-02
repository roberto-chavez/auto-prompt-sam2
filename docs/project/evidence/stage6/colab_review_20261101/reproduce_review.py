"""Reproduce CPU audit from repository root; requires archived ZIPs and stage-2 data.
Uses archived experimental sources, not current documentation hashes.
"""
import json,zipfile,sys,hashlib,shutil
from pathlib import Path
import numpy as np
R=Path.cwd(); sys.path.insert(0,str(R/'src'))
from auto_prompt_sam2.baseline import stage2_inputs
from auto_prompt_sam2.stage5 import plan,supervised_labels,unpack
from auto_prompt_sam2.metrics import chip_metrics
from auto_prompt_sam2.artifacts import digest,sha256_file
out=R/'docs/project/evidence/stage6/colab_review_20261101'; out.mkdir(parents=True,exist_ok=True)
archive=R/'artifacts/stage6-evidence'; archive.mkdir(exist_ok=True)
data=stage2_inputs(R/'artifacts/stage2-persistent',json.loads((R/'configs/data.yaml').read_text()))
nb=out/'06_entrenamiento_decoder_ejecutado_colab.ipynb'; n=json.loads(nb.read_text('utf-8'))
result={'date':'2026-11-01','models_executed_locally':False,'regimes':{},'notebook_sha256':sha256_file(nb),'notebook_counter_exception':'prepare execution_count null; original outputs preserved, resume counts 1/2/3','limitations':['Large checkpoints remain on Drive: hashes and verified publication/reload in Colab, no independent local deserialization.','Frozen DINO features reused, not recomputed.']}
for regime in ['official','leave_paraguay_out']:
 p=archive/f'{regime}_evidence.zip'
 with zipfile.ZipFile(p) as z:
  c=json.loads(z.read('config.json')); v=json.loads(z.read('validation.json')); env=json.loads(z.read('environment.json')); par=json.loads(z.read('parameters.json'))
  assert c['manifest_sha256']==digest(data[regime]['manifest'])
  assert c['preprocessing_sha256']==digest(data[regime]['params'])
  assert all(hashlib.sha256(z.read('source/'+name)).hexdigest()==h for name,h in c['source_sha256'].items())
  rows=plan(data[regime]['manifest'],'val'); labels={r['id']:supervised_labels(R/'artifacts/stage2-persistent',data[regime]['manifest'],r['id'],'val') for r in rows}
  count=0
  for epoch in range(1,11):
   arrays,meta=unpack(z.read(f'epoch_{epoch}_val_masks.npz')); metrics=json.loads(z.read(f'epoch_{epoch}_metrics.json'))['val']
   for row,mask,saved in zip(rows,arrays['masks'],metrics):
    from auto_prompt_sam2.metrics import confusion
    assert row['id']==saved['id']; assert confusion(labels[row['id']],mask)==saved['confusion']; count+=1
  fixed=json.loads(z.read('fixed_prompts.json'))
  assert set(fixed)==set(c['ids']['train']+c['ids']['val'])
  assert all(name.startswith('sam_mask_decoder.') for name in par['trainable'])
  curves=v['progress']['curves']
  coverage={role:sum(bool(fixed[i]['components']) for i in c['ids'][role]) for role in ['train','val']}
  for curve in curves:
   assert curve['contributing']+curve['no_valid']==coverage['train']; assert curve['no_components']==len(c['ids']['train'])-coverage['train']
  small={k:v['progress'][k] for k in ['epoch','steps','best_epoch','best_iou','gradient_checks','invalid_loss_checks','recovery']}
  result['regimes'][regime]={'zip_sha256':sha256_file(p),'validation_masks_recomputed':count,'source_files_verified':len(c['source_sha256']),'coverage':coverage,'total':{k:len(c['ids'][k]) for k in coverage},'progress':small,'baseline':v['baseline_stage5_val_iou'],'delta':small['best_iou']-v['baseline_stage5_val_iou'],'environment':env,'curves':[{'epoch':x['epoch'],'iou':x['val_metrics']['pooled']['values']['water_iou'],'train_loss':x['train_loss']} for x in curves]}
  for name in ['validation.json','environment.json','recovery_validation.json','parameters.json','curves.json','config.json','inventory.json']:
   folder=out/regime; folder.mkdir(exist_ok=True); (folder/name).write_bytes(z.read(name))
  sys.path.insert(0,str(R/'scripts'))
  from review_stage6_evidence import review
  (out/regime/'portable_review.json').write_text(json.dumps(review(p),indent=2)+'\n',encoding='utf-8')
result['identical_sam_base_states']=json.loads((out/'official/parameters.json').read_text())['base']==json.loads((out/'leave_paraguay_out/parameters.json').read_text())['base']
assert result['identical_sam_base_states']
result['notebook_error_outputs']=sum(o['output_type']=='error' for c in n['cells'] for o in c.get('outputs',[]))
assert result['notebook_error_outputs']==0
result['status']='accepted_stage6_with_documented_limits'
(out/'review.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
for r,v in result['regimes'].items(): print(r,v['coverage'],v['progress'],v['environment'].keys())
