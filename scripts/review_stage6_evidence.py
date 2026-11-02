"""Read-only portable review. Never unpickles checkpoints or executes notebook code."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from auto_prompt_sam2.artifacts import digest
from auto_prompt_sam2.metrics import aggregate, scores
from auto_prompt_sam2.stage5 import unpack


def review(path):
    with zipfile.ZipFile(path) as z:
        if len(z.namelist()) != len(set(z.namelist())) or z.testzip() is not None:
            raise ValueError('Duplicate/corrupt ZIP entry')
        config=json.loads(z.read('config.json'))
        report=json.loads(z.read('validation.json'))
        assert report['identity']==config and report['regime']==config['regime']
        assert not report['reserved_access'] and report['final_evaluation'] is False
        assert report['stop_reason']=='fixed_epoch_budget'
        for name, sha in config['source_sha256'].items():
            assert hashlib.sha256(z.read('source/'+name)).hexdigest()==sha
        inventory=json.loads(z.read('inventory.json'))
        for item in inventory:
            assert item['identity']==config and item['identity_sha256']==digest(config)
        # Exported report/array bytes must match a completed persistent generation.
        for name in z.namelist():
            if name.startswith('source/') or name=='inventory.json':
                continue
            key=Path(name).stem
            matches=[m for m in inventory if Path(m['path']).parent.name==key]
            assert any(m['sha256']==hashlib.sha256(z.read(name)).hexdigest() and m['size']==len(z.read(name)) for m in matches), name
        p=report['progress']
        assert p['epoch']==config['config']['epochs'] and len(p['curves'])==p['epoch']
        assert p['gradient_checks'] and p['invalid_loss_checks']
        assert p['recovery']['different_runtime']
        assert p['recovery']['state_bitwise_equal'] and p['recovery']['next_step_replay_bitwise_equal']
        for curve in p['curves']:
            epoch=curve['epoch']
            chip=json.loads(z.read(f'epoch_{epoch}_metrics.json'))
            arrays, metadata=unpack(z.read(f'epoch_{epoch}_val_masks.npz'))
            assert metadata=={'ids':config['ids']['val'],'regime':config['regime'],'epoch':epoch}
            assert arrays['masks'].shape==(len(config['ids']['val']),512,512)
            for role in ('train','val'):
                assert sorted(r['id'] for r in chip[role])==sorted(config['ids'][role])
                assert not any(r['event']=='Bolivia' or (config['regime']=='leave_paraguay_out' and r['event']=='Paraguay') for r in chip[role])
                for row in chip[role]:
                    assert scores(row['confusion'])==row['scores']
                assert aggregate(chip[role])==curve[role+'_metrics']
            assert curve['frozen_unchanged'] and curve['changed_decoder_names']
            assert all(n.startswith('sam_mask_decoder.') for n in curve['changed_decoder_names'])
        best=max(p['curves'],key=lambda c:c['val_metrics']['pooled']['values']['water_iou'])
        assert p['best_epoch']==best['epoch'] and p['best_iou']==best['val_metrics']['pooled']['values']['water_iou']
        assert all(any(Path(m['path']).parent.name==name for m in inventory) for name in ('last','best','decoder_export'))
        return {'status':'portable_consistency_passed_not_stage_closure','regime':config['regime'],
            'zip_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(), 'best_epoch':p['best_epoch'],
            'best_iou':p['best_iou'], 'delta_stage5':p['best_iou']-report['baseline_stage5_val_iou'],
            'pending':['Human review of executed notebook/logs/hardware and coverage',
                'Recompute validation confusion from exported masks and permitted labels',
                'Large checkpoint bytes remain on Drive; inventory is not independent payload verification'],
            'models_executed':False}

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('zip'); parser.add_argument('--output',required=True)
    args=parser.parse_args()
    result=review(args.zip)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
