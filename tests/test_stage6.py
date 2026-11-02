"""Stage-6 CPU tests: tensors without backward, no model construction/training."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO/'src'), str(REPO/'scripts')]
from auto_prompt_sam2.losses import masked_bce_dice, union_logits
from auto_prompt_sam2.training import audit_changes, assert_equal_state
from auto_prompt_sam2.artifacts import digest, ArtifactStore, canonical_json
from run_stage6 import verify_bundle, read_dependency, validate_training_config

class Stage6Tests(unittest.TestCase):
    def test_invalid_values_have_no_effect(self):
        x = torch.tensor([[0., 2.], [-3., float('nan')]])
        valid = torch.tensor([[True, True], [False, False]])
        a = masked_bce_dice(x, torch.tensor([[1., 0.], [-1., -1.]]), valid)
        b = masked_bce_dice(x, torch.tensor([[1., 0.], [123., float('nan')]]), valid)
        for k in a:
            self.assertTrue(torch.equal(a[k], b[k]))
        self.assertTrue(torch.isfinite(a['loss']))

    def test_union_is_soft_and_stable(self):
        xs = [torch.tensor([[1000., -1000.], [1., 2.]]), torch.tensor([[-1., -2000.], [3., -2.]])]
        x = union_logits(xs)
        self.assertTrue(torch.equal(x.sigmoid(), torch.stack(xs).sigmoid().max(0).values))
        result = masked_bce_dice(x, torch.ones_like(x), torch.ones_like(x, dtype=torch.bool))
        self.assertTrue(torch.isfinite(result['loss']))
        self.assertIsNone(union_logits([]))

    def test_no_valid_and_invalid_target(self):
        x = torch.zeros((2, 2)); y = torch.full_like(x, -1)
        self.assertIsNone(masked_bce_dice(x, y, torch.zeros_like(x, dtype=torch.bool)))
        with self.assertRaises(ValueError):
            masked_bce_dice(x, y, torch.ones_like(x, dtype=torch.bool))

    def test_frozen_audit(self):
        a = {'image_encoder.x': 'a', 'sam_mask_decoder.x': 'b'}
        b = dict(a, **{'sam_mask_decoder.x': 'c'})
        self.assertEqual(audit_changes(a,b), ['sam_mask_decoder.x'])
        b['image_encoder.x'] = 'd'
        with self.assertRaises(ValueError):
            audit_changes(a,b)

    def test_full_state_comparison(self):
        a = {'optimizer': {0: torch.tensor([1.])}, 'rng': np.array([1,2]), 'epoch': 1}
        assert_equal_state(a, copy.deepcopy(a))
        b = copy.deepcopy(a); b['epoch'] = 2
        with self.assertRaises(AssertionError):
            assert_equal_state(a,b)

    def test_regime_lineage_rejects_mix(self):
        inp = {'manifest': {'regime': 'official'}, 'params': {'upper': .4}}
        ids = {'train': ['a'], 'val': ['b']}
        b = {'regime': 'official', 'preprocessing': inp['params'], 'validation_ids': ['b'], 'models': {},
             'probe': {'lineage': {'regime': 'official', 'role': 'train', 'ids': ['a'],
                 'manifest_sha256': digest(inp['manifest']), 'preprocessing_sha256': digest(inp['params'])}}}
        verify_bundle(b,'official',inp,ids,{})
        for field, value in [('regime','leave_paraguay_out'), ('validation_ids',['test']), ('preprocessing',{})]:
            bad = copy.deepcopy(b); bad[field] = value
            with self.assertRaises(ValueError):
                verify_bundle(bad,'official',inp,ids,{})

    def test_dependency_corruption_blocks(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); st=ArtifactStore(root/'stage4/environment/runs/run')
            identity={'stage':'stage4'}
            gen=st.publish('array',b'ok',identity)
            report={'run_id':'run','reserved_access':[], 'artifacts':{'array':{'generation':gen,
                'sha256':hashlib.sha256(b'ok').hexdigest()}}}
            payload=canonical_json(report)
            st.publish('validation',payload,identity)
            st.publish('state',canonical_json({'state':'complete'}),identity)
            spec={'run_id':'run','validation_sha256':hashlib.sha256(payload).hexdigest()}
            read_dependency(root,'stage4',spec)
            (st.root/'array'/gen/'payload').write_bytes(b'bad')
            with self.assertRaises(FileNotFoundError):
                read_dependency(root,'stage4',spec)

    def test_preregistered_config(self):
        config=json.loads((REPO/'configs/training.yaml').read_text('utf-8'))
        validate_training_config(config)
        config['precision']='float16'
        with self.assertRaises(ValueError):
            validate_training_config(config)

    def test_workstation_guard_before_model(self):
        from auto_prompt_sam2.training import configure_decoder, forward_components
        with self.assertRaises(RuntimeError):
            configure_decoder(None)
        with self.assertRaises(RuntimeError):
            forward_components(None, None, [])

if __name__ == '__main__':
    unittest.main()
