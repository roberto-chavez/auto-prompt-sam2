"""Colab-only inference and dependency reuse for stage 7. No training or test API."""
from __future__ import annotations

import json
import time
from pathlib import Path
import numpy as np

from .artifacts import ArtifactStore, digest, sha256_file
from .sam_adapter import require_colab_gpu
from .stage5 import unpack, read_image, supervised_labels, no_label_inference


def checked_read(store, name, item):
    payload, marker = store.read(name, item['identity'], item['generation'])
    if marker['sha256'] != item['sha256']:
        raise ValueError('Pinned artifact hash mismatch: '+name)
    return payload


def prompt_diagnostics(config, meta):
    """Recover actual sampled counts from cached prompts, never nominal k*chips."""
    components = meta.get('prompts', {}).get('components', [])
    grid = config['source'] == 'grid'
    return {'components': len(components),
        'positive_points': 1024 if grid else sum(sum(c['point_labels']) for c in components),
        'negative_points': sum(len(c['point_labels'])-sum(c['point_labels']) for c in components),
        'positive_shortfall': sum(c.get('positive_shortfall', 0) for c in components),
        'negative_shortfall': sum(c.get('negative_shortfall', 0) for c in components),
        'sam_predict_calls': (16 if grid else len(components)),
        'image_encoder': (1 if grid else int(bool(components))),
        'mask_decoder': (16 if grid else len(components)),
        'calls_provenance': ('derived_pinned_single_crop_1024_points_batches_64' if grid else
                             'one_forward_per_persisted_component'),
        'point_budget_scope': 'per_component_not_per_chip',
        'grid_semantic_masks_accepted': len(meta.get('grid', {}).get('accepted_indices', [])) if grid else None}


def stage6_inputs(root, spec, inp, bundle, repo):
    """Read exact reviewed generations, not a floating active/best pointer."""
    identity = spec['identity']
    if (identity['regime'] != bundle['regime'] or identity['manifest_sha256'] != digest(inp['manifest']) or
            identity['preprocessing_sha256'] != digest(inp['params']) or
            identity['bundle_sha256'] != digest(bundle) or
            identity['base_checkpoint_sha256'] != bundle['base_checkpoint_sha256']):
        raise ValueError('Stage-6 lineage differs from stage 2/5')
    # Reuse every original library/config without an implicit implementation change.
    for rel, expected in identity['source_sha256'].items():
        if rel.startswith('src/') or (rel.startswith('configs/') and not rel.endswith('stage6_dependencies.json')):
            if sha256_file(repo/rel) != expected:
                raise ValueError('Prior implementation changed: '+rel)
    path = (root/spec['store_path']).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Dependency path escaped persistent root')
    store = ArtifactStore(path)
    payloads = {n: checked_read(store, n, m) for n, m in spec['artifacts'].items()}
    report = json.loads(payloads['validation'])
    if (report['identity'] != identity or report['reserved_access'] or report['final_evaluation'] or
            report['progress']['epoch'] != 10 or report['progress']['best_epoch'] != spec['best_epoch']):
        raise ValueError('Incomplete or incompatible reviewed decoder')
    masks, meta = unpack(payloads['epoch_'+str(spec['best_epoch'])+'_val_masks'])
    if (meta['ids'] != identity['ids']['val'] or meta['regime'] != identity['regime'] or
            meta['epoch'] != spec['best_epoch'] or masks['masks'].dtype != np.bool_ or
            masks['masks'].shape != (len(meta['ids']), 512, 512)):
        raise ValueError('Best mask IDs/regime differ')
    return {'payloads': payloads, 'fixed': json.loads(payloads['fixed_prompts']),
            'masks': masks['masks'], 'parameters': json.loads(payloads['parameters'])}


class ValidationEngine:
    def __init__(self, root, regime, inp, bundle, prior_store, prior, sixth, predictor, dino, baseline_policy):
        require_colab_gpu()
        self.root, self.regime, self.inp, self.bundle = root, regime, inp, bundle
        self.prior_store, self.prior, self.sixth = prior_store, prior, sixth
        self.predictor, self.dino, self.baseline_policy = predictor, dino, baseline_policy
        self.access_log = []

    def read_prior(self, name):
        return checked_read(self.prior_store, name, self.prior['artifacts'][name])

    def feature(self, index, chip):
        arrays, meta = unpack(self.read_prior(f'{self.regime}_features_val_{index:03}'))
        if (meta['id'] != chip['id'] or meta['regime'] != self.regime or meta['role'] != 'val' or
                meta['image_sha256'] != chip['image_sha256'] or
                meta['preprocessing_sha256'] != digest(self.inp['params'])):
            raise ValueError('Feature cache lineage differs')
        return arrays

    def metric(self, chip, mask):
        from .metrics import chip_metrics, pixel_areas_m2
        labels = supervised_labels(self.root, self.inp['manifest'], chip['id'], 'val')
        self.access_log.append({'id': chip['id'], 'role': 'val', 'regime': self.regime, 'kind': 'labels_metrics_after_inference'})
        return chip_metrics(labels, mask, chip_id=chip['id'], event=chip['event'],
                            pixel_area_m2=pixel_areas_m2(chip['geometry']))

    def pipeline(self, config, chip, cached_features=None):
        """One image, complete path, fresh SAM image embedding on every invocation."""
        require_colab_gpu()
        import torch
        from .preprocessing import prepare, sam_predictor_image
        from .features import token_layout, tokens_to_grid
        from .probe import probabilities
        from .automatic_prompts import probability_map, generate_prompts
        from .reference_prompts import simulate_points
        from .sam_adapter import segment_components, select_multimask
        from .mask_selection import make_grid_generator, select_grid_masks
        from .training import forward_components

        stages = {}

        def timed(key, fn):
            torch.cuda.synchronize()
            start = time.perf_counter()
            value = fn()
            torch.cuda.synchronize()
            stages[key] = (time.perf_counter()-start)*1000
            return value

        rgb, valid = timed('read_and_hash_image', lambda: read_image(
            self.root, self.inp['manifest'], chip['id'], 'val'))
        self.access_log.append({'id': chip['id'], 'role': 'val', 'regime': self.regime, 'kind': 'image_inference'})
        source, route = config['source'], config['route']
        automatic = not source.startswith('assisted')
        probability = np.zeros((512, 512), np.float32)
        prompts, components = {}, []
        if source in ('dino_points', 'grid'):
            if cached_features is None:
                x = timed('preprocess_dino', lambda: prepare(rgb, valid, self.inp['params'], model='dino')['pixel_values'][None].cuda())
                def extract():
                    layout = token_layout((512, 512), 16, self.dino.config.num_register_tokens, 1024)
                    tokens = self.dino(pixel_values=x, return_dict=True).last_hidden_state
                    return tokens_to_grid(tokens, layout)[0].permute(1, 2, 0).cpu().numpy().copy()
                features = timed('dino', extract)
                del x
            else:
                if not np.array_equal(valid, cached_features['input_valid']):
                    raise ValueError('Cached input validity differs from image')
                features = cached_features['features']
            probability = timed('probe_and_probability_resize', lambda: probability_map(
                probabilities(features, self.bundle['probe'], regime=self.regime)))
        if source == 'dino_points':
            prompts = timed('prompts', lambda: generate_prompts(probability, valid, config['policy'], config['positive_threshold']))
            prompts.pop('component_map')
            components = prompts['components']
        elif source.startswith('assisted'):
            policy = self.baseline_policy if source == 'assisted_historical' else dict(
                self.baseline_policy, positive_per_component=config['policy']['positive_per_component'],
                negative_per_component=config['policy']['negative_per_component'],
                minimum_distance_px=config['policy']['minimum_distance_px'])
            def assisted():
                labels = supervised_labels(self.root, self.inp['manifest'], chip['id'], 'val')
                self.access_log.append({'id': chip['id'], 'role': 'val', 'regime': self.regime, 'kind': 'labels_assisted_prompts_only'})
                return simulate_points(labels, regime=self.regime, chip_id=chip['id'], seed=42, policy=policy)
            prompts = timed('assisted_label_read_and_prompts', assisted)
            components = prompts['components']

        image = timed('preprocess_sam', lambda: (prepare(rgb, valid, self.inp['params'], model='sam')['pixel_values'][None].cuda()
            if route == 'stage6_direct' else sam_predictor_image(rgb, valid, self.inp['params'])))
        calls = {'image_encoder': 0, 'mask_decoder': 0}
        handles, direct_decisions = [], []
        for name, module in [('image_encoder', self.predictor.model.image_encoder),
                             ('mask_decoder', self.predictor.model.sam_mask_decoder)]:
            def count(module, args, output, name=name):
                calls[name] += 1
                if route == 'stage6_direct' and name == 'mask_decoder':
                    scores = output[1][0].detach().float().cpu().numpy()
                    direct_decisions.append({'predicted_iou': scores.tolist(), 'selected_index': int(scores.argmax())})
            handles.append(module.register_forward_hook(count))

        def sam():
            if route == 'stage6_direct':
                logits = forward_components(self.predictor.model, image, components)
                return {'mask': np.zeros((512, 512), bool) if logits is None else logits.cpu().numpy() > 0,
                        'predict_calls': len(components), 'selection': 'highest_predicted_iou_stage6_direct',
                        'components': direct_decisions}
            if source == 'whole_box':
                self.predictor.set_image(image)
                masks, scores, _ = self.predictor.predict(box=np.array([0, 0, 511, 511], np.float32),
                    multimask_output=True, return_logits=False, normalize_coords=True)
                mask, decision = select_multimask(masks, scores, (512, 512))
                return {'mask': mask, 'box': [0, 0, 511, 511], 'decision': decision, 'predict_calls': 1}
            if source == 'grid':
                generator = make_grid_generator(self.predictor, config['policy']['grid'])
                candidates = generator.generate(image)
                generator.predictor.reset_predictor()
                return {'candidates': candidates, 'predict_calls': calls['mask_decoder']}
            return segment_components(self.predictor, image, components)
        try:
            result = timed('sam', lambda: no_label_inference(sam) if automatic else sam())
        finally:
            for handle in handles:
                handle.remove()
            self.predictor.reset_predictor()
        def postprocess():
            if source == 'grid':
                selected = select_grid_masks(result.pop('candidates'), probability, valid,
                    config['policy']['grid'], config['grid_water_threshold'])
                result.update(selected)
            if source != 'assisted_historical':
                result['mask'] &= valid
            return result.pop('mask')
        mask = timed('postprocess', postprocess)
        counts = {'components': len(components),
            'positive_points': (1024 if source == 'grid' else sum(sum(c['point_labels']) for c in components)),
            'negative_points': sum(len(c['point_labels'])-sum(c['point_labels']) for c in components),
            'positive_shortfall': sum(c.get('positive_shortfall', 0) for c in components),
            'negative_shortfall': sum(c.get('negative_shortfall', 0) for c in components),
            'box_prompts': int(source == 'whole_box'),
            'sam_predict_calls': result['predict_calls'], **calls}
        arrays = {'mask': mask, 'input_valid': valid}
        if source in ('dino_points', 'grid'):
            arrays['probability'] = probability
        return arrays, {
            'id': chip['id'], 'regime': self.regime, 'role': 'val', 'event': chip['event'],
            'image_sha256': chip['image_sha256'], 'config_sha256': digest(config),
            'probe_sha256': config['probe_sha256'], 'prompts': prompts, 'prompt_sha256': digest(prompts),
            'inference': result, 'counts': counts, 'stages_ms': stages,
            'label_free_inference': automatic, 'origin': 'stage7_inference'}

    def run_pipeline(self, config, chip, cached_features=None):
        import torch
        from unittest.mock import patch
        with torch.no_grad():
            if config['source'].startswith('assisted'):
                return self.pipeline(config, chip, cached_features)
            # Tripwire encompasses the WHOLE automatic flow, not just SAM.
            with patch(__name__+'.supervised_labels', side_effect=AssertionError('Automatic labels forbidden')):
                return no_label_inference(self.pipeline, config, chip, cached_features)

    def reuse(self, row_id, config, chip, index, third):
        """Only exact existing policies/routes. Other rows must run in Colab."""
        from .automatic_prompts import generate_prompts, probability_map
        from .probe import probabilities
        if row_id == 'historical_assisted':
            item = next(r for r in third['report']['regimes'][self.regime]['prediction_index']
                        if r['id'] == chip['id'] and r['role'] == 'val')
            store = ArtifactStore(self.root/'regimes'/self.regime/'runs'/third['report']['run_id']/'assisted')
            arrays, meta = unpack(checked_read(store, chip['id'], item))
            meta['origin'] = {'stage': 3, 'artifact': item}
            return arrays, meta
        if row_id in ('a1_points', 'a1_grid'):
            selected = self.prior['regimes'][self.regime]['selection']['selected']
            policy = self.bundle['prompts']
            if row_id == 'a1_points':
                ti = policy['positive_thresholds'].index(self.bundle['positive_threshold'])
                name = f'{self.regime}_points_{selected["probe_index"]}_{ti}_{index:03}'
            else:
                gi = policy['grid']['water_thresholds'].index(self.bundle['grid_water_threshold'])
                name = f'{self.regime}_grid_{gi}_{index:03}'
            arrays, meta = unpack(self.read_prior(name))
            if (meta['id'] != chip['id'] or meta['role'] != 'val' or meta['regime'] != self.regime or
                    meta['probe_sha256'] != config['probe_sha256']):
                raise ValueError('Prediction cache lineage mismatch')
            if row_id == 'a1_points' and (meta['prompts']['policy'] != config['policy'] or
                    meta['prompts']['positive_threshold'] != config['positive_threshold']):
                raise ValueError('Prediction prompt policy changed')
            if row_id == 'a1_grid' and meta['grid']['water_threshold'] != config['grid_water_threshold']:
                raise ValueError('Prediction grid calibration changed')
            meta['origin'] = {'stage': 5, 'name': name, 'artifact': self.prior['artifacts'][name]}
            return arrays, meta
        if row_id == 'a3_tuned':
            feat = self.feature(index, chip)
            p = probability_map(probabilities(feat['features'], self.bundle['probe'], regime=self.regime))
            prompts = generate_prompts(p, feat['input_valid'], config['policy'], config['positive_threshold'])
            prompts.pop('component_map')
            if prompts != self.sixth['fixed'][chip['id']]:
                raise ValueError('A3 probe/prompt policy differs from stage-6 fixed points')
            return {'mask': self.sixth['masks'][index], 'probability': p, 'input_valid': feat['input_valid']}, {
                'prompts': prompts, 'inference': {'selection': 'stage6_highest_predicted_iou_first_tie',
                    'scores': None, 'scores_unavailable_reason': 'stage6_best_masks_did_not_persist_per_component_scores'},
                'origin': {'stage': 6, 'best_epoch': self.sixth['best_epoch']}}
        return None

    def profile(self, config, chip, features, profile_config, conditions, expected_mask):
        from .profiling import benchmark
        result = {'conditions': conditions, 'profile_ids': [chip['id']],
                  'representativeness': 'first_lexicographic_val_chip_only_not_population_latency'}
        for cache in ('cached', 'uncached'):
            observed, masks = [], []
            def forward():
                arrays, meta = self.run_pipeline(config, chip, features if cache == 'cached' else None)
                observed.append(meta)
                masks.append(arrays['mask'])
            result[cache] = benchmark(forward, profile_config)
            if any(not np.array_equal(mask, expected_mask) for mask in masks):
                raise ValueError('Live profile differs from persisted mask: '+cache)
            measured = observed[profile_config['warmup']:]
            result[cache]['stages_ms_samples'] = [m['stages_ms'] for m in measured]
            result[cache]['counts'] = [m['counts'] for m in measured]
            result[cache]['feature_cache'] = ('verified_features_host_RAM' if cache == 'cached' else 'none')
            result[cache]['sam_embedding_cache'] = 'none_recomputed_each_call'
            result[cache]['mask_reconstruction'] = 'bitwise_equal_all_warmup_and_measured_repetitions'
        return result
