"""Stage 6 only: user-run Colab, independent decoder training and epoch recovery."""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import zipfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from auto_prompt_sam2.artifacts import ArtifactStore, canonical_json, digest, sha256_file, safe_name, utc_now


def validate_training_config(config):
    if digest(config) != 'debfce642343ebb36041a8e8b1a648b219d80d3654ebbd1230181d39af58262f':
        raise ValueError('Stage-6 preregistered budget/policy changed; version the protocol explicitly')


def source_paths():
    from auto_prompt_sam2.runtime import source_files
    return sorted(set(source_files(REPO)) | set((REPO/'configs').glob('*.yaml')) | {
        REPO/'requirements-stage2.txt', REPO/'notebooks/06_entrenamiento_decoder.ipynb',
        REPO/'docs/project/validacion_etapa6.md', REPO/'configs/stage6_dependencies.json'})


def read_dependency(root, stage, spec):
    """Pin the prior validation report by SHA, then verify its generation inventory."""
    store = ArtifactStore(root/stage/'environment/runs'/safe_name(spec['run_id']))
    candidates = sorted((store.root/'validation').glob('*/COMPLETE.json'), reverse=True)
    for p in candidates:
        marker = json.loads(p.read_text())
        if marker['sha256'] == spec['validation_sha256']:
            payload, checked = store.read('validation', marker['identity'], marker['generation'])
            report = json.loads(payload)
            if report['run_id'] != spec['run_id'] or report.get('reserved_access'):
                raise ValueError('Prior run/scope mismatch')
            state = json.loads(store.read('state', marker['identity'])[0])
            if state['state'] != 'complete':
                raise ValueError('Prior computation incomplete')
            for name, item in report.get('artifacts', {}).items():
                _, m = store.read(name, item.get('identity', marker['identity']), item['generation'])
                if m['sha256'] != item['sha256']:
                    raise ValueError('Dependency artifact changed: '+name)
            if stage == 'stage3':
                for regime, value in report['regimes'].items():
                    baseline = ArtifactStore(root/'regimes'/regime/'runs'/report['run_id']/'assisted')
                    for item in value['prediction_index']:
                        _, m = baseline.read(item['id'], item['identity'], item['generation'])
                        if m['sha256'] != item['sha256']:
                            raise ValueError('Baseline prediction hash mismatch')
            return store, report, checked
    raise ValueError('Missing pinned dependency report: '+stage)


def verify_bundle(bundle, regime, inp, ids, models):
    lineage = bundle['probe']['lineage']
    if (bundle['regime'] != regime or lineage['regime'] != regime or lineage['role'] != 'train'
        or lineage['ids'] != ids['train'] or bundle['validation_ids'] != ids['val']
        or lineage['manifest_sha256'] != digest(inp['manifest'])
        or lineage['preprocessing_sha256'] != digest(inp['params'])
        or digest(bundle['preprocessing']) != digest(inp['params']) or bundle['models'] != models):
        raise ValueError('Regime/probe/preprocessing/model lineage mismatch')


def main(root, mode):
    from auto_prompt_sam2.sam_adapter import require_colab_gpu
    require_colab_gpu()
    import numpy as np
    import torch
    from auto_prompt_sam2.baseline import stage2_inputs
    from auto_prompt_sam2.runtime import load_config, seed_everything, inventory, verify_installed_pins
    from auto_prompt_sam2.sam_adapter import load_frozen_sam
    from auto_prompt_sam2.preprocessing import prepare
    from auto_prompt_sam2.stage5 import plan, read_image, supervised_labels, pack, unpack, no_label_inference
    from auto_prompt_sam2.probe import probabilities
    from auto_prompt_sam2.automatic_prompts import probability_map, generate_prompts
    from auto_prompt_sam2.metrics import aggregate, chip_metrics, pixel_areas_m2
    from auto_prompt_sam2.losses import masked_bce_dice
    from auto_prompt_sam2.training import (configure_decoder, fingerprints, audit_changes,
        forward_components, rng_state, restore_rng, serialize, deserialize, assert_equal_state)

    root = Path(root).resolve()
    drive = Path('/content/drive').resolve()
    if not root.is_relative_to(drive):
        raise ValueError('Persistent mounted Drive root required')
    os.environ['AUTO_PROMPT_PERSIST_ROOT'] = str(root)
    config = json.loads((REPO/'configs/training.yaml').read_text('utf-8'))
    validate_training_config(config)
    deps = json.loads((REPO/'configs/stage6_dependencies.json').read_text('utf-8'))
    env = load_config(REPO)
    verify_installed_pins(REPO, env, linux_lock=True)
    torch.set_num_threads(1)
    seed_everything(config['seed'])
    inputs = stage2_inputs(root, json.loads((REPO/'configs/data.yaml').read_text('utf-8')))
    first = deps['stage1']
    first_store = ArtifactStore(root/'environment/runs'/first['run_id'])
    for name, markers in first['markers'].items():
        for marker in markers:
            _, checked = first_store.read(name, marker['identity'], marker['generation'])
            if checked['sha256'] != marker['sha256']:
                raise ValueError('Stage-1 generation mismatch')
    dependencies = {'stage1': {'run_id': first['run_id'], 'markers_sha256': digest(first['markers'])}}

    for stage in ('stage3', 'stage4', 'stage5'):
        store, report, marker = read_dependency(root, stage, deps[stage])
        for regime, inp in inputs.items():
            lineage = report['identity']['regimes'][regime]
            if (lineage['manifest_sha256'] != digest(inp['manifest']) or
                lineage['preprocessing_sha256'] != digest(inp['params'])):
                raise ValueError('Prior stage data/preprocessing mismatch: '+stage)
        dependencies[stage] = {'run_id': report['run_id'], 'report_sha256': marker['sha256'],
                               'verified_artifacts': len(report.get('artifacts', {}))}
        if stage == 'stage5':
            prior_store, prior = store, report
    source = {p.relative_to(REPO).as_posix(): sha256_file(p) for p in source_paths()}
    ticket_identity = {'config': config, 'source': source, 'dependencies': dependencies}
    ticket_store = ArtifactStore(root/'stage6')
    if mode == 'prepare':
        try:
            ticket_store.read('prepared', ticket_identity)
        except FileNotFoundError:
            ticket_store.publish('prepared', canonical_json({'at': utc_now()}), ticket_identity)
        else:
            raise ValueError('Experiment already prepared: use resume')
    else:
        ticket_store.read('prepared', ticket_identity)  # Reject changed config/source/dependencies.
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    for regime in config['regimes']:
        seed_everything(config['seed'])  # Independent initialization, order and RNG.
        inp = inputs[regime]
        plans = {role: plan(inp['manifest'], role) for role in config['roles']}
        ids = {role: [r['id'] for r in rows] for role, rows in plans.items()}
        dep_artifacts = {}
        def prior_read(name):
            item = prior['artifacts'][name]
            if item['identity']['regime'] != regime:
                raise ValueError('Cross-regime artifact')
            payload, marker = prior_store.read(name, item['identity'], item['generation'])
            if marker['sha256'] != item['sha256']:
                raise ValueError('Stage-5 payload hash mismatch')
            dep_artifacts[name] = item
            return payload
        bundle = json.loads(prior_read(regime+'_complete_bundle'))
        verify_bundle(bundle, regime, inp, ids, env['models'])
        if (bundle['probe']['lineage']['dino_checkpoint_sha256'] != bundle['base_checkpoint_sha256']['dino']
            or bundle['base_checkpoint_sha256']['dino'] != prior['models']['dino']['checkpoint_sha256']
            or bundle['base_checkpoint_sha256']['sam'] != prior['models']['sam']['checkpoint_sha256']):
            raise ValueError('Base model / feature / probe checkpoint mismatch')
        identity = {'stage': 'phase1_stage6', 'regime': regime, 'config': config,
            'source_sha256': source, 'dependencies': dependencies, 'ids': ids,
            'manifest_sha256': digest(inp['manifest']), 'preprocessing_sha256': digest(inp['params']),
            'bundle_sha256': digest(bundle), 'base_checkpoint_sha256': bundle['base_checkpoint_sha256']}
        store = ArtifactStore(root/'regimes'/regime/'stage6'/digest(identity))
        evidence = store.root/'evidence'
        evidence.mkdir(exist_ok=True)
        def publish_json(name, value):
            payload = canonical_json(value)
            store.publish(name, payload, identity)
            (evidence/(name+'.json')).write_bytes(payload)
        publish_json('config', identity)
        publish_json('build_provenance', json.loads((REPO/'stage6_build_provenance.json').read_text('utf-8')))
        publish_json('environment', {'inventory': inventory(root, REPO), 'boot_id': boot})
        try:
            try:
                saved_payload, last_marker = store.read('last', identity)
            except FileNotFoundError:
                saved_payload = None
            if mode == 'prepare' and saved_payload is not None:
                raise ValueError('Existing last: use resume, never silently restart')
            if mode == 'resume' and saved_payload is None:
                print(regime, 'No committed epoch; initialize independently from verified base', flush=True)
            # Fix every prompt before loading labels/training. DINO is never instantiated.
            fixed = {}
            for role, rows in plans.items():
                for i, row in enumerate(rows):
                    arrays, meta = unpack(prior_read(f'{regime}_features_{role}_{i:03}'))
                    if (meta['id'] != row['id'] or meta['role'] != role or meta['regime'] != regime
                        or meta['image_sha256'] != row['image_sha256']
                        or meta['preprocessing_sha256'] != digest(inp['params'])):
                        raise ValueError('Feature lineage mismatch')
                    p = probability_map(probabilities(arrays['features'], bundle['probe'], regime=regime))
                    prompts = no_label_inference(generate_prompts, p, arrays['input_valid'],
                                                  bundle['prompts'], bundle['positive_threshold'])
                    prompts.pop('component_map')
                    fixed[row['id']] = prompts
            publish_json('fixed_prompts', fixed)
            publish_json('dependency_artifacts', dep_artifacts)
            predictor, sam_info = load_frozen_sam(env['models']['sam2'], root, offline=True,
                expected_sha256=bundle['base_checkpoint_sha256']['sam'])
            model = predictor.model
            del predictor
            before = fingerprints(model)
            trainable = configure_decoder(model)
            publish_json('parameters', {'trainable': trainable, 'base': before,
                'sam': sam_info, 'dino': 'not_loaded_verified_stage5_frozen_feature_cache',
                'dino_sha256': bundle['base_checkpoint_sha256']['dino']})
            optimizer = torch.optim.AdamW(model.sam_mask_decoder.parameters(), lr=config['lr'],
                                          weight_decay=config['weight_decay'])
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
            scaler = torch.amp.GradScaler('cuda', enabled=False)
            progress = {'epoch': 0, 'steps': 0, 'best_iou': None, 'best_epoch': None,
                'early_stopping_counter': 0, 'curves': [], 'sampler': {'cursor': 0, 'order': []},
                'gradient_checks': [], 'invalid_loss_checks': [], 'recovery': None}
            def capture():
                # Serialize immediately; state_dict tensors otherwise alias the live model.
                return serialize({'identity': identity, 'decoder': model.sam_mask_decoder.state_dict(),
                    'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                    'amp': scaler.state_dict(), 'rng': rng_state(), 'progress': progress,
                    'frozen_hashes': {n: h for n, h in before.items() if not n.startswith('sam_mask_decoder.')},
                    'boot_id': boot})
            def restore(payload):
                nonlocal progress
                state = deserialize(payload)
                if state['identity'] != identity:
                    raise ValueError('Checkpoint identity mismatch')
                assert state['frozen_hashes'] == {n: h for n, h in before.items() if not n.startswith('sam_mask_decoder.')}
                model.sam_mask_decoder.load_state_dict(state['decoder'], strict=True)
                optimizer.load_state_dict(state['optimizer'])
                scheduler.load_state_dict(state['scheduler'])
                scaler.load_state_dict(state['amp'])
                progress = state['progress']
                restore_rng(state['rng'])
                return state
            if saved_payload:
                old = restore(saved_payload)
                if old['boot_id'] == boot and progress['epoch'] == 1 and progress['recovery'] is None:
                    raise ValueError('First controlled resume requires a new Colab runtime')
                # Verify ALL restored states, including RNG, before computing anything.
                restored = deserialize(capture())
                for key in ('decoder', 'optimizer', 'scheduler', 'amp', 'rng', 'progress', 'frozen_hashes'):
                    assert_equal_state(old[key], restored[key])
                recovery = {'state_bitwise_equal': True, 'previous_boot_id': old['boot_id'],
                    'current_boot_id': boot, 'different_runtime': old['boot_id'] != boot,
                    'next_epoch': progress['epoch']+1}
                progress.setdefault('recovery_history', []).append(recovery)
                if progress['recovery'] is None:
                    progress['recovery'] = recovery
            def sample(row):
                rgb, valid = read_image(root, inp['manifest'], row['id'], row['role'])
                x = prepare(rgb, valid, inp['params'], model='sam')['pixel_values'][None].cuda()
                logits = forward_components(model, x, fixed[row['id']]['components'])
                labels = supervised_labels(root, inp['manifest'], row['id'], row['role'])
                y = torch.as_tensor(labels, device='cuda')
                loss = None if logits is None else masked_bce_dice(logits, y, y != -1)
                return logits, labels, valid, loss
            def step():
                # Caller normalizes accumulated gradients by actual contributors.
                parameters = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
                if not parameters or not all(torch.isfinite(p.grad).all() for p in parameters):
                    raise ValueError('Missing or nonfinite decoder gradient')
                if any(p.grad is not None for n, p in model.named_parameters() if not n.startswith('sam_mask_decoder.')):
                    raise ValueError('Frozen parameter has gradient')
                norm = torch.nn.utils.clip_grad_norm_(parameters, config['clip_norm'], error_if_nonfinite=True)
                if float(norm) <= 0:
                    raise ValueError('Zero decoder gradient')
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                progress['steps'] += 1
                return float(norm)
            # Controlled recovery check: same permitted train step twice from the same
            # persisted state; exact comparison, then rollback BOTH diagnostic updates.
            if saved_payload and progress['epoch'] < config['epochs']:
                anchor = capture()
                row = next((r for r in plans['train'] if fixed[r['id']]['components']), None)
                if row is None:
                    raise ValueError('No prompted train chip for recovery proof')
                outcomes = []
                for _ in range(2):
                    restore(anchor)
                    optimizer.zero_grad(set_to_none=True)
                    _, _, _, loss = sample(row)
                    if loss is None:
                        raise ValueError('Recovery train chip has no valid supervision')
                    loss['loss'].backward()
                    step()
                    outcomes.append(deserialize(capture()))
                for key in ('decoder', 'optimizer', 'scheduler', 'amp', 'rng', 'progress'):
                    assert_equal_state(outcomes[0][key], outcomes[1][key])
                restore(anchor)
                progress['recovery']['next_step_replay_bitwise_equal'] = True
                publish_json('recovery_validation', progress['recovery'])
            if saved_payload is None:
                store.publish('last', capture(), identity)
            for epoch in range(progress['epoch'], config['epochs']):
                model.eval()
                model.sam_mask_decoder.train()
                order = np.random.permutation(len(plans['train'])).tolist()
                progress['sampler'] = {'order': order, 'cursor': 0}
                optimizer.zero_grad(set_to_none=True)
                count = skipped_empty = skipped_invalid = contributing = 0
                totals = {'bce': 0., 'dice': 0., 'loss': 0.}
                metrics = []
                for cursor, index in enumerate(order):
                    row = plans['train'][index]
                    logits, labels, valid, loss = sample(row)
                    mask = np.zeros((512, 512), bool) if logits is None else (logits.detach().cpu().numpy() > 0) & valid
                    metrics.append(chip_metrics(labels, mask, chip_id=row['id'], event=row['event'],
                                                 pixel_area_m2=pixel_areas_m2(row['geometry'])))
                    if logits is None:
                        skipped_empty += 1
                    elif loss is None:
                        skipped_invalid += 1
                    else:
                        if not progress['invalid_loss_checks']:
                            y = torch.as_tensor(labels, device='cuda')
                            v = y != -1
                            alt = y.clone()
                            alt[~v] = 12345
                            a = masked_bce_dice(logits, alt, v)
                            assert torch.equal(loss['loss'], a['loss'])
                            # Also exercise an explicit invalid pixel if this chip has none.
                            v = v.clone(); v[0, 0] = False
                            alt = y.clone(); alt[0, 0] = 12345
                            assert torch.equal(masked_bce_dice(logits, y, v)['loss'], masked_bce_dice(logits, alt, v)['loss'])
                            gx = torch.autograd.grad(masked_bce_dice(logits, alt, v)['loss'], logits, retain_graph=True)[0]
                            assert torch.count_nonzero(gx[~v]) == 0
                            progress['invalid_loss_checks'].append({'id': row['id'], 'equal': True})
                        loss['loss'].backward()
                        contributing += 1
                        count += 1
                        for k in totals:
                            totals[k] += float(loss[k].detach())
                    if count and (count == config['accumulation'] or cursor == len(order)-1):
                        for p in model.sam_mask_decoder.parameters():
                            if p.grad is not None:
                                p.grad.div_(count)
                        norm = step()
                        if not progress['gradient_checks']:
                            progress['gradient_checks'].append({'id': row['id'], 'norm': norm,
                                'decoder_finite_nonzero': True, 'frozen_no_grad': True})
                        count = 0
                    progress['sampler']['cursor'] = cursor+1
                    if (cursor+1) % 20 == 0:
                        print(regime, 'epoch', epoch+1, 'train', cursor+1, '/', len(order), 'steps', progress['steps'], flush=True)
                if not contributing:
                    raise ValueError('No train gradient coverage; policy review required, no assisted fallback')
                model.eval()
                val_metrics, val_losses, val_masks = [], [], []
                with torch.no_grad():
                    for row in plans['val']:
                        logits, labels, valid, loss = sample(row)
                        mask = np.zeros((512, 512), bool) if logits is None else (logits.cpu().numpy() > 0) & valid
                        val_masks.append(mask)
                        val_metrics.append(chip_metrics(labels, mask, chip_id=row['id'], event=row['event'],
                                                        pixel_area_m2=pixel_areas_m2(row['geometry'])))
                        if loss is not None:
                            val_losses.append({k: float(v) for k, v in loss.items()})
                val = aggregate(val_metrics)
                iou = val['pooled']['values']['water_iou']
                if iou is None:
                    raise ValueError('Undefined validation water IoU')
                changed = audit_changes(before, fingerprints(model))
                if not changed:
                    raise ValueError('No decoder weight changed')
                scheduler.step()
                progress['epoch'] = epoch+1
                progress['sampler'] = {'order': order, 'cursor': len(order)}
                curve = {'epoch': epoch+1, 'steps': progress['steps'], 'train_loss': {k: v/contributing for k,v in totals.items()},
                    'train_metrics': aggregate(metrics), 'val_metrics': val,
                    'val_loss': {k: sum(v[k] for v in val_losses)/len(val_losses) for k in totals} if val_losses else None,
                    'contributing': contributing, 'no_components': skipped_empty, 'no_valid': skipped_invalid,
                    'val_no_components': sum(not fixed[r['id']]['components'] for r in plans['val']),
                    'changed_decoder_names': changed, 'frozen_unchanged': True}
                progress['curves'].append(curve)
                improved = progress['best_iou'] is None or iou > progress['best_iou']
                if improved:
                    progress['best_iou'], progress['best_epoch'] = iou, epoch+1
                # Publish per-chip evidence before the resumable epoch commit.
                masks_payload = pack({'masks': np.stack(val_masks)}, {'ids': ids['val'], 'regime': regime, 'epoch': epoch+1})
                store.publish('epoch_'+str(epoch+1)+'_val_masks', masks_payload, identity)
                (evidence/f'epoch_{epoch+1}_val_masks.npz').write_bytes(masks_payload)
                publish_json('epoch_'+str(epoch+1)+'_metrics', {'train': metrics, 'val': val_metrics})
                payload = capture()
                generation = store.publish('epoch_'+str(epoch+1), payload, identity)
                store.publish('last', payload, identity)
                if improved:
                    store.publish('best', payload, identity)
                publish_json('curves', progress['curves'])
                print(regime, epoch+1, 'val water IoU', iou, 'contributors', contributing, flush=True)
                if mode == 'prepare' and epoch+1 == config['controlled_interrupt_after_epoch']:
                    publish_json('status', {'state': 'controlled_interruption', 'epoch': epoch+1,
                        'last_generation': generation, 'boot_id': boot, 'closure': 'pending_user_resume_and_review'})
                    break
            if progress['epoch'] == config['epochs']:
                publish_json('status', {'state': 'computation_complete_pending_external_review', 'epoch': progress['epoch'], 'boot_id': boot})
                # Repair best pointer if interrupted between last and best publication.
                best_payload = store.read('epoch_'+str(progress['best_epoch']), identity)[0]
                store.publish('best', best_payload, identity)
                best = deserialize(best_payload)
                store.publish('decoder_export', serialize(best['decoder']), identity)
                publish_json('curves', progress['curves'])
                publish_json('validation', {'status': 'computation_complete_pending_external_review',
                    'regime': regime, 'identity': identity, 'progress': progress,
                    'baseline_stage5_val_iou': prior['regimes'][regime]['selection']['selected']['metrics']['pooled']['values']['water_iou'],
                    'reserved_access': [], 'final_evaluation': False, 'stop_reason': 'fixed_epoch_budget',
                    'dino': 'unchanged_not_instantiated_features_reused',
                    'ai_use': 'Codex implementation; user Colab execution; external review pending'})
            # Inventory includes all checkpoint versions; large weights remain on Drive.
            manifest = []
            for marker_path in sorted(store.root.glob('*/*/COMPLETE.json')):
                marker = json.loads(marker_path.read_text())
                _, checked = store.read(marker_path.parent.parent.name, identity, marker['generation'])
                manifest.append({'path': str(marker_path.parent.relative_to(root)), **checked})
            (evidence/'inventory.json').write_bytes(canonical_json(manifest))
            target = store.root/'evidence.zip'
            with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
                for p in evidence.iterdir():
                    archive.write(p, p.name)
                for p in source_paths():
                    archive.write(p, 'source/'+p.relative_to(REPO).as_posix())
            print('Evidence:', target, 'SHA256:', sha256_file(target), flush=True)
            del model, optimizer, scheduler, scaler
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as exc:
            publish_json('failure', {'at': utc_now(), 'type': type(exc).__name__, 'error': str(exc),
                'mode': mode, 'regime': regime, 'resume': 'stop old writer then resume same config; replay uncommitted epoch'})
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'resume'])
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    main(args.root, args.mode)
