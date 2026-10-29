"""Nonoverlapping ownership, PyTorch FLOP audit and synchronized Colab timings."""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import statistics
import time
import math

from .features import hook_inventory
from .sam_adapter import require_colab_gpu


def validate_config(config):
    if (config["stage"] != "phase1_stage4" or config["regimes"] != ["official", "leave_paraguay_out"]
            or config["roles"] != ["train"] or config["selection"] != "lexicographic_id_first_n_train"
            or config["batch_size"] != 1 or config["precision"] != "float32"
            or config["autocast"] is not False or config["sam_size"] != 1024
            or config["dino_size"] != 512 or config["patch_size"] != 16
            or config["projection"] != "none_fixed_channels"
            or config["fixed_prompts"] != {"point_coords": [[256, 256], [32, 32]],
                "point_labels": [1, 0], "multimask_output": True}
            or config["dino_attention"] != "sdpa" or config["reserved_sets"] != "forbidden"):
        raise ValueError("Stage-4 scope, model geometry, precision or prompt contract changed")
    if (config["seed"] != 42 or config["dino_feature_layers"] != [0, 11, 23] or
            config["feature_channels"] != [0, 1, 2] or
            config["repeat_tolerance"] != {"atol": 1e-5, "rtol": 1e-4} or
            config["timing_scope"] != "preloaded_inputs_and_weights_no_hooks_no_io_no_backward" or
            config["erf"] != {"target": "center_spatial_location_channel_0",
                "method": "absolute_input_gradient_sum_rgb", "normalization": "unit_mass",
                "input": "normalized_model_tensor", "sam_output": "vision_features",
                "dino_output": "last_hidden_state_patch_grid"}):
        raise ValueError("Fixed feature/ERF/recovery protocol changed")
    for key in ("warmup", "repetitions", "chips_per_regime"):
        if type(config[key]) is not int or config[key] < (2 if key == "repetitions" else 1):
            raise ValueError("Positive fixed warmup, repetitions and train selection required")


def block_names(model, kind):
    if kind == "sam":
        return [n for n in ("image_encoder", "sam_prompt_encoder", "sam_mask_decoder",
            "memory_attention", "memory_encoder", "obj_ptr_proj", "obj_ptr_tpos_proj")
            if n in dict(model.named_modules())]
    if kind == "dino":
        return ["embeddings", "rope_embeddings", *["layer." + str(i) for i in range(len(model.layer))], "norm"]
    raise ValueError("Unknown model")


def parameter_table(model, blocks):
    """Assign each unique parameter once. Shared objects use first canonical name."""
    if len(set(blocks)) != len(blocks) or any(a != b and b.startswith(a + ".") for a in blocks for b in blocks):
        raise ValueError("Parameter ownership blocks must be disjoint")
    rows = {n: {"block": n, "parameters": 0, "trainable": 0, "parameter_names": []}
            for n in [*blocks, "residual"]}
    seen = set()
    for name, parameter in model.named_parameters():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        matches = [b for b in blocks if name == b or name.startswith(b + ".")]
        key = matches[0] if matches else "residual"
        rows[key]["parameters"] += parameter.numel()
        rows[key]["trainable"] += parameter.numel() if parameter.requires_grad else 0
        rows[key]["parameter_names"].append(name)
    total = sum(p.numel() for p in model.parameters())
    if sum(r["parameters"] for r in rows.values()) != total:
        raise ValueError("Parameter block sum differs from unique model total")
    return {"rows": list(rows.values()), "total": total,
            "trainable": sum(r["trainable"] for r in rows.values()), "sum_verified": True,
            "convention": "unique_parameter_objects_exclusive_blocks_residual_explicit"}


def shape_signature(value):
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {k: shape_signature(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [shape_signature(v) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return {"type": type(value).__name__}


@contextmanager
def block_instrumentation(model, blocks, *, inputs=False):
    """Track active ownership and first call inputs; always clean up every hook."""
    import torch
    from torch.utils._pytree import tree_map
    before, handles = hook_inventory(model), []
    state = {"stack": [], "calls": Counter(), "inputs": {}, "shapes": {}}
    modules = dict(model.named_modules())
    try:
        for name in blocks:
            def pre(module, args, kwargs, key=name):
                state["stack"].append(key)
                state["calls"][key] += 1
                state["shapes"].setdefault(key, {})["inputs"] = shape_signature((args, kwargs))
                if inputs and key not in state["inputs"]:
                    state["inputs"][key] = tree_map(
                        lambda x: x.detach().cpu().clone() if isinstance(x, torch.Tensor) else x,
                        (args, kwargs))
            def post(module, args, output, key=name):
                state["shapes"].setdefault(key, {})["outputs"] = shape_signature(output)
                if not state["stack"] or state["stack"].pop() != key:
                    raise RuntimeError("Unbalanced block hooks")
            handles.append(modules[name].register_forward_pre_hook(pre, with_kwargs=True))
            handles.append(modules[name].register_forward_hook(post, always_call=True))
        yield state
    finally:
        for handle in handles:
            handle.remove()
        if hook_inventory(model) != before:
            raise RuntimeError("Profiling hooks leaked")


def count_flops(model, blocks, forward):
    require_colab_gpu()
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    with block_instrumentation(model, blocks) as active:
        class AuditCounter(FlopCounterMode):
            def __init__(self):
                super().__init__(display=False)
                self.by_block, self.ops = Counter(), Counter()
            def _count_flops(self, func_packet, out, args, kwargs):
                old = self.get_total_flops()
                result = super()._count_flops(func_packet, out, args, kwargs)
                key = active["stack"][-1] if active["stack"] else "residual"
                self.by_block[key] += self.get_total_flops() - old
                self.ops[(key, str(func_packet), func_packet in self.flop_registry)] += 1
                return result
        with torch.no_grad(), AuditCounter() as counter:
            forward()
    total = counter.get_total_flops()
    if total <= 0 or sum(counter.by_block.values()) != total:
        raise ValueError("FLOP counter empty or exclusive block sum mismatch")
    rows = [{"block": name, "executed_calls": active["calls"][name],
             "counted_flops": counter.by_block[name] if active["calls"][name] else None}
            for name in blocks]
    rows.append({"block": "residual", "executed_calls": None,
                 "counted_flops": counter.by_block["residual"]})
    return {"rows": rows, "counted_total": total, "sum_verified": True,
        "convention": "PyTorch_2.7.1_2_FLOPs_per_multiply_add_forward_only_partial",
        "operations": [{"block": b, "operation": op, "formula_registered": covered, "calls": n}
            for (b, op, covered), n in sorted(counter.ops.items())],
        "coverage": "Unregistered dispatched ops after decomposition are listed, not assigned zero true cost. Bias, normalization, activations, softmax, RoPE, interpolation, memory traffic and Python can be absent or partial. Counts are not exact total work.",
        "not_executed": "null FLOPs means present but not called in static path, never a zero-cost model"}


def timing_summary(samples, warmup):
    if warmup < 1 or len(samples) < 2 or any(not math.isfinite(s) or s <= 0 for s in samples):
        raise ValueError("Positive samples and warmup required")
    return {"median_ms": statistics.median(samples), "min_ms": min(samples),
            "max_ms": max(samples), "samples_ms": samples,
            "warmup": warmup, "repetitions": len(samples)}


def benchmark(forward, config):
    require_colab_gpu()
    import torch
    with torch.no_grad():
        for _ in range(config["warmup"]):
            forward()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline_allocated = torch.cuda.memory_allocated()
        baseline_reserved = torch.cuda.memory_reserved()
        samples = []
        for _ in range(config["repetitions"]):
            torch.cuda.synchronize()
            start = time.perf_counter()
            forward()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1000)
    return dict(timing_summary(samples, config["warmup"]),
        baseline_allocated_bytes=baseline_allocated, baseline_reserved_bytes=baseline_reserved,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        incremental_peak_allocated_bytes=torch.cuda.max_memory_allocated() - baseline_allocated,
        scope=config["timing_scope"], batch_size=1, precision="float32", autocast=False,
        synchronization="cuda_synchronize_before_and_after_each_host_wall_clock_sample",
        cache="warm_allocator_and_model_caches_no_empty_cache_between_repetitions")


def benchmark_blocks(model, blocks, forward, config):
    require_colab_gpu()
    import torch
    from torch.utils._pytree import tree_map
    with torch.no_grad(), block_instrumentation(model, blocks, inputs=True) as captured:
        forward()
    modules, results = dict(model.named_modules()), {}
    for name in blocks:
        if name not in captured["inputs"]:
            results[name] = {"status": "not_executed_static_image_path", "latency": None, "memory": None}
            continue
        args, kwargs = tree_map(lambda x: x.cuda() if isinstance(x, torch.Tensor) else x,
                                captured["inputs"].pop(name))
        results[name] = benchmark(lambda: modules[name](*args, **kwargs), config)
        results[name]["scope"] = "isolated_module_replay_preloaded_first_call_inputs_no_transfers_no_hooks_no_io"
        results[name]["calls_in_full_path"] = captured["calls"][name]
        results[name]["shapes"] = captured["shapes"][name]
        del args, kwargs
    return results
