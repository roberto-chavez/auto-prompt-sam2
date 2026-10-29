"""Frozen DINOv3 and explicit spatial features. Real execution is Colab-only."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .artifacts import sha256_file
from .sam_adapter import require_colab_gpu, assert_frozen, state_fingerprint


def load_frozen_dino(config, root, *, offline=False, expected_sha256=None):
    require_colab_gpu()  # Before any download, construction or model import.
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModel

    files = {}
    for filename in ("config.json", config["filename"]):
        files[filename] = Path(hf_hub_download(repo_id=config["repo_id"],
            revision=config["revision"], filename=filename,
            cache_dir=str(Path(root) / "base_models/huggingface"), local_files_only=offline))
    weight = files[config["filename"]]
    sha = sha256_file(weight)
    if expected_sha256 is not None and sha != expected_sha256:
        raise ValueError("DINO checkpoint hash changed")
    model = AutoModel.from_pretrained(str(weight.parent), local_files_only=True,
        trust_remote_code=False, torch_dtype=torch.float32, attn_implementation="sdpa")
    c = model.config
    if (c.model_type != "dinov3_vit" or c.patch_size != 16 or
            c.hidden_size != 1024 or c.num_hidden_layers != 24):
        raise ValueError("Expected authorized DINOv3 ViT-L/16 architecture")
    model.cuda().float().eval()
    model.requires_grad_(False)
    assert_frozen(model)
    return model, {"model": config, "checkpoint_sha256": sha,
        "config_sha256": sha256_file(files["config.json"]),
        "state_sha256": state_fingerprint(model), "precision": "float32",
        "autocast": False, "attention_implementation": "sdpa",
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": 0, "num_register_tokens": c.num_register_tokens,
        "config": c.to_dict()}


def token_layout(input_hw, patch_size, registers, hidden_size):
    h, w = input_hw
    if (h <= 0 or w <= 0 or patch_size <= 0 or h % patch_size or w % patch_size
            or registers < 0 or hidden_size <= 0):
        raise ValueError("Exact positive patch grid required; no silent crop/padding")
    gh, gw = h // patch_size, w // patch_size
    return {"cls_index": 0, "register_slice": [1, 1 + registers],
        "patch_slice": [1 + registers, 1 + registers + gh * gw],
        "sequence_length": 1 + registers + gh * gw,
        "grid_hw": [gh, gw], "hidden_size": hidden_size,
        "reshape": "B,N,C -> remove CLS/registers -> B,Hpatch,Wpatch,C -> B,C,Hpatch,Wpatch",
        "order": "row_major", "patch_size": patch_size,
        "input_patch_center_xy": "((column+0.5)*patch_size-0.5, (row+0.5)*patch_size-0.5)",
        "native_mapping": "Use persisted Coordinates.from_patch; identity when input and native sizes are 512"}


def tokens_to_grid(tokens, layout):
    if (tokens.ndim != 3 or tokens.shape[1] != layout["sequence_length"] or
            tokens.shape[2] != layout["hidden_size"]):
        raise ValueError("Token shape does not match documented CLS/register/patch layout")
    a, b = layout["patch_slice"]
    h, w = layout["grid_hw"]
    patches = tokens[:, a:b, :].reshape(tokens.shape[0], h, w, tokens.shape[2])
    if isinstance(patches, np.ndarray):
        return patches.transpose(0, 3, 1, 2)
    return patches.permute(0, 3, 1, 2)


def hook_inventory(model):
    """Include pre/forward/backward hooks so leaked instrumentation fails closed."""
    return {name: {kind: tuple(getattr(module, kind, {}).keys()) for kind in
        ("_forward_pre_hooks", "_forward_hooks", "_backward_pre_hooks", "_backward_hooks")}
        for name, module in model.named_modules()}


@contextmanager
def capture_features(model, names):
    """Detach only requested outputs to CPU; handles removed also on exceptions."""
    modules = dict(model.named_modules())
    before, handles, captured = hook_inventory(model), [], {}
    try:
        for name in names:
            if name not in modules:
                raise ValueError("Unknown feature layer: " + name)
            def hook(module, args, output, key=name):
                if not hasattr(output, "detach"):
                    raise ValueError("Feature layer must return a tensor: " + key)
                captured[key] = output.detach().float().cpu().numpy().copy()
            handles.append(modules[name].register_forward_hook(hook))
        yield captured
    finally:
        for handle in handles:
            handle.remove()
        if hook_inventory(model) != before:
            raise RuntimeError("Feature hooks leaked")


def normalize_erf(gradient):
    value = np.abs(np.asarray(gradient, dtype=np.float64)).sum(axis=0)
    if value.ndim != 2 or not np.isfinite(value).all() or value.sum() <= 0:
        raise ValueError("ERF gradient is zero, nonfinite or has invalid shape")
    return (value / value.sum()).astype(np.float32)


def effective_receptive_field(forward_grid, tensor):
    require_colab_gpu()
    import torch
    # Frozen parameters, differentiable input only. No optimizer, no backward().
    with torch.enable_grad():
        x = tensor.detach().clone().requires_grad_(True)
        grid = forward_grid(x)
        if grid.ndim != 4 or grid.shape[0] != 1:
            raise ValueError("ERF requires a batch-one BCHW spatial feature")
        y, z = grid.shape[-2] // 2, grid.shape[-1] // 2
        scalar = grid[0, 0, y, z]
        gradient, = torch.autograd.grad(scalar, x, only_inputs=True)
        result = normalize_erf(gradient[0].detach().float().cpu().numpy())
        metadata = {"output_shape": list(grid.shape), "target_bcyx": [0, 0, y, z],
                    "scalar": float(scalar.detach()), "mass": float(result.sum())}
        h, w = result.shape
        metadata["mass_outside_central_half_input"] = float(1 - result[h//4:3*h//4, w//4:3*w//4].sum())
    return result, metadata
