"""Decoder-only training primitives. All gradient/model execution is Colab-only."""
from __future__ import annotations
import hashlib
import io
import random
import numpy as np
from .sam_adapter import require_colab_gpu


def configure_decoder(model):
    require_colab_gpu()
    model.eval()
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith('sam_mask_decoder.'))
        p.grad = None
    model.sam_mask_decoder.train()
    return [n for n, p in model.named_parameters() if p.requires_grad]


def fingerprints(model):
    result = {}
    for name, value in model.state_dict().items():
        v = value.detach().cpu().contiguous()
        result[name] = hashlib.sha256(v.numpy().tobytes()).hexdigest()
    return result


def audit_changes(before, after):
    if before.keys() != after.keys():
        raise ValueError('State keys changed')
    changed = [n for n in before if before[n] != after[n]]
    if any(not n.startswith('sam_mask_decoder.') for n in changed):
        raise ValueError('Frozen SAM state changed')
    return changed


def forward_components(model, image_tensor, components):
    """Direct pinned SAM API, no predictor no_grad wrapper on decoder.

    conv_s0/conv_s1 belong to the decoder: keep their graph outside no_grad.
    Native logits are bilinear 512x512; no hole filling or binary intermediates.
    """
    require_colab_gpu()
    import torch
    import torch.nn.functional as F
    from .losses import union_logits
    if not components:
        return None
    with torch.no_grad():
        backbone = model.image_encoder(image_tensor)
    if model.use_high_res_features_in_sam:
        backbone['backbone_fpn'][0] = model.sam_mask_decoder.conv_s0(backbone['backbone_fpn'][0])
        backbone['backbone_fpn'][1] = model.sam_mask_decoder.conv_s1(backbone['backbone_fpn'][1])
    _, vision, _, sizes = model._prepare_backbone_features(backbone)
    if model.directly_add_no_mem_embed:
        vision[-1] = vision[-1] + model.no_mem_embed
    feats = [v.permute(1, 2, 0).reshape(1, -1, *hw) for v, hw in zip(vision, sizes)]
    selected = []
    for c in components:
        with torch.no_grad():
            xy = torch.tensor(c['point_coords'], device=image_tensor.device, dtype=torch.float32)[None]*2
            labels = torch.tensor(c['point_labels'], device=image_tensor.device, dtype=torch.int32)[None]
            sparse, dense = model.sam_prompt_encoder(points=(xy, labels), boxes=None, masks=None)
            pe = model.sam_prompt_encoder.get_dense_pe()
        masks, scores, _, _ = model.sam_mask_decoder(image_embeddings=feats[-1], image_pe=pe,
            sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
            multimask_output=True, repeat_image=False, high_res_features=feats[:-1])
        index = scores[0].detach().argmax()
        # PyTorch 2.7 CUDA bilinear backward rejects strict determinism.
        # This small differentiable resize runs on the Colab CPU, never the PC.
        native = F.interpolate(masks[:, index:index+1].float().cpu(), (512, 512),
                               mode='bilinear', align_corners=False).to(image_tensor.device)
        selected.append(native[0, 0])
    return union_logits(selected)


def rng_state():
    import torch
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}


def restore_rng(state):
    import torch
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def serialize(value):
    import torch
    stream = io.BytesIO()
    torch.save(value, stream)
    return stream.getvalue()


def deserialize(payload):
    # Only hash-verified, locally produced checkpoints; never untrusted uploads.
    import torch
    return torch.load(io.BytesIO(payload), map_location='cpu', weights_only=False)


def assert_equal_state(a, b):
    import torch
    if isinstance(a, torch.Tensor):
        assert torch.equal(a.cpu(), b.cpu()), 'Tensor recovery differs'
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b), 'Array recovery differs'
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_equal_state(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_equal_state(x, y)
    else:
        assert a == b, 'Scalar recovery differs'
