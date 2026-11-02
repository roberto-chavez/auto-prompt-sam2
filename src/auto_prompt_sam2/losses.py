"""Valid-pixel chip supervision; imports are lazy for CPU-only source checks."""

def union_logits(component_logits):
    """sigmoid(max(logits)) == max(sigmoid(logits)); stable BCE, first tie."""
    import torch
    if not component_logits:
        return None
    return torch.stack(component_logits).max(dim=0).values.float()


def masked_bce_dice(logits, target, valid, *, smooth=1.0):
    import torch
    import torch.nn.functional as F
    if logits.shape != target.shape or valid.shape != target.shape or valid.dtype != torch.bool:
        raise ValueError('Matching native grids and explicit boolean validity required')
    if not valid.any():
        return None
    x, y = logits[valid].float(), target[valid].float()
    if not torch.isfinite(x).all() or not torch.isin(y, y.new_tensor([0., 1.])).all():
        raise ValueError('Finite logits and binary valid targets required')
    bce = F.binary_cross_entropy_with_logits(x, y, reduction='mean')
    p = x.sigmoid()
    dice = 1 - (2*(p*y).sum()+smooth)/(p.sum()+y.sum()+smooth)
    return {'loss': bce+dice, 'bce': bce, 'dice': dice}
