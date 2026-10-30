#!/usr/bin/env python
"""Parameter counts of SAM 2.1 hiera-tiny sub-blocks from the instantiated architecture.

Builds the model with RANDOM weights on CPU (counts do not depend on weights; no checkpoint is
read, nothing is downloaded) and prints/writes a JSON. It only refines the per-block parameter
table measured in stage 4: sums are asserted against the measured totals. It measures no FLOPs,
latency or quality.
"""
import argparse
import json
from pathlib import Path

MEASURED = {"total": 38962498, "image_encoder": 27219136, "sam_prompt_encoder": 6220,
            "sam_mask_decoder": 4215109, "memory_attention": 5922304, "memory_encoder": 1384608}


def count(module):
    return sum(p.numel() for p in module.parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/project/evidence/e2/teardown_architecture_counts.json")
    args = ap.parse_args()
    import torch
    from sam2.build_sam import build_sam2
    torch.manual_seed(0)
    m = build_sam2("configs/sam2.1/sam2.1_hiera_t.yaml", None, device="cpu", mode="eval",
                   apply_postprocessing=False)
    ie, tr, md = m.image_encoder, m.image_encoder.trunk, m.sam_mask_decoder
    stages, start = [], 0
    for s, end in enumerate(tr.stage_ends):
        blocks = range(start, end + 1)
        stages.append({"stage": s, "blocks": [start, end], "n_blocks": len(blocks),
                       "dim_out": int(tr.blocks[end].dim_out),
                       "params": sum(count(tr.blocks[i]) for i in blocks),
                       "blocks_with_query_pooling": [i for i in blocks if tr.blocks[i].q_stride is not None],
                       "global_attention_blocks": [i for i in blocks if tr.blocks[i].window_size == 0]})
        start = end + 1
    out = {
        "note": "Random-weight instantiation of configs/sam2.1/sam2.1_hiera_t.yaml; counts only.",
        "total": count(m),
        "image_encoder": {"total": count(ie), "trunk": count(tr), "neck_fpn": count(ie.neck),
                          "patch_embed": count(tr.patch_embed), "pos_embed": int(tr.pos_embed.numel() + tr.pos_embed_window.numel()),
                          "stages": stages, "neck_convs_in_out": [[c.conv.in_channels, c.conv.out_channels] for c in ie.neck.convs]},
        "sam_prompt_encoder": count(m.sam_prompt_encoder),
        "sam_mask_decoder": {"total": count(md), **{k: count(v) for k, v in md.named_children()},
                             "two_way_transformer_depth": int(md.transformer.depth),
                             "two_way_transformer_heads": int(md.transformer.num_heads),
                             "mask_tokens": int(md.mask_tokens.weight.shape[0])},
        "memory_attention": {"total": count(m.memory_attention), "layers": len(m.memory_attention.layers)},
        "memory_encoder": count(m.memory_encoder),
    }
    assert out["total"] == MEASURED["total"], out["total"]
    assert out["image_encoder"]["total"] == MEASURED["image_encoder"]
    assert out["sam_prompt_encoder"] == MEASURED["sam_prompt_encoder"]
    assert out["sam_mask_decoder"]["total"] == MEASURED["sam_mask_decoder"]
    assert out["memory_attention"]["total"] == MEASURED["memory_attention"]
    assert out["memory_encoder"] == MEASURED["memory_encoder"]
    assert out["image_encoder"]["trunk"] + out["image_encoder"]["neck_fpn"] == out["image_encoder"]["total"]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
