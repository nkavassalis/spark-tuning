#!/usr/bin/env python3
"""Build files/overlay/mtp.py = FP8-dense overlay (mtp.diff) + reduced-vocab drafting, FP8-aware.

Run on the head from the deployment repo root, after files/mtp_patched.py exists
(python3 files/patch_mtp_draft_vocab.py generates it from the image's mtp.py).

Upstream start.sh refuses MTP_DRAFT_VOCAB with FP8_DENSE ("both overlay nvidia/mtp.py").
Two fixes are needed to combine them:
  1. apply mtp.diff (quant_config for the drafter's mixer + lm_head) on top of the
     draft-vocab version of mtp.py instead of the image original;
  2. the draft-vocab slice did `F.linear(h.to(W.dtype), W[rows])`. With an FP8
     per-channel lm_head, W is float8_e4m3fn and its per-row weight_scale would be
     dropped (fp8 F.linear is not even supported). Dequantize the slice once at load
     time: W_bf16[rows] = W_fp8[rows].float() * weight_scale[rows]. _attach_draft_vocab
     runs inside load_weights, i.e. BEFORE process_weights_after_loading transposes the
     FP8 weight, so the checkpoint [out, in] layout still holds there.
"""
import os, subprocess, sys

OV = "files/overlay"
src = "files/mtp_patched.py"
out = os.path.join(OV, "mtp.py")
tmp = os.path.join(OV, "mtp.py.draftvocab-base")
open(tmp, "w").write(open(src).read())
subprocess.run(["patch", "-s", "-o", out, tmp, os.path.join(OV, "mtp.diff")], check=True)

s = open(out).read()
old = '''    model.register_buffer(
        "_draft_lm_head_weight",
        weight.data.index_select(0, rows).contiguous(),
        persistent=False,
    )'''
new = '''    draft_w = weight.data.index_select(0, rows)
    if draft_w.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        # [fp8dense+draftvocab] FP8 per-channel lm_head: dequantize the slice to BF16.
        scale = getattr(lm_head, "weight_scale", None)
        if scale is None:
            logger.warning("MTP draft vocab: FP8 lm_head without weight_scale; skipping.")
            return
        s = scale.data.reshape(-1).float()
        if s.numel() == weight.shape[0]:
            s = s.index_select(0, rows).unsqueeze(1)
        elif s.numel() == 1:
            s = s.reshape(1, 1)
        else:
            logger.warning("MTP draft vocab: unexpected weight_scale shape %s; skipping.",
                           tuple(scale.shape))
            return
        draft_w = (draft_w.float() * s).to(torch.bfloat16)
        logger.info("MTP draft vocab: FP8 lm_head slice dequantized to BF16 (per-channel scales)")
    model.register_buffer(
        "_draft_lm_head_weight",
        draft_w.contiguous(),
        persistent=False,
    )'''
assert s.count(old) == 1, "draft-vocab registration block not found"
s = s.replace(old, new)
# report the slice size in its real dtype
s = s.replace("    cut_gib = model._draft_lm_head_weight.numel() * esize / 2**30",
              "    cut_gib = (model._draft_lm_head_weight.numel()\n"
              "               * model._draft_lm_head_weight.element_size() / 2**30)")
open(out, "w").write(s)
compile(s, out, "exec")
print(f"wrote {out}: fp8dense markers={s.count('fp8dense overlay')}, "
      f"draft-vocab={'_attach_draft_vocab' in s}, fp8-aware={'fp8dense+draftvocab' in s}")
