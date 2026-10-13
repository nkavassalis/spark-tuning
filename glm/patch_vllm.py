#!/usr/bin/env python3
"""Patch vLLM 0.31.0 (inside the stock vllm/vllm-openai image) so GLM-5.3-Flash can use the standalone
NVFP4 MTP draft in <checkpoint>/mtp-nvfp4. Idempotent; run at every container start (glm/up.sh).

Both patches come from autotrust's E224 2x DGX Spark runbook (DEPLOY-2X-DGX-SPARK.md, Appendix B) and
are inert unless --speculative-config is passed:
  1. eagle/utils.py load_eagle_model: build the draft with its OWN model_config and quant_config=None
     (re-derived from the draft's hf_quant_config.json). Otherwise the 288-expert draft is built with
     the pruned target's config (192/224 experts) and fails with a shape mismatch.
  2. glm5next/common/mtp.py: SharedHead(defer_lm_head=False). The draft ships shared_head.head.weight;
     a deferred head raises KeyError on load.
"""
import pathlib
import vllm

SP = pathlib.Path(vllm.__file__).parent


def patch(rel, marker, anchor, new):
    f = SP / rel
    t = f.read_text()
    if marker in t:
        print(f"patch_vllm: {rel}: already applied")
        return
    assert t.count(anchor) == 1, f"patch_vllm: anchor not found in {rel} (vLLM {vllm.__version__})"
    f.write_text(t.replace(anchor, new))
    print(f"patch_vllm: {rel}: applied")


A1 = "    vllm_config = speculative_config.apply_draft_overrides(vllm_config)\n"
patch("v1/worker/gpu/spec_decode/eagle/utils.py", "spark-tuning: draft geometry", A1, A1 + """\
    # spark-tuning: draft geometry and quantization come from the draft checkpoint, not the target.
    if draft_model_config is not None:
        vllm_config = replace(vllm_config, model_config=draft_model_config, quant_config=None)
""")

A2 = """        self.shared_head = SharedHead(
            config=config,
            prefix=prefix,
            defer_lm_head=True,
        )"""
patch("models/glm5next/common/mtp.py", "spark-tuning: draft ships its own head", A2, """\
        self.shared_head = SharedHead(
            config=config,
            prefix=prefix,
            defer_lm_head=False,  # spark-tuning: draft ships its own head (shared_head.head.weight)
        )""")
