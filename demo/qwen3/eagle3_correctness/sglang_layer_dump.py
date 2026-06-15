"""Dump sglang's per-layer PREFILL hidden for Qwen3-30B-A3B (target), to compare
against MPK's per-layer capture (/tmp/mpk_layers.pt).

Wraps Qwen3MoeDecoderLayer.forward to record, at layer ENTRY, the full
residual-stream hidden = hidden_states + (residual or 0). This matches MPK's
capture point `x` (the layer-input hidden after the previous layer's residual
add, demo_30B_A3B_eagle3.py). We capture only the PREFILL forward (the first
forward whose token count == prompt_len).

Run in the `sglang` conda env:
    conda run -n sglang python \
      demo/qwen3/eagle3_correctness/sglang_layer_dump.py --out /tmp/sglang_layers.pt
"""
import argparse
import json

import torch
import sglang as sgl
from transformers import AutoTokenizer

PROMPT = "Give me a short introduction to large language model."
SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
TARGET = "Qwen/Qwen3-30B-A3B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="/tmp/sglang_layers.pt")
    ap.add_argument("--target", type=str, default=TARGET)
    ap.add_argument("--context-length", type=int, default=4096)
    ap.add_argument("--mem-fraction-static", type=float, default=0.8)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.target)
    text = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": PROMPT}],
        tokenize=False, add_generation_prompt=True)
    prompt_len = len(tok(text, add_special_tokens=False)["input_ids"])

    from sglang.srt.models import qwen3_moe
    LayerCls = qwen3_moe.Qwen3MoeDecoderLayer
    orig_forward = LayerCls.forward

    # captured[layer_idx] = hidden at layer entry for the PREFILL forward only.
    # Within one forward, layers run in order 0,1,2,... so first-call order ==
    # layer index. We reset the per-forward counter whenever we see layer-entry
    # for a fresh forward of prompt_len tokens.
    captured = {}
    state = {"counter": 0, "done": False}

    def patched_forward(self, positions, hidden_states, forward_batch,
                        residual=None, *a, **kw):
        is_prefill = (hidden_states is not None
                      and hidden_states.shape[0] == prompt_len)
        # Capture every layer of the FIRST prefill forward (layers run in order
        # 0,1,2,...; counter resets are unneeded since we only do one forward).
        if is_prefill and not state["done"]:
            li = state["counter"]
            state["counter"] += 1
            h = hidden_states if residual is None else hidden_states + residual
            captured[li] = h.detach().float().cpu().clone()
        return orig_forward(self, positions, hidden_states, forward_batch,
                            residual, *a, **kw)

    LayerCls.forward = patched_forward

    llm = sgl.Engine(
        model_path=args.target, dtype="bfloat16", disable_cuda_graph=True,
        context_length=args.context_length,
        mem_fraction_static=args.mem_fraction_static)

    out = llm.generate(
        text,
        sampling_params={"temperature": 0.0, "max_new_tokens": 1},
        return_logprob=False)
    llm.shutdown()

    if not captured:
        print("[sglang-layer-dump] WARNING: no layers captured "
              f"(prompt_len={prompt_len}). Check the prefill token count.")
        return
    n = max(captured) + 1
    stack = torch.stack([captured[i] for i in range(n)], dim=0)
    torch.save({"layers": stack, "prompt_len": prompt_len}, args.out)
    print(f"[sglang-layer-dump] wrote {args.out} shape={tuple(stack.shape)} "
          f"(layers, prompt_len, hidden), prompt_len={prompt_len}")


if __name__ == "__main__":
    main()
