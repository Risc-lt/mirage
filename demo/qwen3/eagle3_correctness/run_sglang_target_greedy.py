"""Plain sglang TARGET greedy decode (NO speculative decoding) on the same
templated prompt the MPK demo uses, dumping output token ids + per-token
logprobs. This is the target-side oracle for the per-position argmax compare
against MPK's greedy run (/tmp/mpk_greedy.json).

Run in the `sglang` conda env:
    conda run -n sglang python \
      demo/qwen3/eagle3_correctness/run_sglang_target_greedy.py \
      --max-new-tokens 256 --out /tmp/sglang_greedy.json
"""
import argparse
import json

import sglang as sgl
from transformers import AutoTokenizer

# Must match demo/qwen3/demo_30B_A3B_eagle3.py (system + user message).
PROMPT = "Give me a short introduction to large language model."
SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
TARGET = "Qwen/Qwen3-30B-A3B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--out", type=str, default="/tmp/sglang_greedy.json")
    ap.add_argument("--target", type=str, default=TARGET)
    ap.add_argument("--context-length", type=int, default=4096)
    ap.add_argument("--mem-fraction-static", type=float, default=0.8)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.target)
    text = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": PROMPT}],
        tokenize=False,
        add_generation_prompt=True,
    )

    # Plain target, NO speculative decoding.
    llm = sgl.Engine(
        model_path=args.target,
        dtype="bfloat16",
        disable_cuda_graph=True,
        context_length=args.context_length,
        mem_fraction_static=args.mem_fraction_static,
    )
    out = llm.generate(
        text,
        sampling_params={"temperature": 0.0, "max_new_tokens": args.max_new_tokens},
        return_logprob=True,
    )
    llm.shutdown()

    meta = out.get("meta_info", {})
    otl = meta.get("output_token_logprobs")  # [(logprob, tok_id, tok_text), ...]
    if otl:
        out_ids = [e[1] for e in otl]
        out_logprobs = [e[0] for e in otl]
    else:
        out_ids = tok(out["text"], add_special_tokens=False)["input_ids"]
        out_logprobs = None

    result = {
        "framework": "sglang", "eagle3": False,
        "target": args.target,
        "prompt": PROMPT, "templated_prompt": text,
        "output_text": out["text"],
        "output_token_ids": out_ids,
        "output_token_logprobs": out_logprobs,
        "completion_tokens": meta.get("completion_tokens"),
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[sglang-greedy] wrote {args.out} ({len(out_ids)} tokens)")
    print(f"[sglang-greedy] first 20 ids: {out_ids[:20]}")
    print(f"[sglang-greedy] text[:160]: {out['text'][:160]!r}")


if __name__ == "__main__":
    main()
