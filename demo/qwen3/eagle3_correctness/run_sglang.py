"""sglang reference generation for qwen3-30b-a3b + EAGLE3.

Produces the correctness oracle for the MPK draft-extend work: greedy output of
Qwen3-30B-A3B with the SGLang EAGLE3 draft, single linear chain (topk=1), on the
same prompt the MPK demo uses. Writes token ids + decoded text + accept stats to
JSON so compare.py can diff it against the MPK run.

Run in the `sglang` conda env:
    conda run -n sglang python demo/qwen3/eagle3_correctness/run_sglang.py \
        --num-draft-steps 4 --max-new-tokens 256 --out /tmp/sglang_eagle3.json
"""
import argparse
import json

import sglang as sgl
from transformers import AutoTokenizer

# Must match demo/qwen3/demo_30B_A3B_eagle3.py
PROMPT = "Give me a short introduction to large language model."
TARGET = "Qwen/Qwen3-30B-A3B"
DRAFT = "lmsys/SGLang-EAGLE3-Qwen3-30B-A3B-Instruct-2507-SpecForge-Nex"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-draft-steps", type=int, default=4, help="K")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--out", type=str, default="/tmp/sglang_eagle3.json")
    ap.add_argument("--target", type=str, default=TARGET)
    ap.add_argument("--draft", type=str, default=DRAFT)
    ap.add_argument("--context-length", type=int, default=4096,
                    help="Bound the KV pool; prompt+max_new_tokens is small")
    ap.add_argument("--mem-fraction-static", type=float, default=0.8)
    args = ap.parse_args()

    k = args.num_draft_steps
    tok = AutoTokenizer.from_pretrained(args.target)
    # Same chat-template construction as the MPK demo.
    text = tok.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=False,
        add_generation_prompt=True,
    )

    # Single linear chain (no tree) to match the MPK design: eagle_topk=1.
    llm = sgl.Engine(
        model_path=args.target,
        speculative_algorithm="EAGLE3",
        speculative_draft_model_path=args.draft,
        speculative_num_steps=k,
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=k + 1,
        dtype="bfloat16",
        disable_cuda_graph=True,
        context_length=args.context_length,
        mem_fraction_static=args.mem_fraction_static,
    )
    out = llm.generate(
        text,
        sampling_params={"temperature": 0.0, "max_new_tokens": args.max_new_tokens},
    )
    llm.shutdown()

    gen_text = out["text"]
    out_ids = tok(gen_text, add_special_tokens=False)["input_ids"]
    meta = out.get("meta_info", {})
    result = {
        "framework": "sglang",
        "target": args.target,
        "draft": args.draft,
        "k": k,
        "prompt": PROMPT,
        "templated_prompt": text,
        "output_text": gen_text,
        "output_token_ids": out_ids,
        "completion_tokens": meta.get("completion_tokens"),
        "spec_verify_ct": meta.get("spec_verify_ct"),
        "accept_length": (
            meta.get("completion_tokens") / meta["spec_verify_ct"]
            if meta.get("spec_verify_ct") else None
        ),
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[run_sglang] wrote {args.out}")
    print(f"[run_sglang] completion_tokens={result['completion_tokens']} "
          f"accept_length={result['accept_length']}")
    print(f"[run_sglang] output_text[:200]={gen_text[:200]!r}")


if __name__ == "__main__":
    main()
