"""Compare MPK vs sglang qwen3-30b-a3b + EAGLE3 outputs.

The correctness oracle for the MPK draft-extend work. Speculative decoding is
output-exact, so under greedy the two frameworks' accepted token streams MUST
match the target model's greedy decode — and therefore each other. Accept length
is a *performance* metric (how close MPK's accept rate is to sglang's), reported
but not used as the correctness gate.

Both inputs are JSON files of the shape produced by run_sglang.py (and by the
MPK-side capture wrapper added alongside the demo):
    {"output_token_ids": [...], "output_text": "...",
     "accept_length": <float|null>, "completion_tokens": <int|null>}

Usage:
    python demo/qwen3/eagle3_correctness/compare.py \
        --mpk /tmp/mpk_eagle3.json --ref /tmp/sglang_eagle3.json
Exit code 0 = token-exact match, 1 = mismatch (first divergence printed).
"""
import argparse
import json
import sys


def load(path):
    with open(path) as f:
        return json.load(f)


def first_divergence(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return n
    return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mpk", required=True, help="MPK output JSON")
    ap.add_argument("--ref", required=True, help="reference (sglang) output JSON")
    args = ap.parse_args()

    mpk = load(args.mpk)
    ref = load(args.ref)
    mpk_ids = mpk.get("output_token_ids") or []
    ref_ids = ref.get("output_token_ids") or []

    print(f"[compare] MPK tokens={len(mpk_ids)}  ref tokens={len(ref_ids)}")
    print(f"[compare] MPK accept_length={mpk.get('accept_length')}  "
          f"ref accept_length={ref.get('accept_length')}")

    div = first_divergence(mpk_ids, ref_ids)
    if div == -1:
        print(f"[compare] TOKEN-EXACT MATCH ({len(mpk_ids)} tokens). "
              f"Correctness: PASS.")
        return 0

    lo = max(0, div - 3)
    print(f"[compare] FIRST DIVERGENCE at index {div}:")
    print(f"[compare]   MPK[{lo}:{div + 1}] = {mpk_ids[lo:div + 1]}")
    print(f"[compare]   ref[{lo}:{div + 1}] = {ref_ids[lo:div + 1]}")
    print(f"[compare]   MPK text[:160] = {mpk.get('output_text', '')[:160]!r}")
    print(f"[compare]   ref text[:160] = {ref.get('output_text', '')[:160]!r}")
    print("[compare] Correctness: FAIL (token streams diverge).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
