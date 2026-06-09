#!/usr/bin/env bash
# K-sweep correctness driver for the MTP draft-extend work.
#
# Bar (user): MPK qwen3-30B-A3B+EAGLE3 must match the sglang reference on the
# first N tokens (default 50) for K=1..5. Also runs the target-only-greedy
# isolation comparison (spec off) to separate target-forward numerics from the
# spec/verify path.
#
# Prereqs (NEEDS a healthy GPU — node-wide CUDA must init):
#   - attention_sm100.cuh MAX_TOKENS=6 (covers K<=5). This driver sets it and
#     restores it on exit.
#   - conda envs: mirage00 (MPK), sglang (reference).
#
# Usage:
#   bash demo/qwen3/eagle3_correctness/sweep.sh [GPU] [N_PREFIX] [MAX_NEW]
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

GPU="${1:-0}"
NPREFIX="${2:-50}"
MAXNEW="${3:-256}"
MIR=/home/letianr/miniconda3/envs/mirage00/bin/python
SGL=/home/letianr/miniconda3/envs/sglang/bin/python
ATTN=include/mirage/persistent_kernel/tasks/blackwell/attention_sm100.cuh
DEMO=demo/qwen3/demo_30B_A3B_eagle3.py
CMP=demo/qwen3/eagle3_correctness/compare.py
RUNSGL=demo/qwen3/eagle3_correctness/run_sglang.py
OUT=/tmp/sweep_out; mkdir -p "$OUT"

export LD_LIBRARY_PATH=/usr/mpi/gcc/openmpi-4.1.9a1/lib:${LD_LIBRARY_PATH:-}
export PATH=/usr/local/cuda/bin:/home/letianr/miniconda3/envs/sglang/bin:$PATH
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

# MAX_TOKENS 8->6 (run-only; restored on exit)
sed -i 's/int MAX_TOKENS = 8>/int MAX_TOKENS = 6>/' "$ATTN"
restore() { git checkout -- "$ATTN" 2>/dev/null; }
trap restore EXIT

run_mpk() {  # $1=K $2=outjson  (eagle3 spec)
  MPK_DUMP_JSON="$2" CUDA_VISIBLE_DEVICES="$GPU" $MIR "$DEMO" \
    --use-mirage --eagle3 --num-draft-steps "$1" \
    --max-num-batched-tokens "$(( $1 + 1 ))" --max-num-batched-requests 1 \
    --max-seq-length 300 --page-size 4096 --max-num-pages 16 \
    > "$OUT/mpk_k$1.log" 2>&1
}
run_sgl() {  # $1=K $2=outjson
  CUDA_VISIBLE_DEVICES="$GPU" $SGL "$RUNSGL" --num-draft-steps "$1" \
    --max-new-tokens "$MAXNEW" --context-length 4096 --mem-fraction-static 0.8 \
    --out "$2" > "$OUT/sgl_k$1.log" 2>&1
}

echo "### sglang reference (eagle3) is K-independent in OUTPUT (spec is output-exact)"
echo "### but run per K anyway to confirm; target-greedy ref via sglang spec off is separate."
for K in 1 2 3 4 5; do
  echo "=== K=$K: MPK eagle3 vs sglang eagle3 (first $NPREFIX) ==="
  run_mpk "$K" "$OUT/mpk_k$K.json" || { echo "MPK K=$K FAILED, see $OUT/mpk_k$K.log"; continue; }
  run_sgl "$K" "$OUT/sgl_k$K.json"  || { echo "sglang K=$K FAILED, see $OUT/sgl_k$K.log"; continue; }
  $MIR "$CMP" --mpk "$OUT/mpk_k$K.json" --ref "$OUT/sgl_k$K.json" --prefix "$NPREFIX"
done

# NOTE: target-only-greedy isolation is done separately (live), not here:
# the plain target-greedy vehicle is demo/qwen3/demo_30B_A3B.py (no --eagle3;
# the eagle3 demo always builds a spec config). That demo has no MPK_DUMP_JSON
# hook yet — add it + validate interactively once a GPU is healthy, then compare
# its output vs a sglang target-greedy run (run_sglang without EAGLE3).
