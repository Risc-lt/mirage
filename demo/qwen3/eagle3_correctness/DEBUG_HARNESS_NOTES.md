# EAGLE3 K>=2 accept-collapse — debug harness + root cause (DEBUG-ONLY, REVERT before merge)

> This whole commit is a **debug scaffold**, not production code. After EAGLE3
> draft-extend development is done, **revert the debug commit** that introduced
> this file (`git revert <hash>` or drop it from the branch). The only *real*
> fix is in a separate prior commit (the prefill-aware draft EXTEND span in
> `persistent_kernel.cuh`) — keep that one.

## Root cause (FACT-established, cos 0.99 — not speculation)

K=1 accept is fine (~0.45 vs sglang 0.56). The bug is **K>=2 collapse** (accept ~0.067).

Per-layer cosine compare (MPK vs sglang, same prompt + greedy, prefill):
- fc out (attention INPUT): cos 0.9983  ✅ correct
- AFTER-ATTN: cos 0.7267  ❌ **first divergence** (rms smaller on MPK)
- midlayer/norm: cos 0.69  ❌ inherited

Cornered write-vs-read by **un-roping the K dumps** (zero GPU) + per-row best-match:
for each MPK draft cache row `p`, the sglang row whose PRE-ROPE K best-matches is at
**cos ~0.99**, shifted by a constant **+4**:
`MPK row0<->sgl3 (0.997), row2<->5 (0.993), row3<->6 (0.990), row8<->11 (0.996), row10<->13 (0.994)`.

=> **MPK draft K/V CONTENT is correct (it IS sglang's draft K) but stored at a cache row
offset by +mbt+1 (=4: mbt=3 prefill-chunk + the eagle3 +1 input shift).** The draft attends a
content-correct-but-position-shifted cache -> after-attn cos 0.73 -> low accept.
This is a **position/index bug in the draft KV WRITE path**, NOT content/rope/aux.

### Eliminated (do NOT re-investigate without new evidence)
target forward (cos 0.997+), aux-OOD (cos 0.9975+, RMS matches sglang), draft fc (0.9983),
read-side geometry (PR3 independent mapping — after-attn unchanged), draft-KV holes (prefill-aware
fix removed them; accept unchanged), QK-norm (none in SpecForge draft), rope-on-query
(un-roping didn't lift cosine; row-shift did).

## The fix (resume here)
Align the draft KV write row to the draft's true sequence position, removing the
mbt-chunk + eagle3-shift offset.
1. **write-row probe**: print actual write rows `[seq_len_override - num_tokens_override .. seq_len_override)`
   per prefill iter in the draft attention kernel to pin the exact line injecting the +4
   (candidates: prepare_next_batch draft `base` vs the draft attn write row).
2. fix the write-row calc.
3. verify: un-rope best-match collapses to row p<->p cos 0.99; after-attn cos -> ~1.0; K=2 accept up.

## Harness contents (all env-gated)

**MPK side (this repo, in this debug commit):**
- `layer_capture` task: `eagle3_ops.cuh` kernel + registrations across
  `task_register.{h,cc}`, `graph.cc`, `runtime.cc`, `runtime_header.h` (TASK_LAYER_CAPTURE=234),
  `tma.cuh` (no-TMA case). `persistent_kernel.py:layer_capture_layer`.
- `builder.py:build_draft_extend(capture=...)` hook captures fc/attn_proj/midlayer/norm at step 0.
- demo blocks: `MPK_DRAFT_DUMP` (draft sub-steps), `MPK_DRAFTKV_DUMP` (draft K/V cache content).
- scripts: `run_sglang_target_greedy.py`, `sglang_layer_dump.py`.

**sglang side (SEPARATE repo `~/sglang` — NOT pullable with this branch):** see
`eagle3-sglang-debug-harness.patch` in this dir. Hooks: `qwen3_moe.py` (target capture),
`llama_eagle3.py` (draft sub-steps), `llama.py` (draft post-rope K/V + positions, MPK_SGLANG_DRAFTKV_DUMP).
Apply on the new machine: `cd ~/sglang && git apply <path>/eagle3-sglang-debug-harness.patch`.

## Reproducible compare method
- Use SpecForge draft on BOTH sides (MPK default is RedHatAI; pass `--eagle3-draft-path` to the
  SpecForge snapshot) for apples-to-apples.
- MPK_DRAFTKV_DUMP: page_size>=max_seq so page0 row p = abs pos p. sglang side: lock the first
  forward whose input_ids carry >=prompt_len real tokens; also dump `positions`.
- Un-rope (numpy, no GPU): `inv_freq = 1/theta^(arange(0,128,2)/128)`, theta=10000, half-rotation
  `x1=k[:64], x2=k[64:]`, `unrope(k,pos)=cat([x1*cos+x2*sin, x2*cos-x1*sin])`. Un-rope every row both
  sides, then per-row best-match cosine -> reveals the +4 row offset at cos 0.99.

## Env / run notes (carry to new machine)
- MPK env `mirage00`; sglang env `sglang`. `LD_LIBRARY_PATH` must include `/usr/mpi/gcc/openmpi-4.1.9a1/lib`.
- The `MAX_TOKENS=6` edit in `attention_sm100.cuh` is a run-only edit — keep OUT of commits.
- Recurring node fault: `torch.cuda.device_count()==0` while nvidia-smi shows GPUs / GPU0 "Unknown Error"
  — needs a node/GPU reset, not a code fix.
