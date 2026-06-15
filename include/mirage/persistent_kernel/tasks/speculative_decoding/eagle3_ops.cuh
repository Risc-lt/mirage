/* Copyright 2025 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include "tasks/common/common_header.cuh"

namespace kernel {

// ============================================================================
// Eagle3 Operations
//
// Kernels supporting Eagle3 speculative decoding:
//   1. copy_layer_kernel        — capture target's aux hidden states (memcpy)
//   2. concat_kernel            — concat N H-dim tensors → N*H along dim 1
//                                 (N=3 for fc/eh_proj, N=2 for draft QKV input)
//   3. eagle3_d2t_remap_kernel  — hot vocab id → target vocab id via d2t table
// ============================================================================

// --- Generic Memcpy ---
// Copy a contiguous (BATCH_SIZE, HIDDEN_DIM) tensor from src to dst.
// Used by Eagle3 to capture target's intermediate hidden states into dedicated
// aux buffers (since MPK's per-layer intermediates are reused across layers).
template <typename T, int BATCH_SIZE, int HIDDEN_DIM>
__device__ __forceinline__ void
    copy_layer_kernel(void const *__restrict__ src_ptr,
                      void *__restrict__ dst_ptr) {
  T const *__restrict__ src = static_cast<T const *>(src_ptr);
  T *__restrict__ dst = static_cast<T *>(dst_ptr);

  int const total = BATCH_SIZE * HIDDEN_DIM;
  int const tid = threadIdx.x;
  int const stride = blockDim.x;

  for (int i = tid; i < total; i += stride) {
    dst[i] = src[i];
  }
}

// --- DEBUG: position-indexed layer capture (per-layer/sub-step compare) ---
// Copy the src [NUM_ROWS, HIDDEN] tensor into a persistent dst[MAX_SEQ_LEN,
// HIDDEN] buffer at absolute rows [step .. step+NUM_ROWS), so the mbt prompt
// positions processed in one prefill chunk all land at their true positions
// (prefill advances step by NUM_ROWS=mbt/iter; capturing only row 0 left
// stride-mbt holes). Env-gated harness; not used in production paths.
template <typename T, int HIDDEN_DIM, int MAX_SEQ_LEN, int NUM_ROWS>
__device__ __forceinline__ void
    layer_capture_kernel(void const *__restrict__ src_ptr,
                         void const *__restrict__ step_ptr,
                         void *__restrict__ dst_ptr,
                         int request_id) {
  T const *__restrict__ src = static_cast<T const *>(src_ptr);
  T *__restrict__ dst = static_cast<T *>(dst_ptr);
  int const *__restrict__ step = static_cast<int const *>(step_ptr);
  int const base = step[request_id];
  int const tid = threadIdx.x;
  int const stride = blockDim.x;
  for (int r = 0; r < NUM_ROWS; r++) {
    int const row = base + r;
    if (row < 0 || row >= MAX_SEQ_LEN) {
      continue;
    }
    for (int i = tid; i < HIDDEN_DIM; i += stride) {
      dst[row * HIDDEN_DIM + i] = src[r * HIDDEN_DIM + i];
    }
  }
}

// --- Tensor Concatenation along dim 1 ---
// Concatenates N (BATCH_SIZE, HIDDEN_DIM) tensors along dim 1, producing
// (BATCH_SIZE, N * HIDDEN_DIM).
//
// Layout: output[b, k*H .. (k+1)*H) = inputs[k][b, :]   for k in [0, N)
//
// Generic helper (not Eagle3-specific). Current Eagle3 uses:
//   - N=3: combine aux h_low / h_mid / h_high before `eh_proj` (fc) (3H → H).
//   - N=2: concat(embed, hidden) for the draft attention QKV input (2H),
//          matching sglang's torch.cat([embeds, hidden_states], dim=-1).
template <typename T, int BATCH_SIZE, int HIDDEN_DIM, int N>
__device__ __forceinline__ void
    concat_kernel(void const *const *__restrict__ input_ptrs,
                  void *__restrict__ output_ptr) {
  T *__restrict__ output = static_cast<T *>(output_ptr);

  int const total = BATCH_SIZE * HIDDEN_DIM;
  int const tid = threadIdx.x;
  int const stride = blockDim.x;

  for (int i = tid; i < total; i += stride) {
    int b = i / HIDDEN_DIM;
    int d = i % HIDDEN_DIM;
    int out_base = b * N * HIDDEN_DIM;
#pragma unroll
    for (int k = 0; k < N; k++) {
      T const *__restrict__ in = static_cast<T const *>(input_ptrs[k]);
      output[out_base + k * HIDDEN_DIM + d] = in[i];
    }
  }
}

// --- Eagle3 d2t Remap (hot vocab → target vocab) ---
// Eagle3 draft head outputs an id in the draft's smaller hot vocabulary
// (draft_vocab_size, typically 32000). The d2t table converts this back to
// the target's full vocab id via:
//
//   target_id = hot_id + d2t[hot_id]
//
// (sglang convention; d2t is signed int64). One thread per batch element.
//
// Padding guard: when lm_head is row-padded (to satisfy TMA 16B alignment),
// argmax can land in the padded range [DRAFT_VOCAB_REAL, padded). d2t only
// has DRAFT_VOCAB_REAL entries, so an OOB read would write garbage. When
// hot_id is out of range, emit 0 — verify will reject (won't match target
// argmax) and the bonus token is committed normally.
//
// Inputs:
//   hot_token: [BATCH_SIZE] int64    — argmax over draft logits
//   d2t:      [DRAFT_VOCAB_REAL] int64
// Outputs:
//   target_token: [BATCH_SIZE] int64 — target vocab id for downstream tasks
template <int BATCH_SIZE, int DRAFT_VOCAB_REAL>
__device__ __forceinline__ void
    eagle3_d2t_remap_kernel(void const *__restrict__ hot_token_ptr,
                            void const *__restrict__ d2t_table_ptr,
                            void *__restrict__ target_token_ptr) {
  long long const *__restrict__ hot =
      static_cast<long long const *>(hot_token_ptr);
  long long const *__restrict__ d2t =
      static_cast<long long const *>(d2t_table_ptr);
  long long *__restrict__ target = static_cast<long long *>(target_token_ptr);

  int b = threadIdx.x;
  if (b < BATCH_SIZE) {
    long long hot_id = hot[b];
    if (hot_id >= 0 && hot_id < (long long)DRAFT_VOCAB_REAL) {
      target[b] = hot_id + d2t[hot_id];
    } else {
      target[b] = 0; // padded-row argmax → sentinel; verify will reject
    }
  }
}

// --- MTP Verify + Commit (merged; draft-extend design, PR2) ---
//
// Replaces the (verify_strict → eagle3_commit) pair on the draft-extend path.
// Folds the strict accept-walk and the token-buffer commit into ONE kernel and
// — crucially — DROPS the `src_slot` selection of `draft_tokens_new`. In the
// draft-extend design the next iteration's draft chain is NOT a slot of this
// iteration's parallel-branch output; it is produced by the extend stage
// (`_build_draft_extend`), which re-seeds the draft from the confirmed tokens +
// the target hidden at the accepted positions. So this kernel's only jobs are:
//
//   1. Strict accept-walk: compare the previous iter's draft chain (carried in
//      tokens[step+1..step+K], written by mtp_draft_token_copy) vs the target's
//      argmax[0..K-1]; accept the matching prefix; accepted_count = (#accepted)
//      + 1 for the bonus token (lies in [1, K+1]).
//   2. Write the confirmed tokens (= target argmax over the accepted prefix +
//      bonus) into tokens_buffer at [step+1 .. step+accepted_count], guarded
//      against overwriting the prompt.
//   3. Publish accepted_count to new_token_nums[req] (scheduler-contract field;
//      the OFFLINE runtime's prepare_next_batch advances target step by it).
//   4. Publish accepted_count to the in-graph `accepted_count_out` consumed by
//      hidden_gather_accepted + the draft extend builder (this iteration).
//
// It does NOT write next-iter drafts and does NOT read any draft slot beyond
// the accept-walk comparison, so accept-0 reads no rejected draft slot (AC-9).
//
// Pluggable acceptance: AcceptPolicy is a compile-time selector. STRICT_GREEDY
// is the only policy implemented here (the probabilistic path stays standalone
// per the path boundaries); the enum is the seam for future policies.
//
// Inputs:
//   argmax_out        [K+1]                  int64 — target argmax (K+1 pos)
//   step              [MAX_REQ]              int32 — current confirmed length
//   prompt_length     [MAX_REQ]              int32 — req's prompt length
//   tokens_buffer also supplies the draft chain at [step+1..step+K] (see #1)
// Outputs:
//   tokens_buffer     [MAX_REQ, MAX_SEQ_LEN] int64 — confirmed-token write
//   new_token_nums    [MAX_REQ]              int32 — accepted_count for runtime
//   accepted_count_out[1]                    int32 — in-graph accepted_count
//   accept_hist       [..]                   int32 — optional instrumentation
enum class AcceptPolicy { STRICT_GREEDY = 0 };

template <int K,
          int MAX_SEQ_LEN,
          AcceptPolicy POLICY = AcceptPolicy::STRICT_GREEDY>
__device__ __forceinline__ void
    mtp_verify_commit_kernel(void const *__restrict__ argmax_out_ptr,
                             void const *__restrict__ step_ptr,
                             void const *__restrict__ prompt_length_ptr,
                             void *__restrict__ tokens_buffer_ptr,
                             void *__restrict__ new_token_nums_ptr,
                             void *__restrict__ accepted_count_out_ptr,
                             void *__restrict__ accept_hist_ptr,
                             int request_id) {
  static_assert(POLICY == AcceptPolicy::STRICT_GREEDY,
                "mtp_verify_commit: only STRICT_GREEDY implemented in PR2");

  long long const *__restrict__ argmax =
      static_cast<long long const *>(argmax_out_ptr);
  int const *__restrict__ step = static_cast<int const *>(step_ptr);
  int const *__restrict__ prompt_length =
      static_cast<int const *>(prompt_length_ptr);
  long long *__restrict__ tokens = static_cast<long long *>(tokens_buffer_ptr);
  int *__restrict__ new_token_nums = static_cast<int *>(new_token_nums_ptr);
  int *__restrict__ accepted_count_out =
      static_cast<int *>(accepted_count_out_ptr);

  int t_id = threadIdx.x;
  int req = request_id;

  int cur_step = step[req];
  int prompt_len = prompt_length[req];

  // 1. Strict accept-walk (single-thread; K is tiny so no need to parallelize).
  //    The draft chain for THIS iter lives in tokens[cur_step+1 .. cur_step+K]:
  //    the previous iter's draft-token copy (mtp_draft_token_copy) wrote it
  //    there, and prepare_next_batch advanced step to point at it. tokens is an
  //    attach_input carried across the iteration barrier, so this read is the
  //    same cross-iter carrier drafts_prev used to be (BL-20260610). Step 2
  //    below overwrites tokens[cur_step+1 ..] with the confirmed argmax, but
  //    the
  //    __syncthreads() guarantees every thread finishes this read first.
  __shared__ int ac_smem;
  if (t_id == 0) {
    int accepted = K;
    for (int i = 0; i < K; i++) {
      long long draft_i = tokens[req * MAX_SEQ_LEN + cur_step + 1 + i];
      if (draft_i != argmax[i]) {
        accepted = i;
        break;
      }
    }
    ac_smem = accepted + 1; // +1 bonus token; in [1, K+1]
  }
  __syncthreads();
  int ac = ac_smem;

  // 2. Write confirmed tokens at step+1 .. step+ac (only past prompt). Values
  //    come from the target argmax over the accepted prefix + bonus.
  if (t_id < ac) {
    int pos = cur_step + 1 + t_id;
    if (pos < MAX_SEQ_LEN && pos >= prompt_len) {
      tokens[req * MAX_SEQ_LEN + pos] = argmax[t_id];
    }
  }

  // 3/4. Publish accepted_count to the runtime (scheduler-contract) and to the
  //      in-graph consumers (hidden_gather_accepted + draft extend).
  if (t_id == 0) {
    new_token_nums[req] = ac;
    accepted_count_out[0] = ac;
    if (accept_hist_ptr != nullptr) {
      int *hist = static_cast<int *>(accept_hist_ptr);
      atomicAdd(&hist[ac], 1);
    }
  }
}

} // namespace kernel
