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
// MTP Token Operations
//
// Small utility kernels for MTP speculative decoding token management.
// ============================================================================

// --- Token Scatter ---
// Copy a single token ID per batch element from src[batch, 1] to a specific
// column of dst[batch, num_draft_tokens].
// Used after each MTP argmax step to collect draft tokens.
//
// Inputs:
//   src: [BATCH_SIZE, 1] int64 — single draft token from argmax
//   dst: [BATCH_SIZE, NUM_SLOTS] int64 — collection buffer
// Params:
//   SLOT_IDX: which column to write (compile-time, from static unroll)
//   NUM_SLOTS: total columns in dst
template <int BATCH_SIZE, int NUM_SLOTS, int SLOT_IDX>
__device__ __forceinline__ void
    mtp_token_scatter_kernel(void const *__restrict__ src_ptr,
                             void *__restrict__ dst_ptr) {

  long long const *__restrict__ src = static_cast<long long const *>(src_ptr);
  long long *__restrict__ dst = static_cast<long long *>(dst_ptr);

  int b = threadIdx.x;
  if (b < BATCH_SIZE) {
    dst[b * NUM_SLOTS + SLOT_IDX] = src[b];
  }
}

// --- Float Scatter (for draft probabilities) ---
// Same pattern as token scatter but for float32 values.
// Used to accumulate P_draft(token) per draft step.
template <int BATCH_SIZE, int NUM_SLOTS, int SLOT_IDX>
__device__ __forceinline__ void
    mtp_float_scatter_kernel(void const *__restrict__ src_ptr,
                             void *__restrict__ dst_ptr) {
  float const *__restrict__ src = static_cast<float const *>(src_ptr);
  float *__restrict__ dst = static_cast<float *>(dst_ptr);
  int b = threadIdx.x;
  if (b < BATCH_SIZE) {
    dst[b * NUM_SLOTS + SLOT_IDX] = src[b];
  }
}

// --- Draft Tokens to Sequence Buffer ---
// After MTP draft generation, copy K draft tokens from all_draft_ids into the
// main token sequence buffer (config.tokens) at the correct positions.
// Also writes the main model's token (bonus/base) at position 0.
//
// This prepares the input for the next iteration's verification forward:
//   tokens[request, step+1] = main_token
//   tokens[request, step+2] = draft_0
//   ...
//   tokens[request, step+K+1] = draft_{K-1}
//
// Inputs:
//   main_token:     [BATCH_SIZE, 1] int64 — main model's argmax output
//   draft_tokens:   [BATCH_SIZE, NUM_DRAFT] int64 — collected draft tokens
//   tokens_buffer:  [MAX_REQUESTS, MAX_SEQ_LEN] int64 — full token sequence
//   step:           [MAX_REQUESTS] int32 — current step per request
// Outputs:
//   num_new_tokens: [MAX_REQUESTS] int32 — set to NUM_DRAFT + 1
template <int NUM_DRAFT, int MAX_SEQ_LEN>
__device__ __forceinline__ void
    mtp_prepare_verify_input_kernel(void const *__restrict__ main_token_ptr,
                                    void const *__restrict__ draft_tokens_ptr,
                                    void *__restrict__ tokens_buffer_ptr,
                                    void const *__restrict__ step_ptr,
                                    void *__restrict__ num_new_tokens_ptr,
                                    int request_id) {

  long long const *__restrict__ main_token =
      static_cast<long long const *>(main_token_ptr);
  long long const *__restrict__ draft_tokens =
      static_cast<long long const *>(draft_tokens_ptr);
  long long *__restrict__ tokens = static_cast<long long *>(tokens_buffer_ptr);
  int const *__restrict__ step = static_cast<int const *>(step_ptr);
  int *__restrict__ num_new_tokens = static_cast<int *>(num_new_tokens_ptr);

  int t_id = threadIdx.x;
  // Use task metadata request_id (not blockIdx.x which is worker block ID
  // in persistent kernel)
  int req = request_id;

  int cur_step = step[req];

  // Thread 0: write main token at step+1, set num_new_tokens
  if (t_id == 0) {
    if (cur_step + 1 < MAX_SEQ_LEN) {
      tokens[req * MAX_SEQ_LEN + cur_step + 1] = main_token[req];
    }
    // Clamp num_new_tokens so we don't exceed MAX_SEQ_LEN
    int max_new = MAX_SEQ_LEN - cur_step - 1;
    if (max_new < 0) {
      max_new = 0;
    }
    num_new_tokens[req] = (NUM_DRAFT + 1 < max_new) ? NUM_DRAFT + 1 : max_new;
  }

  // Threads 0..NUM_DRAFT-1: write draft tokens (bounds-checked)
  if (t_id < NUM_DRAFT) {
    int write_pos = cur_step + 2 + t_id;
    if (write_pos < MAX_SEQ_LEN) {
      tokens[req * MAX_SEQ_LEN + write_pos] =
          draft_tokens[req * NUM_DRAFT + t_id];
    }
  }
}

// --- Build MTP Embedding Input (vLLM-aligned) ---
// vLLM's MTP (see vllm/v1/spec_decode/eagle.py L666-669) embeds shifted
// ground-truth prompt tokens during prefill, falling back to generated tokens
// for positions past the prompt boundary. Prior MPK behavior used main model's
// argmax for all mbt positions, which equals ground-truth only when main is
// trained accurately (wrong for partial-layer tests).
//
// This kernel constructs MTP's per-iteration embedding input:
//   For i in [0..BATCH_SIZE-2]: read tokens[req, step+i+1] (shifted prompt
//       positions, populated in demo.py's tokens buffer from the actual prompt
//       and, for later iterations, from previous iterations'
//       prepare_next_batch copies).
//   For i == BATCH_SIZE-1: read output_tokens[i] = main's argmax for the
//       current iteration's last position. This is the token that would have
//       been appended to the sequence had prepare_next_batch run first.
//
// Prefill (step=0, mbt=M, prompt fills tokens[0..M-1]):
//   i < M-1 → tokens[step+i+1] = prompt[i+1]  (ground truth)
//   i == M-1 → output_tokens[M-1] = main's first generated token
//
// Decode (step=N, mbt=1):
//   Only i == BATCH_SIZE-1 = 0 path runs → output_tokens[0] = current argmax
//
// Inputs:
//   tokens_buffer:   [MAX_REQUESTS, MAX_SEQ_LEN] int64 — full sequence buffer
//   output_tokens:   [BATCH_SIZE, 1] int64 — current iter's argmax
//   step:            [MAX_REQUESTS] int32 — per-request step
// Outputs:
//   mtp_input_tokens: [BATCH_SIZE, 1] int64 — MTP's embed input
template <int BATCH_SIZE, int MAX_SEQ_LEN>
__device__ __forceinline__ void
    mtp_build_embed_input_kernel(void *__restrict__ mtp_input_tokens_ptr,
                                 void const *__restrict__ tokens_buffer_ptr,
                                 void const *__restrict__ output_tokens_ptr,
                                 void const *__restrict__ step_ptr,
                                 void const *__restrict__ prompt_length_ptr,
                                 int request_id) {
  long long *__restrict__ mtp_input =
      static_cast<long long *>(mtp_input_tokens_ptr);
  long long const *__restrict__ tokens =
      static_cast<long long const *>(tokens_buffer_ptr);
  long long const *__restrict__ output_tokens =
      static_cast<long long const *>(output_tokens_ptr);
  int const *__restrict__ step = static_cast<int const *>(step_ptr);
  int const *__restrict__ prompt_length =
      static_cast<int const *>(prompt_length_ptr);

  int req = request_id;
  int cur_step = step[req];
  int plen = prompt_length[req];

  // Position-aware teacher-forcing (eagle3 draft EXTEND seed). Row i seeds the
  // draft for absolute sequence position cur_step+i; its embed token is the
  // NEXT token at cur_step+i+1.
  //   PREFILL (cur_step+i+1 < plen): the next token is the real prompt token,
  //     so teacher-force from tokens[cur_step+i+1] for ALL rows (including the
  //     last) — using the main model's argmax there would feed a mispredicted
  //     token and corrupt the draft seed.
  //   DECODE (cur_step+i+1 >= plen): no ground truth ahead; use the main
  //     model's argmax (output_tokens[i]) as the autoregressive seed.
  for (int i = threadIdx.x; i < BATCH_SIZE; i += blockDim.x) {
    int pos = cur_step + i + 1;
    long long val;
    if (pos < plen && pos < MAX_SEQ_LEN) {
      val = tokens[req * MAX_SEQ_LEN + pos];
    } else {
      val = output_tokens[i];
    }
    mtp_input[i] = val;
  }
}

// --- Build the draft KV mapping (independent draft paged-KV) ---
// PR1 (behavior-preserving): mirror the target paged-KV mapping into the draft
// mapping so the draft attention/gather can read its OWN draft_* buffers while
// reproducing the current (shared-mapping) behavior exactly. This decouples the
// draft from the target's qo_indptr without changing any output; the later
// accepted-count-driven independent advance (and the draft's own page
// free-list) replaces this mirror once draft-extend exists.
//
// Single-threaded (free-list / indptr writes are inherently sequential).
//
// Inputs  (target mapping, read):  qo_indptr, paged_kv_indptr,
//   paged_kv_indices, paged_kv_last_page_len, step
// Outputs (draft mapping, write):  draft_qo_indptr, draft_paged_kv_indptr,
//   draft_paged_kv_indices, draft_paged_kv_last_page_len, draft_step
// MAX_BATCHED_REQS = MPK_MAX_NUM_BATCHED_REQUESTS (compile-time).
template <int MAX_BATCHED_REQS>
__device__ __forceinline__ void mtp_build_draft_indptr_kernel(
    void const *__restrict__ qo_indptr_ptr,
    void const *__restrict__ paged_kv_indptr_ptr,
    void const *__restrict__ paged_kv_indices_ptr,
    void const *__restrict__ paged_kv_last_page_len_ptr,
    void const *__restrict__ step_ptr,
    void *__restrict__ draft_qo_indptr_ptr,
    void *__restrict__ draft_paged_kv_indptr_ptr,
    void *__restrict__ draft_paged_kv_indices_ptr,
    void *__restrict__ draft_paged_kv_last_page_len_ptr,
    void *__restrict__ draft_step_ptr,
    int total_num_requests) {
  if (threadIdx.x != 0) {
    return;
  }
  int const *__restrict__ qo_indptr = static_cast<int const *>(qo_indptr_ptr);
  int const *__restrict__ paged_kv_indptr =
      static_cast<int const *>(paged_kv_indptr_ptr);
  int const *__restrict__ paged_kv_indices =
      static_cast<int const *>(paged_kv_indices_ptr);
  int const *__restrict__ paged_kv_last_page_len =
      static_cast<int const *>(paged_kv_last_page_len_ptr);
  int const *__restrict__ step = static_cast<int const *>(step_ptr);
  int *__restrict__ draft_qo_indptr = static_cast<int *>(draft_qo_indptr_ptr);
  int *__restrict__ draft_paged_kv_indptr =
      static_cast<int *>(draft_paged_kv_indptr_ptr);
  int *__restrict__ draft_paged_kv_indices =
      static_cast<int *>(draft_paged_kv_indices_ptr);
  int *__restrict__ draft_paged_kv_last_page_len =
      static_cast<int *>(draft_paged_kv_last_page_len_ptr);
  int *__restrict__ draft_step = static_cast<int *>(draft_step_ptr);

  for (int i = 0; i < total_num_requests; i++) {
    draft_step[i] = step[i];
  }
  for (int i = 0; i <= MAX_BATCHED_REQS; i++) {
    draft_qo_indptr[i] = qo_indptr[i];
    draft_paged_kv_indptr[i] = paged_kv_indptr[i];
  }
  for (int i = 0; i < MAX_BATCHED_REQS; i++) {
    draft_paged_kv_last_page_len[i] = paged_kv_last_page_len[i];
  }
  int const total_pages = paged_kv_indptr[MAX_BATCHED_REQS];
  for (int i = 0; i < total_pages; i++) {
    draft_paged_kv_indices[i] = paged_kv_indices[i];
  }
}

// --- Hidden Gather Accepted (draft-extend seed, PR2) ---
//
// Selects the target verify-hidden rows at the ACCEPTED positions
// (rows 0..accepted_count-1) into a contiguous [accepted_count, H] seed buffer
// consumed by `_build_draft_extend`. accepted_count comes in-graph from
// mtp_verify_commit and lies in [1, K+1] (always >=1 because of the bonus
// token), so the gather always copies at least the bonus position and NEVER
// reads a rejected draft slot (rows >= accepted_count are zero-filled, not read
// from the draft) — satisfying the AC-9 accept-0 invariant by construction.
//
// Grid: (K+1, 1, 1) — one block per potential row; block copies H elements.
// Inputs:
//   verify_hidden:    [K+1, H]  bf16 — target hidden at the K+1 verify
//   positions accepted_count:   [1]       int32 — from mtp_verify_commit
//   (in-graph)
// Outputs:
//   extend_seed:      [K+1, H]  bf16 — rows [0..ac-1] = verify_hidden; rest = 0
template <typename T, int NUM_DRAFT, int HIDDEN_DIM>
__device__ __forceinline__ void
    hidden_gather_accepted_kernel(void const *__restrict__ verify_hidden_ptr,
                                  void const *__restrict__ accepted_count_ptr,
                                  void *__restrict__ extend_seed_ptr) {
  T const *__restrict__ verify_hidden =
      static_cast<T const *>(verify_hidden_ptr);
  int const *__restrict__ accepted_count =
      static_cast<int const *>(accepted_count_ptr);
  T *__restrict__ extend_seed = static_cast<T *>(extend_seed_ptr);

  int row = blockIdx.x; // 0 .. NUM_DRAFT (K+1 rows total)
  if (row > NUM_DRAFT) {
    return;
  }
  int ac = accepted_count[0]; // in [1, NUM_DRAFT+1]

  if (row < ac) {
    // Copy verify_hidden[row, :] -> extend_seed[row, :]
    for (int c = threadIdx.x; c < HIDDEN_DIM; c += blockDim.x) {
      extend_seed[row * HIDDEN_DIM + c] = verify_hidden[row * HIDDEN_DIM + c];
    }
  } else {
    // Zero-fill rows beyond the accepted prefix (never read from the draft).
    for (int c = threadIdx.x; c < HIDDEN_DIM; c += blockDim.x) {
      extend_seed[row * HIDDEN_DIM + c] = static_cast<T>(0.0f);
    }
  }
}

// --- MTP Draft Token Copy (next-iter draft chain → global tokens, PR2) ---
//
// After all draft steps complete, copies THIS iteration's draft chain (the
// row-0 chain of all_draft_ids, [0..K-1]) into the global token buffer at
// tokens[step+ac+1 .. step+ac+K], so NEXT iteration's verify forward reads it
// as its K+1 candidate input AND mtp_verify_commit reads it (at tokens[step'+1
// .. step'+K] after prepare_next_batch advances step' = step+ac) for the
// accept-walk.
//
// This restores the only draft-token write that the legacy eagle3_commit_kernel
// performed (eagle3_ops.cuh:229-234), sourced from the extend output. The
// separate drafts_prev snapshot the legacy kernel ALSO did is now redundant —
// tokens is itself the cross-iter carrier (an attach_input, NOT a tracked graph
// edge; the iteration barrier carries iter N's value to iter N+1), so the draft
// chain lives in exactly one place (BL-20260610). No race: prepare_next_batch's
// decode branch writes no tokens, and iter N+1's verify reads these positions
// before its own step-2 overwrite (guarded by __syncthreads).
//
// Single linear chain per request ⇒ copy the row-0 chain.
// Grid: (1, 1, 1).
// Inputs:
//   all_draft_ids:  [mbt, K]              int64 — this iter's draft chains
//   step:           [MAX_REQ]             int32 — current confirmed length
//   prompt_length:  [MAX_REQ]             int32 — req's prompt length
//   accepted_count: [1]                   int32 — ac from mtp_verify_commit
// Outputs:
//   tokens_buffer:  [MAX_REQ, MAX_SEQ_LEN] int64 — global seq buffer (attach)
template <int K, int MAX_SEQ_LEN>
__device__ __forceinline__ void
    mtp_draft_token_copy_kernel(void const *__restrict__ all_draft_ids_ptr,
                                void const *__restrict__ step_ptr,
                                void const *__restrict__ prompt_length_ptr,
                                void const *__restrict__ accepted_count_ptr,
                                void *__restrict__ tokens_buffer_ptr,
                                int request_id) {
  long long const *__restrict__ all_draft_ids =
      static_cast<long long const *>(all_draft_ids_ptr);
  int const *__restrict__ step = static_cast<int const *>(step_ptr);
  int const *__restrict__ prompt_length =
      static_cast<int const *>(prompt_length_ptr);
  int const *__restrict__ accepted_count =
      static_cast<int const *>(accepted_count_ptr);
  long long *__restrict__ tokens = static_cast<long long *>(tokens_buffer_ptr);
  int req = request_id;
  int cur_step = step[req];
  int prompt_len = prompt_length[req];
  int ac = accepted_count[0];
  int t_id = threadIdx.x;
  if (t_id < K) {
    // LAST-lane chain → tokens[step+ac+1+t_id]. The draft is mbt-parallel; the
    // chain that continues the decode is lane (ac-1) — the last consumed token
    // (= the position the draft attention chains on via q_row_offset), NOT lane
    // 0 (the first consumed token, which shifts the chain back by ac-1 and
    // collapses the accept rate for ac>=2). ac = accepted_count == the EXTEND
    // span in decode. Guard against the seq bound AND the prompt region: during
    // prefill iters (step < prompt_len) the draft produces garbage, so writes
    // into [0, prompt_len) must be inert (BL-20260610).
    int row = ac > 0 ? ac - 1 : 0;
    int pos = cur_step + ac + 1 + t_id;
    if (pos < MAX_SEQ_LEN && pos >= prompt_len) {
      tokens[req * MAX_SEQ_LEN + pos] = all_draft_ids[row * K + t_id];
    }
  }
}

} // namespace kernel
