"""Regression test for the K>1 `src_slot=0` accept-chain collapse (PR2 / AC-6).

Bug site: include/mirage/persistent_kernel/tasks/speculative_decoding/eagle3_ops.cuh
    `int src_slot = (K > 1) ? 0 : (ac - 1);`

For K=1 the commit kernel runs `mbt` parallel branches and picks the chain
`src_slot = ac-1` that aligns with the next expected position. For K>1 it
FORCES `src_slot=0` regardless of the accepted count `ac`, so the next-iter
draft chain committed to `drafts_prev` (and into the token buffer) is always
slot 0's chain — even when the accepted chain lived in a different slot. That
commits a misaligned chain for the next iteration and collapses the K>1 accept
rate.

This test exercises the commit stage in isolation via the `runtime_kernel`
extension. It is written to FAIL on the current `src_slot=0` behavior and PASS
once `mtp_verify_commit` (task7) selects the chain aligned with `ac` for K>1.

Oracle note (BL-20260609): correctness here is internal to MPK (which slot the
commit selects), NOT a cross-stack token match — so no sglang/bf16-tie concerns.

Build the extension first:
    cd tests/runtime_python && python setup.py build_ext --inplace
"""
import unittest

import torch

import runtime_kernel


def _run_commit(K, ac, draft_tokens_new, *, step0=0, prompt_len=0,
                max_seq_len=512):
    """Run eagle3_commit for one request and return (committed_tokens_slice,
    drafts_prev_after). `draft_tokens_new` is a [K+1, K] int64 CUDA tensor."""
    batch_size = K + 1
    MAX_REQ = 1
    req = 0

    tokens_buffer = torch.full((MAX_REQ, max_seq_len), -1, dtype=torch.int64,
                               device="cuda")
    # argmax_out: target argmax over K+1 verify positions. Values are
    # irrelevant to the chain-selection bug (they fill the accepted prefix),
    # so use a distinct sentinel range.
    argmax_out = torch.arange(900, 900 + (K + 1), dtype=torch.int64,
                              device="cuda")
    accepted_count = torch.tensor([ac], dtype=torch.int32, device="cuda")
    step = torch.tensor([step0], dtype=torch.int32, device="cuda")
    prompt_length = torch.tensor([prompt_len], dtype=torch.int32,
                                 device="cuda")
    new_token_nums = torch.full((MAX_REQ,), -1, dtype=torch.int32,
                                device="cuda")
    drafts_prev = torch.full((MAX_REQ, K), -1, dtype=torch.int64,
                             device="cuda")
    accept_hist = torch.zeros((K + 2,), dtype=torch.int32, device="cuda")

    runtime_kernel.eagle3_commit(
        tokens_buffer, argmax_out, draft_tokens_new.reshape(-1).contiguous(),
        accepted_count, step, prompt_length, new_token_nums, drafts_prev,
        accept_hist, K, batch_size, max_seq_len, req)
    torch.cuda.synchronize()

    # Next-iter draft chain was written to tokens[step+ac+1 .. step+ac+K] and
    # mirrored into drafts_prev[0..K-1].
    start = step0 + ac + 1
    committed = tokens_buffer[req, start:start + K].clone()
    return committed, drafts_prev[req].clone()


class Eagle3CommitK2Regression(unittest.TestCase):
    def test_k2_selects_chain_aligned_with_ac_not_slot0(self):
        """K=2, ac=2: the committed next-iter chain must come from the chain
        aligned with the accepted count, NOT hard-coded slot 0.

        We make each slot's chain uniquely identifiable so the committed chain
        reveals which slot was selected. Slot 0 holds a sentinel 'garbage'
        chain; the slot aligned with ac=2 (slot ac-1 = slot 1) holds the
        'good' chain. The current kernel forces slot 0 -> commits garbage.
        """
        K = 2
        ac = 2  # accepted 1 draft + bonus
        # draft_tokens_new[slot, t]: slot s, token t -> encode as 10*s + t + 1
        # slot 0 = [1, 2]  (the garbage slot the buggy kernel forces)
        # slot 1 = [11,12] (the chain aligned with ac-1 = 1)
        # slot 2 = [21,22]
        draft = torch.tensor(
            [[1, 2], [11, 12], [21, 22]], dtype=torch.int64, device="cuda")

        committed, drafts_prev = _run_commit(K, ac, draft)

        # Correct behavior: commit the chain aligned with ac (slot ac-1 = 1),
        # i.e. [11, 12]. Buggy behavior commits slot 0 = [1, 2].
        expected = torch.tensor([11, 12], dtype=torch.int64, device="cuda")
        got_committed = committed.cpu().tolist()
        got_drafts_prev = drafts_prev.cpu().tolist()

        self.assertEqual(
            got_committed, expected.cpu().tolist(),
            msg=(f"K>1 committed next-iter chain came from slot 0 "
                 f"(src_slot=0 bug): tokens={got_committed}, expected the "
                 f"ac-aligned chain {expected.cpu().tolist()}"))
        self.assertEqual(
            got_drafts_prev, expected.cpu().tolist(),
            msg=(f"K>1 drafts_prev snapshot came from slot 0 (src_slot=0 bug): "
                 f"{got_drafts_prev}, expected {expected.cpu().tolist()}"))

    def test_k1_unchanged_picks_ac_minus_1(self):
        """K=1 must remain correct (src_slot = ac-1). ac=1 -> slot 0."""
        K = 1
        ac = 1
        draft = torch.tensor([[7], [99]], dtype=torch.int64, device="cuda")
        committed, drafts_prev = _run_commit(K, ac, draft)
        # ac-1 = 0 -> slot 0 chain = [7]
        self.assertEqual(committed.cpu().tolist(), [7])
        self.assertEqual(drafts_prev.cpu().tolist(), [7])


if __name__ == "__main__":
    unittest.main()
