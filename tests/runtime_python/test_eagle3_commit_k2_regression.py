"""Regression test for the K>1 `src_slot=0` accept-chain collapse (PR2 / AC-6).

History: the legacy eagle3_commit_kernel selected the next-iteration draft chain
with `src_slot = (K>1) ? 0 : (ac-1)`, forcing slot 0 for K>1 and committing a
misaligned chain — collapsing the K>1 accept rate.

PR2 fix: eagle3_commit is replaced by `mtp_verify_commit`, which does the strict
accept-walk + confirmed-token write + accepted_count and DROPS chain selection
entirely (the next chain is produced by the draft-extend stage, not by picking a
slot of this iteration's parallel-branch output). So the correct post-fix
behavior has NO src_slot logic at all.

This test pins the correct mtp_verify_commit behavior for K in {1,2,3}:
  - accepted_count = (#matching-prefix) + 1 (bonus), in [1, K+1];
  - confirmed tokens at [step+1 .. step+accepted_count] = target argmax;
  - identical logic for K=1 and K>1 (no K-dependent slot branch).

Oracle note (BL-20260609): correctness here is internal to MPK (the verify+commit
contract), NOT a cross-stack token match — no sglang/bf16-tie concerns.

Build the extension first:
    cd tests/runtime_python && python setup.py build_ext --inplace
"""
import unittest

import torch

import runtime_kernel


def _run_verify_commit(K, draft_ids, argmax, *, step0=0, prompt_len=0,
                       max_seq_len=512):
    """Run mtp_verify_commit for one request. draft_ids: [K] i64 (this iter's
    draft chain); argmax: [K+1] i64 (target argmax over K+1 verify positions).
    Returns (accepted_count, confirmed_tokens_written)."""
    MAX_REQ = 1
    req = 0
    draft_t = torch.tensor(draft_ids, dtype=torch.int64, device="cuda")
    argmax_t = torch.tensor(argmax, dtype=torch.int64, device="cuda")
    step = torch.tensor([step0], dtype=torch.int32, device="cuda")
    prompt_length = torch.tensor([prompt_len], dtype=torch.int32, device="cuda")
    tokens_buffer = torch.full((MAX_REQ, max_seq_len), -1, dtype=torch.int64,
                               device="cuda")
    new_token_nums = torch.full((MAX_REQ,), -1, dtype=torch.int32,
                                device="cuda")
    accepted_count_out = torch.full((1,), -1, dtype=torch.int32, device="cuda")
    accept_hist = torch.zeros((K + 2,), dtype=torch.int32, device="cuda")

    runtime_kernel.mtp_verify_commit(
        draft_t, argmax_t, step, prompt_length, tokens_buffer,
        new_token_nums, accepted_count_out, accept_hist, K, max_seq_len, req)
    torch.cuda.synchronize()

    ac = int(accepted_count_out[0].item())
    # confirmed tokens written at [step+1 .. step+ac]
    written = tokens_buffer[req, step0 + 1:step0 + 1 + ac].cpu().tolist()
    return ac, written, int(new_token_nums[req].item())


def _expected_ac(K, draft_ids, argmax):
    """Strict accept-walk reference: #matching prefix + 1 bonus."""
    accepted = K
    for i in range(K):
        if draft_ids[i] != argmax[i]:
            accepted = i
            break
    return accepted + 1


class Eagle3CommitK2Regression(unittest.TestCase):
    def test_k2_no_slot_selection_correct_commit(self):
        """K=2: verify+commit must accept the matching prefix and write the
        confirmed tokens from the target argmax — with NO K-dependent slot
        branch (the old src_slot=0 collapse is gone)."""
        K = 2
        # draft chain [11, 12]; target argmax over K+1=3 positions [11, 99, 7].
        # Strict: draft[0]=11==argmax[0]=11 accept; draft[1]=12 != argmax[1]=99
        # → accepted prefix = 1, ac = 2. Confirmed = argmax[0:2] = [11, 99].
        draft = [11, 12]
        argmax = [11, 99, 7]
        ac, written, ntn = _run_verify_commit(K, draft, argmax)
        self.assertEqual(ac, _expected_ac(K, draft, argmax))  # = 2
        self.assertEqual(ac, 2)
        self.assertEqual(written, [11, 99])
        self.assertEqual(ntn, 2)  # new_token_nums == accepted_count

    def test_k2_all_accept(self):
        """K=2, all drafts match → ac = K+1 = 3, confirmed = all argmax."""
        K = 2
        draft = [11, 12]
        argmax = [11, 12, 55]
        ac, written, ntn = _run_verify_commit(K, draft, argmax)
        self.assertEqual(ac, 3)
        self.assertEqual(written, [11, 12, 55])
        self.assertEqual(ntn, 3)

    def test_k2_accept_zero(self):
        """K=2, first draft already mismatches → ac = 1 (bonus only)."""
        K = 2
        draft = [11, 12]
        argmax = [99, 0, 0]
        ac, written, ntn = _run_verify_commit(K, draft, argmax)
        self.assertEqual(ac, 1)
        self.assertEqual(written, [99])
        self.assertEqual(ntn, 1)

    def test_k1_unchanged(self):
        """K=1 must remain correct: match → ac=2; mismatch → ac=1."""
        # match
        ac, written, _ = _run_verify_commit(1, [7], [7, 8])
        self.assertEqual((ac, written), (2, [7, 8]))
        # mismatch
        ac, written, _ = _run_verify_commit(1, [7], [9, 8])
        self.assertEqual((ac, written), (1, [9]))

    def test_k3_partial(self):
        """K=3, accept 2 then mismatch → ac=3, confirmed = argmax[0:3]."""
        K = 3
        draft = [1, 2, 3]
        argmax = [1, 2, 88, 0]
        ac, written, ntn = _run_verify_commit(K, draft, argmax)
        self.assertEqual(ac, 3)
        self.assertEqual(written, [1, 2, 88])
        self.assertEqual(ntn, 3)


if __name__ == "__main__":
    unittest.main()
