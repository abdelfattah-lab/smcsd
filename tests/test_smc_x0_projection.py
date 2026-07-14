"""Tests for the x0 prefill seed's logits projection (``SMCWorker._forward_extend``).

The x0 seed is drawn from the TARGET's first-token distribution by projecting
the last prompt token's post-norm hidden state through the target's own
``LogitsProcessor`` in DECODE mode (PR #24 review follow-up).  These tests
guard the three properties that projection must have:

* **tp=1 identity** — the DECODE-mode LogitsProcessor projection equals the
  plain ``hidden @ lm_head.weight.T`` matmul when there is a single vocab
  shard and no logit post-processing.
* **post-processing** — ``final_logit_softcapping`` (and friends) ARE applied
  by the processor path; a raw matmul would silently skip them.
* **tp>1 full vocabulary** — under tensor parallelism the lm_head is
  vocab-sharded; the processor all-gathers the shards so every rank sees
  logits over the FULL vocab.  A raw per-rank matmul yields a truncated
  ``vocab/tp`` slice — the silent-wrongness this projection was adopted to
  fix.  Exercised with a 2-process gloo spawn, CPU-only.

Together with ``TestSampleTargetPower`` (test_smc_sampling_equivalence.py),
which checks the Gumbel-max power draw over given logits, this covers the x0
chain end to end at the unit level: hidden -> full-vocab logits -> p_T^alpha.
"""

import json
import os
import socket
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

import torch


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _StubServerArgs:
    """Every flag the LogitsProcessor consults reads as unset/disabled."""

    def __getattr__(self, name):
        return None


def _init_sglang_single_process(port: int):
    """Gloo-backed world-size-1 model parallel, enough to build a LogitsProcessor."""
    from sglang.srt.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    if not torch.distributed.is_initialized():
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            backend="gloo",
        )
        initialize_model_parallel(1, 1)
    import sglang.srt.server_args as server_args_mod

    server_args_mod.set_global_server_args_for_scheduler(_StubServerArgs())
    _stub_dp_attention()


def _stub_dp_attention():
    # DP attention is off in every SMC configuration; the LogitsProcessor only
    # reads these globals to decide it has nothing to do.
    import sglang.srt.layers.dp_attention as dpa

    dpa._ATTN_DP_SIZE = 1
    dpa._LOCAL_ATTN_DP_SIZE = 1
    dpa._ATTN_DP_RANK = 0
    dpa._LOCAL_ATTN_DP_RANK = 0
    dpa._ENABLE_DP_ATTENTION_FLAG = False


def _project(last_hidden, lm_head, softcapping=None):
    """The worker's x0 projection: DECODE-mode LogitsProcessor on last-token
    hidden states — one logit row per request (mirrors _forward_extend)."""
    from sglang.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
    from sglang.srt.model_executor.forward_batch_info import (
        CaptureHiddenMode,
        ForwardMode,
    )

    vocab, hid = lm_head.num_embeddings, lm_head.embedding_dim
    config = SimpleNamespace(
        vocab_size=vocab, hidden_size=hid, final_logit_softcapping=softcapping
    )
    processor = LogitsProcessor(config)
    metadata = LogitsMetadata(
        forward_mode=ForwardMode.DECODE,
        capture_hidden_mode=CaptureHiddenMode.NULL,
    )
    return processor(None, last_hidden, lm_head, metadata).next_token_logits


class TestX0ProjectionTP1(unittest.TestCase):
    """DECODE-mode LogitsProcessor projection at tp=1."""

    @classmethod
    def setUpClass(cls):
        _init_sglang_single_process(_free_port())

    def _make_head(self, vocab=128, hid=16, seed=0):
        from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead

        head = ParallelLMHead(vocab, hid)
        torch.manual_seed(seed)
        head.weight.data.normal_()
        return head

    def test_matches_manual_matmul(self):
        head = self._make_head()
        torch.manual_seed(1)
        hidden = torch.randn(5, 16)
        out = _project(hidden, head)
        ref = (hidden.to(head.weight.dtype) @ head.weight.T).float()
        self.assertEqual(tuple(out.shape), (5, 128))  # one row per request
        self.assertEqual((out - ref).abs().max().item(), 0.0)

    @unittest.skipUnless(
        torch.cuda.is_available(), "softcap uses a Triton kernel (GPU-only)"
    )
    def test_applies_final_logit_softcapping(self):
        # The raw-matmul path this projection replaced skipped softcapping;
        # the processor path must apply cap * tanh(logits / cap).
        head = self._make_head(seed=2)
        head = head.cuda()
        torch.manual_seed(3)
        hidden = (torch.randn(4, 16) * 4.0).cuda()
        cap = 30.0
        out = _project(hidden, head, softcapping=cap)
        raw = (hidden.to(head.weight.dtype) @ head.weight.T).float()
        ref = cap * torch.tanh(raw / cap)
        self.assertLess((out - ref).abs().max().item(), 1e-4)
        self.assertGreater((out - raw).abs().max().item(), 1e-3)

    def test_x0_draw_is_power_target_distributed(self):
        # Chain the projection into the worker's Gumbel-max power draw and
        # check x0's empirical law against softmax(alpha * logits / T).
        head = self._make_head(vocab=50, hid=16, seed=4)
        torch.manual_seed(5)
        hidden = torch.randn(1, 16)
        logits = _project(hidden, head)
        alpha, temperature = 2.0, 0.8

        n = 200_000
        scaled = (alpha * logits / temperature).expand(n, -1)
        gumbel = -torch.log(
            -torch.log(
                torch.rand_like(scaled).clamp_min_(torch.finfo(scaled.dtype).tiny)
            )
        )
        draws = torch.argmax(scaled + gumbel, dim=-1)
        freq = torch.bincount(draws, minlength=50).float() / n
        probs = torch.softmax(alpha * logits[0] / temperature, dim=-1)
        self.assertLess((freq - probs).abs().max().item(), 0.01)


def _tp2_worker(rank, world, port, out_path):
    """Runs in a plain subprocess (NOT multiprocessing.spawn: under
    ``python -m unittest`` a spawn child re-imports unittest's __main__, which
    re-runs test discovery).  Writes a JSON result dict to ``out_path``."""
    try:
        from sglang.srt.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )

        init_distributed_environment(
            world_size=world,
            rank=rank,
            local_rank=rank,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            backend="gloo",
        )
        initialize_model_parallel(world, 1)
        import sglang.srt.server_args as server_args_mod

        server_args_mod.set_global_server_args_for_scheduler(_StubServerArgs())
        _stub_dp_attention()
        from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead

        vocab, hid = 128, 16
        head = ParallelLMHead(vocab, hid)
        # Same deterministic FULL weight on both ranks; each holds one shard.
        torch.manual_seed(0)
        full_weight = torch.randn(vocab, hid)
        shard = vocab // world
        head.weight.data.copy_(full_weight[rank * shard : (rank + 1) * shard])

        hidden = torch.arange(5 * hid, dtype=torch.float32).reshape(5, hid) / 100.0
        out = _project(hidden, head)
        full_ref = (hidden @ full_weight.T).float()
        shard_only = (hidden @ head.weight.data.T).float()
        result = {
            "rank": rank,
            "cols": out.shape[-1],
            "full_diff": (out[..., :vocab] - full_ref).abs().max().item(),
            "shard_cols": shard_only.shape[-1],
        }
    except Exception:  # surface subprocess tracebacks in the parent assert
        import traceback

        result = {"rank": rank, "error": traceback.format_exc()}
    with open(out_path, "w") as f:
        json.dump(result, f)


class TestX0ProjectionTP2(unittest.TestCase):
    """tp=2: the processor all-gathers vocab shards; a raw matmul would not."""

    @unittest.skipUnless(
        torch.cuda.device_count() >= 2,
        "sglang's model-parallel init wants one GPU per rank "
        "(pynccl communicators are built even under a gloo world)",
    )
    def test_full_vocab_on_every_rank(self):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        port = _free_port()
        with tempfile.TemporaryDirectory() as tmp:
            out_paths = [os.path.join(tmp, f"rank{r}.json") for r in range(2)]
            procs = [
                subprocess.Popen(
                    [
                        sys.executable,
                        "-c",
                        f"import sys; sys.path.insert(0, {repo_root!r}); "
                        "from tests.test_smc_x0_projection import _tp2_worker; "
                        f"_tp2_worker({rank}, 2, {port}, {out_paths[rank]!r})",
                    ],
                )
                for rank in range(2)
            ]
            try:
                for p in procs:
                    p.wait(timeout=180)
            finally:
                for p in procs:
                    if p.poll() is None:
                        p.kill()
            results = []
            for path in out_paths:
                self.assertTrue(
                    os.path.exists(path), f"tp2 worker wrote no result: {path}"
                )
                with open(path) as f:
                    results.append(json.load(f))
        for r in results:
            self.assertNotIn("error", r, r.get("error"))
            # Full vocab from every rank, matching the unsharded matmul...
            self.assertGreaterEqual(r["cols"], 128)
            self.assertEqual(r["full_diff"], 0.0)
            # ...while the raw per-rank matmul only covers its 64-wide shard.
            self.assertEqual(r["shard_cols"], 64)


if __name__ == "__main__":
    unittest.main()
