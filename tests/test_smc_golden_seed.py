"""Golden fixed-seed regression test for the SMC engine.

Runs a tiny end-to-end generation with a fixed seed and compares output
token IDs (exact) and log Z_hat (tolerance) against a recorded golden.
Determinism holds only in the sequential event loop — the overlap loop
intentionally shifts RNG consumption — so the engine is pinned to
enable_overlap=False here.

Opt-in: the engine spin-up takes minutes and goldens are specific to the
recording hardware/software stack, so the test is skipped unless
SMC_GOLDEN=1.  To (re)record after an intentional behavior change, run with
SMC_GOLDEN=1 SMC_GOLDEN_RECORD=1 and commit the regenerated golden file.
"""

import json
import os
from pathlib import Path

import pytest

GOLDEN_PATH = Path(__file__).parent / "goldens" / "smc_golden_seed0.json"

MODEL = "meta-llama/Llama-3.2-1B-Instruct"
CONFIG = {
    "model": MODEL,
    "draft_model": MODEL,
    "n_particles": 4,
    "gamma": 4,
    "seed": 0,
    "max_new_tokens": 32,
    # One prompt only: multiple prompts arrive over ZMQ and can land in one
    # prefill batch or two depending on poll timing, which alone breaks
    # restart determinism.
    "prompts": [
        "The capital of France is",
    ],
}


@pytest.mark.skipif(
    os.environ.get("SMC_GOLDEN") != "1",
    reason="opt-in golden test (SMC_GOLDEN=1); goldens are stack-specific",
)
def test_smc_golden_seed0():
    from smcsd.engine import SMCEngine

    engine = SMCEngine(
        model_path=CONFIG["model"],
        draft_model_path=CONFIG["draft_model"],
        n_particles=CONFIG["n_particles"],
        gamma=CONFIG["gamma"],
        random_seed=CONFIG["seed"],
        enable_overlap=False,
        mem_fraction_static=0.25,
        page_size=1,
        attention_backend="triton",
        trust_remote_code=True,
    )
    try:
        outputs = engine.generate(
            CONFIG["prompts"],
            {"max_new_tokens": CONFIG["max_new_tokens"], "temperature": 0.7},
        )
    finally:
        engine.shutdown()

    got = {
        "config": CONFIG,
        "output_ids": [list(map(int, o["output_ids"])) for o in outputs],
        "log_Z_hat": [float(o.get("smc_log_Z_hat") or 0.0) for o in outputs],
    }

    if os.environ.get("SMC_GOLDEN_RECORD") == "1" or not GOLDEN_PATH.exists():
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_PATH.write_text(json.dumps(got, indent=1))
        pytest.skip(f"golden recorded at {GOLDEN_PATH}; re-run to compare")

    golden = json.loads(GOLDEN_PATH.read_text())
    assert golden["config"] == CONFIG, (
        "golden was recorded under a different config; re-record deliberately"
    )
    assert got["output_ids"] == golden["output_ids"], (
        "fixed-seed output token IDs diverged from golden"
    )
    for got_z, gold_z in zip(got["log_Z_hat"], golden["log_Z_hat"]):
        assert abs(got_z - gold_z) < 1e-3, (
            f"log Z_hat diverged: {got_z} vs golden {gold_z}"
        )
