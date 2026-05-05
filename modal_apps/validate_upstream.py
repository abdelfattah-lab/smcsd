"""Validate the smcsd `upstream` branch end-to-end on a single H100.

Runs:
  1. Imports smcsd.engine.SMCEngine + a few other scripts (catches import-time
     incompatibilities with the upgraded sglang).
  2. pytest tests/ (smoke for SMC core kernels + scheduler).

Mirrors the runtime env setup used by run_gsm8k.py:
  - CUDA 13 base image
  - torch==2.11.0 + sglang-kernel==0.4.2 installed at runtime
  - smcsd-side patches applied to the cloned source for users running on
    this branch before our 5 fixes are committed (configure_scheduler_process
    rename, scheduler init tuple unpack, external_corpus_manager,
    self_check_during_idle no-op, etc.). Once those land in origin/upstream,
    these runtime sed-patches become no-ops.

Usage:
    modal run modal_apps/validate_upstream.py
"""

from __future__ import annotations

import modal

APP_NAME = "smcsd-validate-upstream"
SMCSD_DIR = "/root/smcsd"
HF_HOME = "/root/.cache/huggingface"
SMCSD_REPO = "https://github.com/abdelfattah-lab/smcsd.git"
SMCSD_BRANCH = "upstream"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git", "build-essential", "ca-certificates", "curl", "protobuf-compiler")
    .run_commands(
        # Upstream sglang has a Rust gRPC extension (sglang.srt.grpc._core)
        # built via setuptools-rust during pip install -e — needs cargo/rustc.
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable",
        'git config --global url."https://github.com/".insteadOf "git@github.com:"',
        f"git clone --branch {SMCSD_BRANCH} --recurse-submodules {SMCSD_REPO} {SMCSD_DIR}",
        f"cd {SMCSD_DIR} && pip install --upgrade pip wheel",
        f'export PATH=$HOME/.cargo/bin:$PATH && cd {SMCSD_DIR} && pip install -e 3rdparty/sglang/python',
        f"cd {SMCSD_DIR} && pip install -e .",
        "pip install datasets transformers numpy pytest",
    )
    .run_commands(
        "apt-get update && apt-get install -y --no-install-recommends libnuma1 && rm -rf /var/lib/apt/lists/*"
    )
    .env({"HF_HOME": HF_HOME, "PYTHONUNBUFFERED": "1"})
)

app = modal.App(APP_NAME, image=image)
hf_cache = modal.Volume.from_name("smcsd-hf-cache", create_if_missing=True)


@app.function(
    gpu="H100",
    timeout=60 * 30,
    secrets=[modal.Secret.from_name("hf-llama-token")],
    volumes={HF_HOME: hf_cache},
)
def validate() -> None:
    import os
    import subprocess
    import sys

    # ── HF token normalize ─────────────────────────────────────────────────
    for k in (
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
        "HUGGINGFACE_TOKEN",
        "HF_HUB_TOKEN",
    ):
        token = os.environ.get(k)
        if token:
            os.environ["HF_TOKEN"] = token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = token
            break

    # ── smcsd-side patches that are not yet pushed to origin/upstream ──────
    # These are no-ops once the corresponding commits land. Idempotent.
    sched_path = f"{SMCSD_DIR}/smcsd/core/scheduler.py"
    src = open(sched_path).read()
    edits = []
    if "configure_scheduler_process" not in src:
        src = src.replace(
            "from sglang.srt.managers.scheduler import Scheduler, configure_scheduler\n",
            "from sglang.srt.managers.scheduler import Scheduler, configure_scheduler_process\n",
        )
        old_call = (
            "    dp_rank = configure_scheduler(\n"
            "        server_args, tp_rank, attn_cp_rank, moe_dp_rank, moe_ep_rank, pp_rank, dp_rank\n"
            "    )\n"
            "\n"
            "    kill_itself_when_parent_died()\n"
        )
        new_call = (
            "    dp_rank = configure_scheduler_process(\n"
            "        server_args, gpu_id, tp_rank, attn_cp_rank, moe_dp_rank, moe_ep_rank, pp_rank, dp_rank\n"
            "    )\n"
        )
        src = src.replace(old_call, new_call)
        edits.append("configure_scheduler_process rename + gpu_id arg")
    if "self.external_corpus_manager = None" not in src:
        src = src.replace(
            "    def maybe_init_draft_worker(self):\n        from smcsd.core.worker import SMCWorker\n",
            "    def maybe_init_draft_worker(self):\n"
            "        self.external_corpus_manager = None\n"
            "        from smcsd.core.worker import SMCWorker\n",
        )
        edits.append("external_corpus_manager=None in maybe_init_draft_worker")
    if "self.self_check_during_idle()" in src:
        src = src.replace(
            "self.self_check_during_idle()",
            "pass  # self_check_during_idle removed upstream",
        )
        edits.append("dropped self_check_during_idle call")
    open(sched_path, "w").write(src)

    eng_path = f"{SMCSD_DIR}/smcsd/engine.py"
    esrc = open(eng_path).read()
    old_eng = (
        "        self._scheduler_init_result = Engine._launch_scheduler_processes(\n"
        "            server_args, port_args, run_smc_scheduler_process\n"
        "        )\n"
    )
    new_eng = (
        "        self._scheduler_init_result, _ = Engine._launch_scheduler_processes(\n"
        "            server_args, port_args, run_smc_scheduler_process\n"
        "        )\n"
    )
    if old_eng in esrc:
        open(eng_path, "w").write(esrc.replace(old_eng, new_eng))
        edits.append("scheduler init tuple unpack in engine.py")

    print(f"[validate] runtime smcsd patches applied: {edits or 'none (already in source)'}", flush=True)

    # ── Test 1: smcsd module import graph ──────────────────────────────────
    print("\n[validate] Test 1: import smcsd.engine + scheduler/kernels", flush=True)
    rc = subprocess.run(
        [
            sys.executable,
            "-c",
            "from smcsd.engine import SMCEngine; "
            "from smcsd.core import scheduler as _; "
            "from smcsd.core.kernels.fused_collect import batched_collect_fused; "
            "from smcsd.core.kernels.fused_resample_kv import batched_resample_kv; "
            "print('IMPORT_OK')",
        ],
        cwd=SMCSD_DIR,
    )
    if rc.returncode != 0:
        print(f"[validate] IMPORT FAIL (rc={rc.returncode})", flush=True)
        sys.exit(1)
    print("[validate] IMPORT OK", flush=True)

    # ── Test 2: pytest tests/ ──────────────────────────────────────────────
    print("\n[validate] Test 2: pytest tests/ -v", flush=True)
    rc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short", "-x"],
        cwd=SMCSD_DIR,
    )
    print(f"[validate] pytest rc={rc.returncode}", flush=True)


@app.local_entrypoint()
def main():
    validate.remote()
