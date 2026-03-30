from __future__ import annotations

import json
import os
import time

_SMC_DIAG_PATH_ENV = "SGLANG_SMC_DIAG_PATH"
_SMC_PROBE_RECORD_PATH_ENV = "SGLANG_SMC_PROBE_RECORD_PATH"


def _append_jsonl_record(env_var_name: str, record: dict) -> None:
    record_path = os.environ.get(env_var_name)
    if not record_path:
        return
    payload = dict(record)
    payload["pid"] = os.getpid()
    payload["timestamp_ns"] = time.perf_counter_ns()
    with open(record_path, "a", encoding="utf-8") as fout:
        fout.write(json.dumps(payload, sort_keys=True) + "\n")


def append_smc_diag_record(record: dict) -> None:
    _append_jsonl_record(_SMC_DIAG_PATH_ENV, record)


def append_smc_probe_record(record: dict) -> None:
    _append_jsonl_record(_SMC_PROBE_RECORD_PATH_ENV, record)
