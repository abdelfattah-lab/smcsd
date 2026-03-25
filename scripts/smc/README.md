# SMC Experiment Scripts

This directory is the unified home for ad hoc SMC experiment entrypoints.

Current scripts:
- `smc_offline_e2e_probe.py`: offline correctness and stress probe for overlap SMC.
- `smc_profile_engine.py`: offline `sgl.Engine(...)` profiler harness for overlap SMC.

Typical usage:

```bash
source .venv/bin/activate

python scripts/smc/smc_offline_e2e_probe.py --suite correctness --show-all-outputs
python scripts/smc/smc_offline_e2e_probe.py --suite stress

python scripts/smc/smc_profile_engine.py --output-dir /tmp/sglang-smc-profile
python scripts/smc/smc_profile_engine.py --profile-v2 --decode-only
```

Notes:
- `SGLANG_ENABLE_SPEC_V2=1` is the intended default for overlap-SMC experiments.
- `FLASHINFER_WORKSPACE_BASE=/tmp/cc2869-flashinfer` is often needed on this machine when running GPU-backed experiments.
