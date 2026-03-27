import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest


REPO_ROOT = Path(__file__).resolve().parents[4]
PROBE_PATH = REPO_ROOT / "scripts" / "smc" / "smc_offline_e2e_probe.py"

_PROBE_SPEC = importlib.util.spec_from_file_location("smc_offline_e2e_probe", PROBE_PATH)
assert _PROBE_SPEC is not None and _PROBE_SPEC.loader is not None
smc_probe = importlib.util.module_from_spec(_PROBE_SPEC)
_PROBE_SPEC.loader.exec_module(smc_probe)


class SMCProbeScriptTest(unittest.TestCase):
    def _make_args(self, **overrides):
        values = dict(
            suite=smc_probe.CORRECTNESS_SUITE,
            cuda_graph_max_bs=None,
            smc_n_particles=4,
            smc_gamma=4,
            model_path="model",
            draft_model_path=None,
            smc_draft_temperature=None,
            smc_target_temperature=None,
            temperature=0.0,
            attention_backend="triton",
            disable_cuda_graph=False,
        )
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_default_smc_temperatures_follow_request_temperature(self):
        args = self._make_args(temperature=0.0)

        engine_kwargs = smc_probe._build_engine_kwargs(args, prompt_count=4)

        self.assertEqual(
            engine_kwargs["smc_draft_temperature"],
            smc_probe.SMC_MIN_TEMPERATURE,
        )
        self.assertEqual(
            engine_kwargs["smc_target_temperature"],
            smc_probe.SMC_MIN_TEMPERATURE,
        )

    def test_explicit_smc_temperatures_override_request_temperature(self):
        args = self._make_args(
            temperature=0.0,
            smc_draft_temperature=0.75,
            smc_target_temperature=0.5,
        )

        engine_kwargs = smc_probe._build_engine_kwargs(args, prompt_count=4)

        self.assertEqual(engine_kwargs["smc_draft_temperature"], 0.75)
        self.assertEqual(engine_kwargs["smc_target_temperature"], 0.5)


if __name__ == "__main__":
    unittest.main()
