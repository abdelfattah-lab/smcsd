"""Unit tests for the offline SMC probe script helpers."""

from argparse import Namespace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from unittest import TestCase


def _load_probe_module():
    repo_root = Path(__file__).resolve().parents[4]
    module_path = repo_root / "scripts" / "smc" / "smc_offline_e2e_probe.py"
    spec = spec_from_file_location("smc_offline_e2e_probe", module_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestSMCOfflineE2EProbeHelpers(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.probe = _load_probe_module()

    def test_correctness_suite_defaults_are_clean(self):
        prompts = self.probe._build_prompts(
            prompt_repeat=self.probe._default_prompt_repeat(
                self.probe.CORRECTNESS_SUITE
            ),
            annotate_prompts=self.probe._should_annotate_prompts(
                self.probe.CORRECTNESS_SUITE,
                False,
            ),
        )
        self.assertEqual(prompts, self.probe.BASE_PROMPTS)

    def test_stress_suite_defaults_repeat_and_annotate(self):
        prompts = self.probe._build_prompts(
            prompt_repeat=self.probe._default_prompt_repeat(self.probe.STRESS_SUITE),
            annotate_prompts=self.probe._should_annotate_prompts(
                self.probe.STRESS_SUITE,
                False,
            ),
        )
        self.assertEqual(len(prompts), len(self.probe.BASE_PROMPTS) * 3)
        self.assertTrue(prompts[0].startswith("[probe batch 1 item 1] "))

    def test_correctness_suite_scales_cuda_graph_batch_size(self):
        args = Namespace(
            suite=self.probe.CORRECTNESS_SUITE,
            model_path=self.probe.MODEL_PATH,
            draft_model_path=None,
            smc_n_particles=4,
            smc_gamma=4,
            smc_draft_temperature=1.25,
            cuda_graph_max_bs=None,
            attention_backend="triton",
        )
        engine_kwargs = self.probe._build_engine_kwargs(args, prompt_count=4)
        self.assertEqual(engine_kwargs["cuda_graph_max_bs"], 16)

    def test_stress_suite_keeps_small_default_cuda_graph_batch_size(self):
        args = Namespace(
            suite=self.probe.STRESS_SUITE,
            model_path=self.probe.MODEL_PATH,
            draft_model_path=None,
            smc_n_particles=4,
            smc_gamma=4,
            smc_draft_temperature=1.25,
            cuda_graph_max_bs=None,
            attention_backend="triton",
        )
        engine_kwargs = self.probe._build_engine_kwargs(args, prompt_count=12)
        self.assertEqual(engine_kwargs["cuda_graph_max_bs"], 4)
