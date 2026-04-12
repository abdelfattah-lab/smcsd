import concurrent.futures
import os
import unittest

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_STANDALONE,
    DEFAULT_TARGET_MODEL_STANDALONE,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=120, suite="stage-b-test-small-1-gpu")

SMC_SERVER_ARGS = [
    "--trust-remote-code",
    "--attention-backend",
    "triton",
    "--cuda-graph-max-bs",
    "8",
    "--speculative-algorithm",
    "SMC",
    "--speculative-draft-model-path",
    DEFAULT_DRAFT_MODEL_STANDALONE,
    "--smc-n-particles",
    "4",
    "--smc-gamma",
    "4",
    "--page-size",
    "1",
    "--mem-fraction-static",
    "0.7",
]


class TestSMCOverlapSpeculativeDecoding(CustomTestCase):
    base_url = DEFAULT_URL_FOR_TEST

    @classmethod
    def setUpClass(cls):
        envs.SGLANG_ENABLE_SPEC_V2.set(True)
        cls.process = popen_launch_server(
            DEFAULT_TARGET_MODEL_STANDALONE,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=SMC_SERVER_ARGS,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        if "SGLANG_ENABLE_SPEC_V2" in os.environ:
            envs.SGLANG_ENABLE_SPEC_V2.set(False)

    def test_server_info_reports_overlap(self):
        server_info = requests.get(self.base_url + "/server_info").json()
        self.assertEqual(server_info["speculative_algorithm"], "SMC")
        self.assertFalse(server_info["disable_overlap_schedule"])
        self.assertEqual(server_info["smc_n_particles"], 4)
        self.assertEqual(server_info["smc_gamma"], 4)
        self.assertIsNone(self.process.poll())

    def test_generate_smoke(self):
        prompts = [
            "The capital of France is",
            "Write one sentence about speculative decoding.",
            "List two prime numbers:",
        ]
        for prompt in prompts:
            response = requests.post(
                self.base_url + "/generate",
                json={
                    "text": prompt,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 16,
                        "ignore_eos": True,
                    },
                },
            )
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertIn("text", payload)
            self.assertTrue(payload["text"])
        self.assertIsNone(self.process.poll())

    def test_concurrent_generate_smoke(self):
        prompts = [
            "Name a mammal that lives in the ocean.",
            "Complete: The quick brown fox",
            "What is 2 plus 2?",
            "Give one color name.",
        ]

        def send(prompt: str):
            response = requests.post(
                self.base_url + "/generate",
                json={
                    "text": prompt,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 12,
                        "ignore_eos": True,
                    },
                },
            )
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertIn("text", payload)
            self.assertTrue(payload["text"])

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as pool:
            futures = [pool.submit(send, prompt) for prompt in prompts]
            for future in concurrent.futures.as_completed(futures):
                future.result()

        self.assertIsNone(self.process.poll())


if __name__ == "__main__":
    unittest.main()
