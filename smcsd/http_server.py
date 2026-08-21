"""HTTP serving for SMC (Sequential Monte Carlo) speculative decoding.

The standalone ``SMCEngine`` (``smcsd/engine.py``) is offline-only: it bypasses
the TokenizerManager/DetokenizerManager and drains results from a single ZMQ
socket in a blocking loop, so it cannot multiplex concurrent online requests.

This module bridges SMC to online serving by reusing sglang's *standard* serving
stack (``http_server`` + ``TokenizerManager`` + ``DetokenizerManager`` + FastAPI)
and only swapping in the SMC scheduler process. sglang's ``launch_server`` already
accepts the scheduler-run callable as a parameter, so we pass
``run_smc_scheduler_process`` and let the normal stack handle tokenization,
per-request concurrency multiplexing, and the HTTP API. No SMC source changes.

Three pieces of SMC-specific glue (everything else is stock sglang):

1. Inject ``run_smc_scheduler_process`` -- the base scheduler cannot build an SMC
   worker (``SpeculativeAlgorithm.create_worker`` has no SMC branch; the SMC
   worker wiring lives only in ``SMCScheduler``).
2. Replicate ``SMCEngine``'s ``max_running_requests *= (n_particles + 1)``
   expansion -- each SMC group briefly co-allocates N+1 Reqs (1 parent + N
   particles), and that math lives only in ``SMCEngine.__init__``.
3. ``disable_radix_cache=True`` for parity with ``SMCEngine`` (not auto-forced
   for SMC) and ``page_size=1`` (SMC requires it). ``disable_overlap_schedule``,
   ``speculative_num_steps`` and ``speculative_num_draft_tokens`` are already
   auto-forced by ServerArgs for ``speculative_algorithm="SMC"``.

Unlike ``SMCEngine`` we keep ``skip_tokenizer_init=False`` (the default), so the
TokenizerManager/DetokenizerManager are wired up -- that is what gives us the
async, concurrent HTTP front end.

Usage::

    python -m smcsd.http_server \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --draft-model meta-llama/Llama-3.2-1B-Instruct \
        -N 8 -g 8 --port 30000 --trust-remote-code

Then query the standard sglang endpoints, e.g. ``POST /generate`` or
``/v1/completions``.
"""

import argparse
import json
import logging
from typing import Optional

from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import ServerArgs

from smcsd.core.scheduler import run_smc_scheduler_process

logger = logging.getLogger(__name__)

# SMC runs two model runners (target + draft), each sizing its own KV-cache pool
# from mem_fraction_static, so the usable fraction per runner is ~halved. This is
# the dual-runner-safe default the offline SMC scripts use; override per model size.
DEFAULT_MEM_FRACTION_STATIC = 0.4


def build_smc_server_args(
    model_path: str,
    draft_model_path: str,
    *,
    n_particles: int = 4,
    gamma: int = 4,
    draft_temperature: float = 0.7,
    target_temperature: float = 1.0,
    resample_threshold: float = 0.5,
    resample_method: str = "systematic",
    max_running_requests: Optional[int] = None,
    **kwargs,
) -> ServerArgs:
    """Build ``ServerArgs`` for an online SMC server (mirrors ``SMCEngine.__init__``).

    ``max_running_requests`` is the user-facing number of concurrent SMC *groups*;
    it is expanded by ``(n_particles + 1)`` so the core ``req_to_token_pool`` is
    sized correctly, exactly as ``SMCEngine`` does. ``SMCScheduler`` backs the
    user-facing group count out of the expanded value.
    """
    if max_running_requests is not None:
        expanded = max_running_requests * (n_particles + 1)
        logger.info(
            "SMC server: scaling max_running_requests %d -> %d (= %d * (N+1=%d))",
            max_running_requests,
            expanded,
            max_running_requests,
            n_particles + 1,
        )
        kwargs["max_running_requests"] = expanded

    # SMC requires the triton or fa3 attention backend; default to triton.
    kwargs.setdefault("attention_backend", "triton")
    if kwargs["attention_backend"] == "triton":
        # B200-swept decode default (neutral short-ctx, +7% at 4k).
        kwargs.setdefault("triton_attention_num_kv_splits", 16)

    # Decode perf optimizations default ON via the worker/scheduler-side
    # smc_* attr defaults (getattr(..., True)); SMC_* env vars remain as
    # kill switches.

    # SMC loads two independent model runners (target + draft) and each sizes
    # its own KV-cache pool from mem_fraction_static, so the fraction is counted
    # roughly twice. sglang's ~0.88 default makes the target grab almost all VRAM,
    # then the draft KV-pool init OOMs on a single GPU. Default to the same
    # dual-runner-safe fraction the offline SMC scripts use; an explicit
    # --mem-fraction-static still overrides it.
    kwargs.setdefault("mem_fraction_static", DEFAULT_MEM_FRACTION_STATIC)

    forced = dict(
        model_path=model_path,
        speculative_algorithm="SMC",
        speculative_draft_model_path=draft_model_path,
        disable_overlap_schedule=True,  # also auto-forced for SMC; explicit for parity
        disable_radix_cache=True,  # parity with SMCEngine; not auto-forced for SMC
        page_size=1,  # SMC requires page_size == 1
        smc_n_particles=n_particles,
        smc_gamma=gamma,
        smc_draft_temperature=draft_temperature,
        smc_target_temperature=target_temperature,
        smc_resample_threshold=resample_threshold,
        smc_resample_method=resample_method,
    )
    # skip_tokenizer_init stays False (default) so the TokenizerManager and
    # DetokenizerManager handle concurrent HTTP requests.
    merged = {**kwargs, **forced}
    return ServerArgs(**merged)


def launch_smc_http_server(server_args: ServerArgs, launch_callback=None) -> None:
    """Launch the full sglang HTTP server driving the SMC scheduler.

    Blocks (runs uvicorn) until the process is terminated.
    """
    launch_server(
        server_args,
        run_scheduler_process_func=run_smc_scheduler_process,
        launch_callback=launch_callback,
    )


def _json_object(value: str) -> dict:
    """Parse a JSON object for CLI options that feed chat-template kwargs."""
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("expected a JSON object")
    return parsed


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch an HTTP server for SMC speculative decoding.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", "--model-path", dest="model_path", required=True)
    parser.add_argument(
        "--draft-model", "--draft-model-path", dest="draft_model_path", required=True
    )
    parser.add_argument("--particles", "-N", type=int, default=4)
    parser.add_argument("--gamma", "-g", type=int, default=4)
    parser.add_argument("--draft-temperature", type=float, default=0.7)
    parser.add_argument("--target-temperature", type=float, default=1.0)
    parser.add_argument("--resample-threshold", type=float, default=0.5)
    parser.add_argument(
        "--resample-method", default="systematic", choices=["systematic", "multinomial"]
    )
    parser.add_argument(
        "--max-running-requests",
        type=int,
        default=None,
        help="User-facing concurrent SMC groups; expanded by (N+1) internally.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument(
        "--served-model-name",
        default=None,
        help="Model id advertised by /v1/models and accepted by OpenAI clients.",
    )
    parser.add_argument(
        "--chat-template",
        default=None,
        help="Builtin chat-template name or path used by /v1/chat/completions.",
    )
    parser.add_argument(
        "--hf-chat-template-name",
        default=None,
        help="Named Hugging Face chat template (for tokenizers with several).",
    )
    parser.add_argument(
        "--tool-call-parser",
        default=None,
        help="SGLang tool-call parser name; 'auto' detects from the template.",
    )
    parser.add_argument(
        "--reasoning-parser",
        default=None,
        help="SGLang reasoning parser name; 'auto' detects from the template.",
    )
    parser.add_argument(
        "--default-chat-template-kwargs",
        type=_json_object,
        default=None,
        metavar="JSON",
        help="Defaults applied to chat requests, e.g. '{\"enable_thinking\": false}'.",
    )
    parser.add_argument(
        "--mem-fraction-static",
        type=float,
        default=None,
        help=f"GPU memory fraction. Defaults to {DEFAULT_MEM_FRACTION_STATIC} "
        "(SMC runs target+draft runners, each with its own KV pool).",
    )
    parser.add_argument(
        "--attention-backend", default="triton", choices=["triton", "fa3"]
    )
    parser.add_argument("--cuda-graph-max-bs", type=int, default=None)
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Seed used by SGLang workers; set it for reproducible experiment runs.",
    )
    parser.add_argument(
        "--enable-metrics",
        action="store_true",
        help="Expose SGLang Prometheus serving metrics at /metrics.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--skip-server-warmup", action="store_true")
    parser.add_argument(
        "--disable-cuda-graph",
        action="store_true",
        help="Use eager execution; useful as a correctness fallback for new models.",
    )
    parser.add_argument(
        "--disable-flashinfer-autotune",
        action="store_true",
        help="Skip FlashInfer startup autotuning.",
    )
    parser.add_argument("--log-level", default="info", help="sglang server log level.")
    return parser


def server_args_from_cli(args: argparse.Namespace) -> ServerArgs:
    """Translate the dedicated SMC CLI into SGLang ``ServerArgs``."""

    extra = dict(
        host=args.host,
        port=args.port,
        tp_size=args.tp_size,
        attention_backend=args.attention_backend,
        log_level=args.log_level,
    )
    for field in (
        "served_model_name",
        "chat_template",
        "hf_chat_template_name",
        "tool_call_parser",
        "reasoning_parser",
        "default_chat_template_kwargs",
    ):
        value = getattr(args, field)
        if value is not None:
            extra[field] = value
    if args.mem_fraction_static is not None:
        extra["mem_fraction_static"] = args.mem_fraction_static
    if args.cuda_graph_max_bs is not None:
        # v0.5.17 split the ServerArgs field into per-mode knobs.
        extra["cuda_graph_max_bs_decode"] = args.cuda_graph_max_bs
    if args.random_seed is not None:
        extra["random_seed"] = args.random_seed
    if args.enable_metrics:
        extra["enable_metrics"] = True
    if args.trust_remote_code:
        extra["trust_remote_code"] = True
    if args.skip_server_warmup:
        extra["skip_server_warmup"] = True
    if args.disable_cuda_graph:
        extra["disable_cuda_graph"] = True
    if args.disable_flashinfer_autotune:
        extra["disable_flashinfer_autotune"] = True
    return build_smc_server_args(
        model_path=args.model_path,
        draft_model_path=args.draft_model_path,
        n_particles=args.particles,
        gamma=args.gamma,
        draft_temperature=args.draft_temperature,
        target_temperature=args.target_temperature,
        resample_threshold=args.resample_threshold,
        resample_method=args.resample_method,
        max_running_requests=args.max_running_requests,
        **extra,
    )


def main() -> None:
    args = build_cli_parser().parse_args()
    server_args = server_args_from_cli(args)
    launch_smc_http_server(server_args)


if __name__ == "__main__":
    main()
