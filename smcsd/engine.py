"""Minimal standalone engine for SMC (Sequential Monte Carlo) offline inference.

This engine bypasses TokenizerManager and DetokenizerManager, communicating
directly with the Scheduler subprocess via ZMQ.  It exposes a single
``generate()`` API suitable for offline / batch inference.
"""

import atexit
import logging
import os
import time
import uuid
from collections import deque
from typing import Deque, Dict, List, Optional, Type, TypeVar, Union

import zmq
from transformers import AutoTokenizer

from sglang.srt.entrypoints.engine import Engine, _set_envs_and_config
from sglang.srt.managers.io_struct import (
    AbortReq,
    BatchTokenIDOutput,
    ProfileReq,
    ProfileReqOutput,
    ProfileReqType,
    RpcReqInput,
    RpcReqOutput,
    TokenizedGenerateReqInput,
)
from smcsd.core.scheduler import run_smc_scheduler_process
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import configure_logger, get_bool_env_var, kill_process_tree
from sglang.srt.utils.network import get_zmq_socket

logger = logging.getLogger(__name__)
T = TypeVar("T")


class SMCEngine:
    """Lightweight engine for offline SMC inference.

    Bypasses TokenizerManager / DetokenizerManager entirely.  Tokenization
    and detokenization happen in-process; communication with the Scheduler
    subprocess is via two ZMQ sockets (PUSH to send requests, PULL to
    receive results).

    Usage::

        engine = SMCEngine(
            model_path="meta-llama/Llama-3.1-8B-Instruct",
            draft_model_path="meta-llama/Llama-3.2-1B-Instruct",
        )
        result = engine.generate("What is 2+2?")
        print(result["text"])
        engine.shutdown()
    """

    def __init__(
        self,
        model_path: str,
        draft_model_path: str,
        *,
        # SMC hyper-parameters
        n_particles: int = 4,
        gamma: int = 4,
        draft_temperature: float = 0.7,
        target_temperature: float = 1.0,
        resample_threshold: float = 0.5,
        resample_method: str = "systematic",
        # Hardware
        tp_size: int = 1,
        base_gpu_id: int = 0,
        # Extra ServerArgs overrides
        **kwargs,
    ):
        # -- 1. Build ServerArgs --
        # Each SMC group needs N+1 Req slots (1 parent + N particles co-exist
        # briefly during materialize).  Expand max_running_requests upfront so
        # the core req_to_token_pool is sized correctly; SMCScheduler backs
        # out the user-facing concurrency from the expanded value.
        user_max = kwargs.pop("max_running_requests", None)
        if user_max is not None:
            expanded = user_max * (n_particles + 1)
            logger.info(
                "SMCEngine: scaling max_running_requests %d -> %d "
                "(= %d * (N+1=%d)) for per-particle Reqs.",
                user_max, expanded, user_max, n_particles + 1,
            )
        else:
            expanded = None

        forced = dict(
            model_path=model_path,
            speculative_algorithm="SMC",
            speculative_draft_model_path=draft_model_path,
            skip_tokenizer_init=True,  # routes results back on tokenizer_ipc
            disable_overlap_schedule=True,
            disable_radix_cache=True,
            smc_n_particles=n_particles,
            smc_gamma=gamma,
            smc_draft_temperature=draft_temperature,
            smc_target_temperature=target_temperature,
            smc_resample_threshold=resample_threshold,
            smc_resample_method=resample_method,
            tp_size=tp_size,
            base_gpu_id=base_gpu_id,
        )
        if expanded is not None:
            forced["max_running_requests"] = expanded
        # User kwargs can override anything not in `forced`
        merged = {**kwargs, **forced}
        if "log_level" not in merged:
            merged["log_level"] = "error"

        server_args = ServerArgs(**merged)
        self.server_args = server_args

        # -- 2. Global env / config (mirrors Engine._launch_subprocesses) --
        configure_logger(server_args)
        _set_envs_and_config(server_args)
        server_args.check_server_args()

        # -- 3. Load tokenizer in-process --
        self.tokenizer = AutoTokenizer.from_pretrained(
            server_args.tokenizer_path or model_path,
            trust_remote_code=server_args.trust_remote_code,
        )

        # -- 4. Allocate IPC channels --
        port_args = PortArgs.init_new(server_args)
        self.port_args = port_args

        # -- 5. Set up ZMQ sockets (bind side -- scheduler connects) --
        self._zmq_context = zmq.Context(2)
        # We send tokenized requests to the scheduler on this socket
        self.send_to_scheduler = get_zmq_socket(
            self._zmq_context, zmq.PUSH, port_args.scheduler_input_ipc_name, True
        )
        # We receive results from the scheduler on this socket.
        # With skip_tokenizer_init=True the scheduler sends both
        # send_to_tokenizer AND send_to_detokenizer to tokenizer_ipc_name.
        self.recv_from_scheduler = get_zmq_socket(
            self._zmq_context, zmq.PULL, port_args.tokenizer_ipc_name, True
        )
        # RPC socket (needed for scheduler init handshake, weight updates, etc.)
        self.send_to_rpc = get_zmq_socket(
            self._zmq_context, zmq.DEALER, port_args.rpc_ipc_name, True
        )
        # Note: With skip_tokenizer_init=True the scheduler connects to
        # tokenizer_ipc_name for both send_to_tokenizer and send_to_detokenizer,
        # so detokenizer_ipc_name is unused and needs no bind.

        # -- 6. Launch scheduler subprocess(es) --
        # Upstream changed the return to a 2-tuple (SchedulerInitResult, procs);
        # procs is None for RayEngine.
        self._scheduler_init_result, self._scheduler_procs = (
            Engine._launch_scheduler_processes(
                server_args, port_args, run_smc_scheduler_process
            )
        )
        self._scheduler_init_result.wait_for_ready()
        logger.info("SMCEngine: Scheduler is ready.")

        # -- 7. Lifecycle --
        self._shutdown_called = False
        self._pending_scheduler_outputs: Deque[object] = deque()
        atexit.register(self.shutdown)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        sampling_params: Optional[Union[Dict, List[Dict]]] = None,
        input_ids: Optional[Union[List[int], List[List[int]]]] = None,
    ) -> Union[Dict, List[Dict]]:
        """Run SMC inference on one or more prompts.

        Args:
            prompt: A single string or list of strings.
            sampling_params: Sampling config dict(s) passed to ``SamplingParams``.
            input_ids: Pre-tokenized input(s).  Mutually exclusive with *prompt*.

        Returns:
            A dict (single prompt) or list of dicts with keys:
            ``text``, ``prompt_tokens``, ``completion_tokens``, ``output_ids``.
        """
        # -- Normalise inputs to lists --
        is_single = isinstance(prompt, str) or (
            prompt is None
            and input_ids is not None
            and len(input_ids) > 0
            and isinstance(input_ids[0], int)
        )

        if prompt is not None:
            prompts: List[str] = [prompt] if isinstance(prompt, str) else list(prompt)
            ids_list: List[List[int]] = [
                self.tokenizer.encode(p) for p in prompts
            ]
        elif input_ids is not None:
            if is_single:
                ids_list = [input_ids]
            else:
                ids_list = list(input_ids)
            prompts = [self.tokenizer.decode(ids) for ids in ids_list]
        else:
            raise ValueError("Either prompt or input_ids must be provided.")

        if sampling_params is None:
            sampling_params_list = [{}] * len(prompts)
        elif isinstance(sampling_params, dict):
            sampling_params_list = [sampling_params] * len(prompts)
        else:
            sampling_params_list = list(sampling_params)

        # -- Build and send requests --
        rids: List[str] = []
        for text, ids, sp_dict in zip(prompts, ids_list, sampling_params_list):
            rid = uuid.uuid4().hex
            rids.append(rid)

            sp = SamplingParams(**sp_dict) if isinstance(sp_dict, dict) else sp_dict
            sp.normalize(self.tokenizer)

            req = TokenizedGenerateReqInput(
                rid=rid,
                input_text=text,
                input_ids=ids,
                mm_inputs=None,
                sampling_params=sp,
                return_logprob=False,
                logprob_start_len=0,
                top_logprobs_num=0,
                token_ids_logprob=[],
                stream=False,
            )
            self.send_to_scheduler.send_pyobj(req)

        # -- Collect results --
        # Results arrive as BatchTokenIDOutput (possibly multiple per request,
        # since the scheduler emits incremental slices every ~50 tokens even
        # with stream=False).  We accumulate output_ids per rid until we see
        # a non-None finish reason.
        pending = set(rids)
        results: Dict[str, Dict] = {}

        while pending:
            msg = self._recv_scheduler_output()

            # Handle aborted requests to avoid infinite hang
            if isinstance(msg, AbortReq):
                rid = msg.rid
                if rid in pending:
                    pending.discard(rid)
                    if rid not in results:
                        results[rid] = {
                            "output_ids": [],
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "aborted": True,
                        }
                    else:
                        results[rid]["aborted"] = True
                continue

            if not isinstance(msg, BatchTokenIDOutput):
                # Other control messages -- safe to skip
                continue

            for i, rid in enumerate(msg.rids):
                if rid not in pending:
                    continue

                if rid not in results:
                    results[rid] = {
                        "output_ids": [],
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                    }

                entry = results[rid]

                # output_ids is populated when skip_tokenizer_init is True.
                # Each message contains an incremental slice (new tokens since
                # the last send_token_offset), so we extend rather than replace.
                if msg.output_ids is not None and msg.output_ids[i] is not None:
                    entry["output_ids"].extend(msg.output_ids[i])

                if msg.prompt_tokens:
                    entry["prompt_tokens"] = msg.prompt_tokens[i]
                if msg.completion_tokens:
                    entry["completion_tokens"] = msg.completion_tokens[i]

                # Check if this request is finished
                if (
                    msg.finished_reasons
                    and msg.finished_reasons[i] is not None
                ):
                    pending.discard(rid)

        # -- Detokenize and return --
        outputs: List[Dict] = []
        for rid in rids:
            entry = results[rid]
            out_ids = entry["output_ids"]
            text = self.tokenizer.decode(out_ids, skip_special_tokens=True)
            outputs.append(
                {
                    "text": text,
                    "output_ids": out_ids,
                    "prompt_tokens": entry["prompt_tokens"],
                    "completion_tokens": entry["completion_tokens"],
                }
            )

        return outputs[0] if is_single else outputs

    def start_profile(
        self,
        output_dir: Optional[str] = None,
        start_step: Optional[int] = None,
        num_steps: Optional[int] = None,
        activities: Optional[List[str]] = None,
        with_stack: Optional[bool] = None,
        record_shapes: Optional[bool] = None,
        profile_by_stage: bool = False,
        merge_profiles: bool = False,
        profile_prefix: Optional[str] = None,
        profile_stages: Optional[List[str]] = None,
    ) -> ProfileReqOutput:
        env_with_stack: bool = get_bool_env_var("SGLANG_PROFILE_WITH_STACK", "true")
        with_stack = False if with_stack is False or env_with_stack is False else True
        env_record_shapes: bool = get_bool_env_var(
            "SGLANG_PROFILE_RECORD_SHAPES", "true"
        )
        record_shapes = (record_shapes is not False) and env_record_shapes

        req = ProfileReq(
            type=ProfileReqType.START_PROFILE,
            output_dir=output_dir,
            start_step=start_step,
            num_steps=num_steps,
            activities=activities,
            with_stack=with_stack,
            record_shapes=record_shapes,
            profile_by_stage=profile_by_stage,
            profile_id=str(time.time()),
            merge_profiles=merge_profiles,
            profile_prefix=profile_prefix,
            profile_stages=profile_stages,
        )
        return self._execute_profile(req)

    def stop_profile(self) -> ProfileReqOutput:
        return self._execute_profile(ProfileReq(type=ProfileReqType.STOP_PROFILE))

    def shutdown(self):
        """Kill all scheduler subprocesses."""
        if self._shutdown_called:
            return
        self._shutdown_called = True
        kill_process_tree(os.getpid(), include_parent=False)

    def collective_rpc(self, method: str, **kwargs) -> str:
        req = RpcReqInput(method=method, parameters=kwargs or None)
        self.send_to_rpc.send_pyobj(req)
        recv_req = self.send_to_rpc.recv_pyobj(zmq.BLOCKY)
        assert isinstance(recv_req, RpcReqOutput)
        assert recv_req.success, recv_req.message
        return recv_req.message

    def _execute_profile(self, req: ProfileReq) -> ProfileReqOutput:
        self.send_to_scheduler.send_pyobj(req)
        result = self._recv_expected_scheduler_output(ProfileReqOutput)
        if not result.success:
            raise RuntimeError(result.message)
        return result

    def _recv_scheduler_output(self) -> object:
        if self._pending_scheduler_outputs:
            return self._pending_scheduler_outputs.popleft()
        return self.recv_from_scheduler.recv_pyobj()

    def _recv_expected_scheduler_output(self, expected_type: Type[T]) -> T:
        retained: Deque[object] = deque()
        while self._pending_scheduler_outputs:
            recv_obj = self._pending_scheduler_outputs.popleft()
            if isinstance(recv_obj, expected_type):
                self._pending_scheduler_outputs.extendleft(reversed(retained))
                return recv_obj
            retained.append(recv_obj)
        self._pending_scheduler_outputs.extendleft(reversed(retained))

        while True:
            recv_obj = self.recv_from_scheduler.recv_pyobj()
            if isinstance(recv_obj, expected_type):
                return recv_obj
            self._pending_scheduler_outputs.append(recv_obj)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False
