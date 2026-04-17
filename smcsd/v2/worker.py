"""SMC Worker v2: standalone worker using SMCDecodeContext API.

Fully self-contained — no inheritance from SMCWorker.
Can replace SMCWorker entirely once the dedicated scheduler adopts v2.

Draft model performs gamma+1 autoregressive decode steps.
Score model performs one extend forward pass on the drafted tokens.
Computes logprob difference between the two models per request.
No rejection — all drafted tokens are accepted.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from typing import Optional

import torch
import torch.nn.functional as F

_SMC_EAGLE_DEBUG = os.environ.get("SMC_EAGLE_DEBUG", "0") == "1"
_SMC_EAGLE_DEBUG_MAX_CYCLES = int(os.environ.get("SMC_EAGLE_DEBUG_MAX_CYCLES", "2"))

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.eagle_info import EagleDraftInput
from smcsd.v2.info import (
    SMCDecodeContext,
    SMCDraftInputLike,
    SMCEagleDraftInputV2,
    SMCDraftInputV2,
    get_smc_draft_input_cls,
)
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker

logger = logging.getLogger(__name__)


class SMCWorkerV2(BaseSpecWorker):
    """Standalone SMC worker using v2 API draft carriers."""

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.device = server_args.device
        self._target_worker = target_worker  # score model

        self.gamma = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = self.gamma + 1
        self.smc_draft_temperature = server_args.smc_draft_temperature
        self.smc_target_temperature = max(
            float(server_args.smc_target_temperature), 1e-5
        )
        self.smc_draft_kind = server_args.smc_draft_kind
        self._draft_input_cls = get_smc_draft_input_cls(self.smc_draft_kind)

        # Share req_to_token_pool, separate KV caches
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        # Set class-level constant for KV allocation
        SMCDraftInputV2.ALLOC_LEN_PER_DECODE = self.speculative_num_draft_tokens
        SMCEagleDraftInputV2.ALLOC_LEN_PER_DECODE = self.speculative_num_draft_tokens

        # Override context length of draft model to match score model
        server_args.context_length = target_worker.model_runner.model_config.context_len

        # Do not capture cuda graph during TpModelWorker init —
        # we capture manually after the draft model is fully set up.
        # EAGLE-mode draft capture is currently unsupported: the EAGLE head
        # requires a non-None spec_info at forward time, which the base
        # cuda-graph runner does not inject.
        backup_disable_cuda_graph = server_args.disable_cuda_graph
        if self.smc_draft_kind == "eagle":
            backup_disable_cuda_graph = True
        server_args.disable_cuda_graph = True

        # Create draft TpModelWorker — fully independent, no shared lm_head/embed
        self._draft_worker = TpModelWorker(
            server_args=server_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            pp_rank=0,
            dp_rank=dp_rank,
            moe_ep_rank=moe_ep_rank,
            attn_cp_rank=attn_cp_rank,
            moe_dp_rank=moe_dp_rank,
            nccl_port=nccl_port,
            is_draft_worker=True,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            memory_pool_config=target_worker.model_runner.memory_pool_config,
        )

        self.draft_runner = self._draft_worker.model_runner
        self.score_runner = self._target_worker.model_runner

        # Multi-step draft attention backend
        from sglang.srt.speculative.draft_utils import DraftBackendFactory

        factory = DraftBackendFactory(
            server_args,
            self.draft_runner,
            topk=1,
            speculative_num_steps=self.gamma + 2,
        )
        self.draft_attn_backend = factory.create_decode_backend()

        # Restore cuda graph and capture for draft model
        server_args.disable_cuda_graph = backup_disable_cuda_graph
        self.draft_runner.server_args.disable_cuda_graph = backup_disable_cuda_graph
        if not backup_disable_cuda_graph:
            self.draft_runner.init_device_graphs()

        # SMC EAGLE runs single-chain sampling/verify: each per-step draft
        # forward sees 1 token per particle, and target verify scores gamma+1
        # linear tokens per particle with standard causal attention. The draft
        # and target attention backends inherit topk=smc_eagle_topk>1 from
        # server_args, which pushes them into tree-decode / tree-verify
        # branches that divide out_cache_loc by speculative_num_steps (=0 in
        # the draft) or require a spec_info.custom_mask (in target verify).
        # Force topk=1 on both backends so the per-step decode and target
        # verify paths use the single-chain branch.
        if self.smc_draft_kind == "eagle":
            for backend_owner in (self.draft_runner, self.score_runner):
                backend = getattr(backend_owner, "attn_backend", None)
                if backend is not None and hasattr(backend, "topk"):
                    backend.topk = 1

        self._smc_eagle_debug_cycle = 0

    # ── Properties (required by BaseSpecWorker / scheduler) ──

    @property
    def target_worker(self):
        return self._target_worker

    @property
    def draft_worker(self):
        return self._draft_worker

    @property
    def model_config(self):
        return self._target_worker.model_config

    @property
    def model_runner(self):
        return self._target_worker.model_runner

    def clear_cache_pool(self):
        pass

    def materialize_smc_parent_draft_prefix(self, req) -> None:
        """No-op: _forward_extend already prefills both models."""
        pass

    # ── Main entry point ──

    def forward_batch_generation(self, batch):
        if isinstance(batch, ScheduleBatch):
            batch = batch.get_model_worker_batch()

        if _SMC_EAGLE_DEBUG:
            print(
                f"[SMC_EAGLE_TRACE] forward_batch_generation "
                f"mode={batch.forward_mode} "
                f"is_extend_in_batch={batch.is_extend_in_batch} "
                f"bs={len(batch.seq_lens) if batch.seq_lens is not None else 'n/a'} "
                f"kind={self.smc_draft_kind}",
                flush=True,
            )

        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            return self._forward_extend(batch)
        else:
            return self._forward_decode(batch)

    # ── EXTEND (prefill) ──

    def _forward_extend(self, batch: ModelWorkerBatch):
        if self._draft_input_cls is SMCEagleDraftInputV2:
            return self._forward_extend_eagle(batch)

        # Score model prefill
        score_result = self._target_worker.forward_batch_generation(batch)

        # Draft model prefill — samples the first token (x0)
        draft_batch = self._make_clean_batch(batch)
        draft_result = self._draft_worker.forward_batch_generation(draft_batch)

        # Use draft model's sampled token as verified_id
        bs = len(batch.seq_lens)
        score_result.next_token_ids = draft_result.next_token_ids

        # x0 KV is NOT written during prefill — first decode writes it.
        score_result.next_draft_input = self._draft_input_cls(
            verified_id=draft_result.next_token_ids,
            num_tokens_per_req=self.speculative_num_draft_tokens,
        )
        score_result.accept_lens = torch.zeros(
            bs, dtype=torch.int32, device=self.device
        )
        return score_result

    def _forward_extend_eagle(self, batch: ModelWorkerBatch):
        if _SMC_EAGLE_DEBUG:
            print(
                f"[SMC_EAGLE_TRACE] _forward_extend_eagle ENTER "
                f"bs={len(batch.seq_lens)} "
                f"seq_lens={batch.seq_lens[:len(batch.seq_lens)].tolist()}",
                flush=True,
            )
        target_batch = dataclasses.replace(
            batch, capture_hidden_mode=CaptureHiddenMode.FULL
        )
        score_result = self._target_worker.forward_batch_generation(target_batch)

        target_hidden_states = score_result.logits_output.hidden_states
        if target_hidden_states is None:
            raise RuntimeError(
                "SMC EAGLE prefill requires target hidden states, but the target "
                "prefill result did not return any."
            )

        eagle_logits_output = self._run_eagle_prefill_forward(
            batch,
            target_hidden_states=target_hidden_states,
            next_token_ids=score_result.next_token_ids,
            mm_input_embeds=score_result.logits_output.mm_input_embeds,
        )
        score_result.next_draft_input = self._build_eagle_prefill_next_draft_input(
            verified_id=score_result.next_token_ids,
            logits_output=eagle_logits_output,
        )
        score_result.accept_lens = torch.zeros(
            len(batch.seq_lens), dtype=torch.int32, device=self.device
        )
        if _SMC_EAGLE_DEBUG:
            print(
                f"[SMC_EAGLE_TRACE] _forward_extend_eagle EXIT "
                f"next_token_ids={score_result.next_token_ids.tolist()} "
                f"next_draft_input.type={type(score_result.next_draft_input).__name__} "
                f"accept_lens={score_result.accept_lens.tolist()}",
                flush=True,
            )
        return score_result

    # ── DECODE ──

    def _forward_decode(self, batch: ModelWorkerBatch):
        if batch.forward_mode.is_idle():
            return self._forward_idle(batch)

        if self._draft_input_cls is SMCEagleDraftInputV2:
            return self._forward_decode_eagle(batch)

        current_stream = torch.get_device_module(self.device).current_stream()
        if batch.req_pool_indices is not None:
            batch.req_pool_indices.record_stream(current_stream)

        draft_input: SMCDraftInputLike = batch.spec_info
        ctx: SMCDecodeContext = draft_input.decode_ctx

        if draft_input.verified_id is not None:
            draft_input.verified_id.record_stream(current_stream)

        bs = len(ctx.orig_seq_lens)
        gamma = self.gamma

        # ---- 1. Prepare draft ----
        draft_fb, can_cuda_graph, cache_locs, all_positions, all_seq_lens = (
            ctx.prepare_for_draft(
                draft_input.verified_id,
                self.req_to_token_pool,
                batch,
                self.draft_runner.graph_runner
                if hasattr(self.draft_runner, "graph_runner")
                else None,
                self.draft_runner,
            )
        )

        # ---- 2. Draft AR: gamma+1 decode steps ----
        use_multistep = self.draft_attn_backend is not None and not can_cuda_graph
        if use_multistep and not draft_fb.forward_mode.is_idle():
            draft_fb.spec_info = draft_input
            draft_fb.seq_lens = ctx.orig_seq_lens
            draft_fb.seq_lens_cpu = ctx.orig_seq_lens_cpu
            self.draft_attn_backend.init_forward_metadata(draft_fb)

        x0 = draft_input.verified_id
        all_tokens = [x0]
        draft_logprobs = []
        current_ids = x0

        for step in range(gamma + 1):
            draft_fb.input_ids = current_ids
            draft_fb.positions = all_positions[:, step].contiguous()
            draft_fb.out_cache_loc = cache_locs[:, step].contiguous()

            if use_multistep:
                draft_fb.attn_backend = self.draft_attn_backend.attn_backends[step]
                draft_out = self.draft_runner.forward(
                    draft_fb, skip_attn_backend_init=True
                )
            else:
                draft_fb.seq_lens = all_seq_lens[:, step].contiguous()
                draft_fb.seq_lens_sum = ctx.orig_seq_lens_sum + bs * (step + 1)
                draft_fb.seq_lens_cpu = ctx.orig_seq_lens_cpu + (step + 1)
                draft_out = self.draft_runner.forward(draft_fb)

            logits = draft_out.logits_output.next_token_logits

            scaled_logits = logits / self.smc_draft_temperature
            log_probs = torch.log_softmax(scaled_logits, dim=-1)
            if self.smc_draft_temperature > 0:
                next_token = torch.multinomial(
                    log_probs.exp(), num_samples=1
                ).squeeze(-1)
            else:
                next_token = torch.argmax(logits, dim=-1)

            if step < gamma:
                token_logprob = log_probs.gather(
                    1, next_token.unsqueeze(1)
                ).squeeze(1)
                draft_logprobs.append(token_logprob)

            all_tokens.append(next_token)
            current_ids = next_token

        draft_logprobs_stacked = torch.stack(draft_logprobs, dim=1)

        # ---- 3. Score verify ----
        verify_forward_batch, can_run_cuda_graph = ctx.prepare_for_verify(
            self.req_to_token_pool,
            batch,
            self._target_worker,
            all_tokens,
            cache_locs,
        )

        score_result = self._target_worker.forward_batch_generation(
            model_worker_batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
            skip_attn_backend_init=True,
        )

        # ---- 4. Extract score logprobs ----
        score_logits = score_result.logits_output.next_token_logits
        expected_rows = bs * (gamma + 1)
        assert score_logits.shape[0] == expected_rows, (
            f"TARGET_VERIFY logits truncated: got {score_logits.shape[0]} rows, "
            f"expected {expected_rows} (bs={bs}, gamma+1={gamma + 1}, "
            f"cuda_graph={can_run_cuda_graph})"
        )
        score_log_probs = torch.log_softmax(score_logits, dim=-1)
        score_log_probs = score_log_probs.reshape(bs, gamma + 1, -1)
        target_tokens = torch.stack(all_tokens[1 : gamma + 1], dim=1)
        score_logprobs_stacked = score_log_probs[:, :gamma, :].gather(
            2, target_tokens.unsqueeze(2)
        ).squeeze(2)

        # ---- 5. Logprob diff ----
        logprob_diff = (score_logprobs_stacked - draft_logprobs_stacked).sum(dim=1)

        # ---- 6. Bonus token ----
        bonus_logits = score_logits.reshape(bs, gamma + 1, -1)[:, -1, :]
        bonus_log_probs = torch.log_softmax(
            bonus_logits / self.smc_target_temperature, dim=-1
        )
        bonus = torch.multinomial(bonus_log_probs.exp(), num_samples=1).squeeze(-1)

        # ---- 7. Output ----
        output_token_ids = torch.stack(
            all_tokens[1 : gamma + 1] + [bonus], dim=1
        )
        next_token_ids = output_token_ids.reshape(-1)
        accept_lens = torch.full(
            (bs,), gamma + 1, dtype=torch.int32, device=self.device
        )
        next_verified_id = bonus

        next_token_ids.record_stream(current_stream)
        accept_lens.record_stream(current_stream)
        next_verified_id.record_stream(current_stream)
        logprob_diff.record_stream(current_stream)

        next_draft_input = self._draft_input_cls(
            verified_id=next_verified_id,
            logprob_diff=logprob_diff,
            num_tokens_per_req=self.speculative_num_draft_tokens,
        )

        return GenerationBatchResult(
            logits_output=score_result.logits_output,
            next_token_ids=next_token_ids,
            accept_lens=accept_lens,
            next_draft_input=next_draft_input,
            logprob_diff=logprob_diff,
            can_run_cuda_graph=can_run_cuda_graph,
        )

    def _forward_decode_eagle(self, batch: ModelWorkerBatch):
        if _SMC_EAGLE_DEBUG:
            print(
                f"[SMC_EAGLE_TRACE] _forward_decode_eagle ENTER "
                f"bs={len(batch.seq_lens)} "
                f"spec_info_type={type(batch.spec_info).__name__}",
                flush=True,
            )
        current_stream = torch.get_device_module(self.device).current_stream()
        if batch.req_pool_indices is not None:
            batch.req_pool_indices.record_stream(current_stream)

        draft_input = batch.spec_info
        if not isinstance(draft_input, SMCEagleDraftInputV2):
            raise TypeError(
                "SMC EAGLE decode requires SMCEagleDraftInputV2 on batch.spec_info."
            )

        ctx = draft_input.decode_ctx
        if ctx is None:
            raise RuntimeError("SMC EAGLE decode requires decode_ctx.")
        if draft_input.hidden_states is None:
            raise RuntimeError(
                "SMC EAGLE decode requires hidden_states in next_draft_input."
            )
        if draft_input.verified_id is None:
            raise RuntimeError(
                "SMC EAGLE decode requires verified_id in next_draft_input."
            )

        if draft_input.verified_id is not None:
            draft_input.verified_id.record_stream(current_stream)
        draft_input.hidden_states.record_stream(current_stream)
        if draft_input.topk_p is not None:
            draft_input.topk_p.record_stream(current_stream)
        if draft_input.topk_index is not None:
            draft_input.topk_index.record_stream(current_stream)

        bs = len(ctx.orig_seq_lens)
        gamma = self.gamma
        _, can_cuda_graph, cache_locs, all_positions, all_seq_lens = (
            ctx.prepare_for_draft(
                draft_input.verified_id,
                self.req_to_token_pool,
                batch,
                None,
                self.draft_runner,
            )
        )

        debug = (
            _SMC_EAGLE_DEBUG
            and self._smc_eagle_debug_cycle < _SMC_EAGLE_DEBUG_MAX_CYCLES
        )
        if debug:
            cyc = self._smc_eagle_debug_cycle
            print(f"\n[SMC_EAGLE_DBG cycle={cyc}] ==== DECODE ENTRY ====", flush=True)
            print(
                f"[SMC_EAGLE_DBG cycle={cyc}] bs={bs} gamma={gamma} "
                f"verified_id[:bs]={draft_input.verified_id[:bs].tolist()} "
                f"hidden_states.shape={tuple(draft_input.hidden_states.shape)} "
                f"hidden_states[0].norm={draft_input.hidden_states[0].norm().item():.4f} "
                f"topk_p_present={draft_input.topk_p is not None} "
                f"topk_index_present={draft_input.topk_index is not None}",
                flush=True,
            )
            print(
                f"[SMC_EAGLE_DBG cycle={cyc}] orig_seq_lens[:bs]="
                f"{ctx.orig_seq_lens[:bs].tolist()} "
                f"cache_locs.shape={tuple(cache_locs.shape)} "
                f"all_positions[0]={all_positions[0].tolist()} "
                f"all_seq_lens[0]={all_seq_lens[0].tolist()}",
                flush=True,
            )
            if draft_input.topk_p is not None:
                print(
                    f"[SMC_EAGLE_DBG cycle={cyc}] incoming topk_p[0]="
                    f"{draft_input.topk_p[0].tolist()} "
                    f"topk_index[0]={draft_input.topk_index[0].tolist()}",
                    flush=True,
                )

        all_tokens = [draft_input.verified_id]
        draft_logprobs = []
        current_hidden_states = draft_input.hidden_states
        topk_p = draft_input.topk_p
        topk_index = draft_input.topk_index
        kv_step_offset = 0

        if topk_p is None or topk_index is None:
            if debug:
                print(
                    f"[SMC_EAGLE_DBG cycle={cyc}] BOOTSTRAP run: "
                    f"input_ids={draft_input.verified_id[:bs].tolist()} "
                    f"out_cache_loc={cache_locs[:, 0][:bs].tolist()} "
                    f"position={all_positions[:, 0][:bs].tolist()} "
                    f"seq_lens={all_seq_lens[:, 0][:bs].tolist()}",
                    flush=True,
                )
            bootstrap_out = self._run_eagle_decode_step(
                batch,
                token_ids=draft_input.verified_id,
                hidden_states=current_hidden_states,
                seq_lens=all_seq_lens[:, 0].contiguous(),
                seq_lens_cpu=ctx.orig_seq_lens_cpu + 1,
                seq_lens_sum=ctx.orig_seq_lens_sum + bs,
                out_cache_loc=cache_locs[:, 0].contiguous(),
                position=all_positions[:, 0].contiguous(),
            )
            topk_p, topk_index, current_hidden_states = self._extract_eagle_state(
                bootstrap_out
            )
            kv_step_offset = 1
            if debug:
                print(
                    f"[SMC_EAGLE_DBG cycle={cyc}] BOOTSTRAP out: "
                    f"topk_p[0]={topk_p[0].tolist()} "
                    f"topk_index[0]={topk_index[0].tolist()} "
                    f"hidden_states[0].norm={current_hidden_states[0].norm().item():.4f}",
                    flush=True,
                )

        for step in range(gamma):
            if debug:
                print(
                    f"[SMC_EAGLE_DBG cycle={cyc}] --- step={step} ---",
                    flush=True,
                )
                print(
                    f"[SMC_EAGLE_DBG cycle={cyc}] step={step} pre-sample "
                    f"topk_p[0]={topk_p[0].tolist()} "
                    f"topk_index[0]={topk_index[0].tolist()}",
                    flush=True,
                )
            next_token, token_logprob = self._sample_from_eagle_topk(topk_p, topk_index)
            draft_logprobs.append(token_logprob)
            all_tokens.append(next_token)

            kv_step = step + kv_step_offset
            if kv_step >= cache_locs.shape[1]:
                raise RuntimeError(
                    "SMC EAGLE decode ran out of allocated KV slots while advancing "
                    f"the sampled chain (kv_step={kv_step}, alloc={cache_locs.shape[1]})."
                )

            if debug:
                print(
                    f"[SMC_EAGLE_DBG cycle={cyc}] step={step} sampled "
                    f"token[:bs]={next_token[:bs].tolist()} "
                    f"log_q[:bs]={token_logprob[:bs].tolist()}",
                    flush=True,
                )
                print(
                    f"[SMC_EAGLE_DBG cycle={cyc}] step={step} draft forward: "
                    f"input_ids[:bs]={next_token[:bs].tolist()} "
                    f"input_ids.shape={tuple(next_token.shape)} "
                    f"out_cache_loc[:bs]={cache_locs[:, kv_step][:bs].tolist()} "
                    f"position[:bs]={all_positions[:, kv_step][:bs].tolist()} "
                    f"seq_lens[:bs]={all_seq_lens[:, kv_step][:bs].tolist()} "
                    f"kv_step={kv_step}",
                    flush=True,
                )

            draft_out = self._run_eagle_decode_step(
                batch,
                token_ids=next_token,
                hidden_states=current_hidden_states,
                seq_lens=all_seq_lens[:, kv_step].contiguous(),
                seq_lens_cpu=ctx.orig_seq_lens_cpu + (kv_step + 1),
                seq_lens_sum=ctx.orig_seq_lens_sum + bs * (kv_step + 1),
                out_cache_loc=cache_locs[:, kv_step].contiguous(),
                position=all_positions[:, kv_step].contiguous(),
            )
            topk_p, topk_index, current_hidden_states = self._extract_eagle_state(
                draft_out
            )
            if debug:
                print(
                    f"[SMC_EAGLE_DBG cycle={cyc}] step={step} post-forward "
                    f"topk_p[0]={topk_p[0].tolist()} "
                    f"topk_index[0]={topk_index[0].tolist()} "
                    f"hidden_states[0].norm={current_hidden_states[0].norm().item():.4f}",
                    flush=True,
                )

        draft_logprobs_stacked = torch.stack(draft_logprobs, dim=1)

        verify_forward_batch, can_run_cuda_graph = ctx.prepare_for_verify(
            self.req_to_token_pool,
            batch,
            self._target_worker,
            all_tokens,
            cache_locs,
            capture_hidden_mode=CaptureHiddenMode.FULL,
        )
        score_result = self._target_worker.forward_batch_generation(
            model_worker_batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
            skip_attn_backend_init=True,
        )

        score_logits = score_result.logits_output.next_token_logits
        expected_rows = bs * (gamma + 1)
        assert score_logits.shape[0] == expected_rows, (
            f"TARGET_VERIFY logits truncated: got {score_logits.shape[0]} rows, "
            f"expected {expected_rows} (bs={bs}, gamma+1={gamma + 1}, "
            f"cuda_graph={can_run_cuda_graph})"
        )
        score_log_probs = torch.log_softmax(score_logits, dim=-1).reshape(
            bs, gamma + 1, -1
        )
        target_tokens = torch.stack(all_tokens[1 : gamma + 1], dim=1)
        score_logprobs_stacked = score_log_probs[:, :gamma, :].gather(
            2, target_tokens.unsqueeze(2)
        ).squeeze(2)
        logprob_diff = (score_logprobs_stacked - draft_logprobs_stacked).sum(dim=1)

        bonus_logits = score_logits.reshape(bs, gamma + 1, -1)[:, -1, :]
        if self.smc_target_temperature > 0:
            bonus_probs = torch.softmax(
                bonus_logits / self.smc_target_temperature, dim=-1
            )
            bonus = torch.multinomial(bonus_probs, num_samples=1).squeeze(-1)
        else:
            bonus = torch.argmax(bonus_logits, dim=-1)

        output_token_ids = torch.stack(
            all_tokens[1 : gamma + 1] + [bonus], dim=1
        )
        next_token_ids = output_token_ids.reshape(-1)
        accept_lens = torch.full(
            (bs,), gamma + 1, dtype=torch.int32, device=self.device
        )

        if debug:
            print(
                f"[SMC_EAGLE_DBG cycle={cyc}] ==== VERIFY ==== "
                f"target_tokens[0]={target_tokens[0].tolist()} "
                f"draft_logq[0]={draft_logprobs_stacked[0].tolist()} "
                f"target_logp[0]={score_logprobs_stacked[0].tolist()} "
                f"logprob_diff[:bs]={logprob_diff[:bs].tolist()} "
                f"bonus[:bs]={bonus[:bs].tolist()}",
                flush=True,
            )
            self._smc_eagle_debug_cycle += 1

        next_token_ids.record_stream(current_stream)
        accept_lens.record_stream(current_stream)
        bonus.record_stream(current_stream)
        logprob_diff.record_stream(current_stream)
        next_draft_input = self._build_eagle_next_draft_input_from_decode(
            batch=batch,
            ctx=ctx,
            score_result=score_result,
            next_token_ids=next_token_ids,
            accept_lens=accept_lens,
            bonus_ids=bonus,
            logprob_diff=logprob_diff,
        )

        return GenerationBatchResult(
            logits_output=score_result.logits_output,
            next_token_ids=next_token_ids,
            accept_lens=accept_lens,
            next_draft_input=next_draft_input,
            logprob_diff=logprob_diff,
            can_run_cuda_graph=can_run_cuda_graph,
        )

    def _forward_idle(self, batch: ModelWorkerBatch):
        return GenerationBatchResult(
            logits_output=LogitsProcessorOutput(next_token_logits=None),
            next_token_ids=torch.empty(0, dtype=torch.int64, device=self.device),
            accept_lens=torch.empty(0, dtype=torch.int32, device=self.device),
            next_draft_input=self._make_idle_draft_input(),
        )

    def _make_idle_draft_input(self) -> SMCDraftInputLike:
        if self._draft_input_cls is SMCEagleDraftInputV2:
            return self._draft_input_cls.create_idle_input(
                self.device, topk=self.server_args.smc_eagle_topk,
            )
        return self._draft_input_cls.create_idle_input(self.device)

    def _make_clean_batch(self, batch: ModelWorkerBatch) -> ModelWorkerBatch:
        """Copy batch with no spec_info (for draft model)."""
        return dataclasses.replace(
            batch, spec_info=None, capture_hidden_mode=CaptureHiddenMode.NULL
        )

    def _sample_from_eagle_topk(
        self,
        topk_p: torch.Tensor,
        topk_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if topk_p is None or topk_index is None:
            raise RuntimeError("SMC EAGLE sampling requires topk_p/topk_index.")

        proposal_probs = topk_p / topk_p.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        if self.smc_draft_temperature > 0:
            sample_idx = torch.multinomial(proposal_probs, num_samples=1).squeeze(-1)
            sampled_logprob = proposal_probs.gather(
                1, sample_idx.unsqueeze(1)
            ).squeeze(1).clamp_min(1e-8).log()
        else:
            sample_idx = torch.argmax(proposal_probs, dim=-1)
            sampled_logprob = torch.zeros(
                proposal_probs.shape[0], dtype=proposal_probs.dtype, device=proposal_probs.device
            )
        sampled_token = topk_index.gather(1, sample_idx.unsqueeze(1)).squeeze(1)
        return sampled_token, sampled_logprob

    def _extract_eagle_state(
        self,
        logits_output: LogitsProcessorOutput,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if logits_output.hidden_states is None:
            raise RuntimeError(
                "SMC EAGLE draft step requires hidden states from the draft model."
            )
        if logits_output.next_token_logits is None:
            raise RuntimeError(
                "SMC EAGLE draft step requires logits from the draft model."
            )

        topk = min(
            self.server_args.smc_eagle_topk,
            logits_output.next_token_logits.shape[-1],
        )
        topk_logits, topk_index = torch.topk(
            logits_output.next_token_logits,
            k=topk,
            dim=-1,
        )
        topk_p = F.softmax(topk_logits, dim=-1)
        return topk_p, topk_index, logits_output.hidden_states

    def _run_eagle_decode_step(
        self,
        batch: ModelWorkerBatch,
        *,
        token_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        seq_lens_sum: int,
        out_cache_loc: torch.Tensor,
        position: torch.Tensor,
    ) -> LogitsProcessorOutput:
        eagle_input = EagleDraftInput(
            verified_id=token_ids,
            hidden_states=hidden_states,
            num_tokens_per_req=1,
            num_tokens_for_logprob_per_req=1,
        )
        draft_batch = dataclasses.replace(
            batch,
            input_ids=token_ids,
            out_cache_loc=out_cache_loc,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            seq_lens_sum=seq_lens_sum,
            spec_info=eagle_input,
            capture_hidden_mode=CaptureHiddenMode.LAST,
        )
        forward_batch = ForwardBatch.init_new(draft_batch, self.draft_runner)
        forward_batch.positions = position
        return self.draft_runner.forward(forward_batch).logits_output

    def _build_eagle_next_draft_input_from_decode(
        self,
        *,
        batch: ModelWorkerBatch,
        ctx: SMCDecodeContext,
        score_result: GenerationBatchResult,
        next_token_ids: torch.Tensor,
        accept_lens: torch.Tensor,
        bonus_ids: torch.Tensor,
        logprob_diff: torch.Tensor,
    ) -> SMCEagleDraftInputV2:
        if score_result.logits_output.hidden_states is None:
            raise RuntimeError(
                "SMC EAGLE decode requires target hidden states to initialize the "
                "next draft state."
            )

        # Target verify returned FULL hidden states for
        #   bs * (gamma + 1) positions in row-major (particle, step) order.
        # For the next cycle we carry the target hidden at each particle's
        # last-verify position (= seq_len + gamma), which the EAGLE head will
        # concat with embed(bonus) at the next cycle's bootstrap draft step.
        bs = len(ctx.orig_seq_lens)
        last_index = (
            torch.arange(bs, device=self.device, dtype=torch.int64)
            * self.speculative_num_draft_tokens
            + self.speculative_num_draft_tokens
            - 1
        )
        last_hidden = score_result.logits_output.hidden_states[last_index]

        return SMCEagleDraftInputV2(
            verified_id=bonus_ids,
            hidden_states=last_hidden,
            topk_p=None,
            topk_index=None,
            logprob_diff=logprob_diff,
            num_tokens_per_req=self.speculative_num_draft_tokens,
        )

    def _run_eagle_prefill_forward(
        self,
        batch: ModelWorkerBatch,
        *,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        mm_input_embeds: Optional[torch.Tensor],
    ) -> LogitsProcessorOutput:
        draft_batch = self._make_eagle_prefill_batch(
            batch,
            target_hidden_states=target_hidden_states,
            next_token_ids=next_token_ids,
        )
        forward_batch = ForwardBatch.init_new(draft_batch, self.draft_runner)
        if mm_input_embeds is not None:
            forward_batch.mm_input_embeds = mm_input_embeds
        return self.draft_runner.forward(forward_batch).logits_output

    def _make_eagle_prefill_batch(
        self,
        batch: ModelWorkerBatch,
        *,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
    ) -> ModelWorkerBatch:
        eagle_input = EagleDraftInput(
            hidden_states=target_hidden_states,
            verified_id=next_token_ids,
            num_tokens_per_req=1,
            num_tokens_for_logprob_per_req=1,
        )
        return dataclasses.replace(
            batch,
            input_ids=batch.input_ids.clone(),
            spec_info=eagle_input,
            capture_hidden_mode=CaptureHiddenMode.LAST,
        )

    def _build_eagle_prefill_next_draft_input(
        self,
        *,
        verified_id: torch.Tensor,
        logits_output: LogitsProcessorOutput,
    ) -> SMCEagleDraftInputV2:
        if logits_output.hidden_states is None:
            raise RuntimeError(
                "SMC EAGLE prefill requires draft hidden states, but the draft "
                "prefill result did not return any."
            )
        if logits_output.next_token_logits is None:
            raise RuntimeError(
                "SMC EAGLE prefill requires draft logits, but the draft prefill "
                "result did not return any."
            )

        topk = min(self.server_args.smc_eagle_topk, logits_output.next_token_logits.shape[-1])
        topk_logits, topk_index = torch.topk(
            logits_output.next_token_logits,
            k=topk,
            dim=-1,
        )
        topk_p = torch.softmax(topk_logits, dim=-1)
        return SMCEagleDraftInputV2(
            verified_id=verified_id,
            hidden_states=logits_output.hidden_states,
            topk_p=topk_p,
            topk_index=topk_index,
            num_tokens_per_req=self.speculative_num_draft_tokens,
        )
