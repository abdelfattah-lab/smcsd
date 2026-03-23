from __future__ import annotations

import copy
import logging
from typing import Dict, List, Optional, Sequence, Union

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    ModelWorkerBatch,
    Req,
    ScheduleBatch,
)
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.speculative.smc_info import (
    SMCDraftInput,
    SMCParentUpdate,
    SMCParticleState,
    SMCScoreInput,
    SMCRequestState,
    build_smc_causal_mask,
    build_smc_positions,
    effective_sample_size,
    initialize_smc_request_state,
    multinomial_resample,
    normalize_log_weights,
    resolve_smc_proposal_length,
    resolve_smc_seed_output_ids,
    systematic_resample,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.standalone_worker_v2 import StandaloneWorkerV2


logger = logging.getLogger(__name__)


class _SMCInternalTreeCache:
    """Minimal allocator wrapper for worker-local internal SMC batches."""

    def __init__(self, token_to_kv_pool_allocator):
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.page_size = token_to_kv_pool_allocator.page_size

    def is_chunk_cache(self) -> bool:
        return False

    def evict(self, *args, **kwargs) -> None:
        return None

    def pretty_print(self) -> None:
        return None

    def supports_mamba(self) -> bool:
        return False

    def supports_swa(self) -> bool:
        return False


class SMCWorkerV2(StandaloneWorkerV2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.smc_gamma = self.server_args.smc_gamma
        self._internal_tree_cache = _SMCInternalTreeCache(
            self.token_to_kv_pool_allocator
        )

    @property
    def model_runner(self):
        return self.target_worker.model_runner

    @property
    def model_config(self):
        return self.target_worker.model_config

    def forward_batch_generation(
        self, batch: Union[ScheduleBatch, ModelWorkerBatch]
    ) -> GenerationBatchResult:
        is_overlap_batch = isinstance(batch, ModelWorkerBatch)
        overlap_draft_input = (
            batch.spec_info
            if is_overlap_batch and isinstance(batch.spec_info, SMCDraftInput)
            else None
        )

        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            model_worker_batch = batch if is_overlap_batch else batch.get_model_worker_batch()
            result = self.target_worker.forward_batch_generation(model_worker_batch)
            if is_overlap_batch:
                result.next_draft_input = self._build_prefill_overlap_input(
                    model_worker_batch, result
                )
            return result

        if not batch.reqs:
            return self._build_empty_decode_result(is_overlap_batch)

        parent_updates: List[SMCParentUpdate] = []
        any_cuda_graph = False
        for req_idx, parent_req in enumerate(batch.reqs):
            if parent_req.finished():
                parent_updates.append(
                    SMCParentUpdate(
                        done=True,
                        best_particle_idx=0,
                        best_output_ids=list(parent_req.output_ids),
                        finish_reason=copy.copy(parent_req.finished_reason),
                        finished_len=parent_req.finished_len,
                    )
                )
                continue

            if parent_req.smc_state is None:
                seed_output_ids = None
                if overlap_draft_input is not None:
                    try:
                        seed_output_ids = resolve_smc_seed_output_ids(
                            parent_req,
                            overlap_last_token_id=int(
                                overlap_draft_input.last_token_ids[req_idx].item()
                            ),
                            overlap_new_seq_len=int(
                                overlap_draft_input.new_seq_lens[req_idx].item()
                            ),
                        )
                    except ValueError as exc:
                        error = str(exc)
                        parent_req.finished_reason = FINISH_ABORT(error)
                        parent_updates.append(
                            SMCParentUpdate(
                                done=True,
                                best_particle_idx=0,
                                best_output_ids=list(parent_req.output_ids),
                                finish_reason=copy.copy(parent_req.finished_reason),
                                finished_len=parent_req.finished_len,
                            )
                        )
                        continue
                error = initialize_smc_request_state(
                    parent_req,
                    server_args=self.server_args,
                    req_to_token_pool=self.req_to_token_pool,
                    token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                    seed_output_ids=seed_output_ids,
                )
                if error is not None:
                    parent_req.finished_reason = FINISH_ABORT(error)
                    parent_updates.append(
                        SMCParentUpdate(
                            done=True,
                            best_particle_idx=0,
                            best_output_ids=list(parent_req.output_ids),
                            finish_reason=copy.copy(parent_req.finished_reason),
                            finished_len=parent_req.finished_len,
                        )
                    )
                    continue

            try:
                self._ensure_draft_prefix_filled(parent_req)
                any_cuda_graph |= self._run_parent_step(parent_req)
            except Exception as exc:
                logger.exception("SMC parent step failed for request %s", parent_req.rid)
                parent_req.finished_reason = FINISH_ABORT(str(exc))
                parent_req.finished_len = len(parent_req.output_ids)
                for particle in parent_req.smc_state.particles:
                    particle.target_req.finished_reason = copy.copy(
                        parent_req.finished_reason
                    )
                    particle.target_req.finished_len = parent_req.finished_len
                    particle.draft_req.finished_reason = copy.copy(
                        parent_req.finished_reason
                    )
                    particle.draft_req.finished_len = parent_req.finished_len

            parent_updates.append(self._build_parent_update(parent_req))

        if is_overlap_batch:
            next_draft_input = self._build_overlap_draft_input(batch.reqs, parent_updates)
            return GenerationBatchResult(
                logits_output=self._empty_logits_output(),
                next_token_ids=next_draft_input.last_token_ids.to(dtype=torch.int32),
                can_run_cuda_graph=any_cuda_graph,
                next_draft_input=next_draft_input,
                smc_parent_updates=parent_updates,
            )

        return GenerationBatchResult(
            next_token_ids=torch.empty((0,), dtype=torch.int32, device=self.device),
            can_run_cuda_graph=any_cuda_graph,
            smc_parent_updates=parent_updates,
        )

    def _build_empty_decode_result(self, is_overlap_batch: bool) -> GenerationBatchResult:
        next_draft_input = (
            SMCDraftInput.create_idle_input(self.device) if is_overlap_batch else None
        )
        return GenerationBatchResult(
            logits_output=self._empty_logits_output() if is_overlap_batch else None,
            next_token_ids=torch.empty((0,), dtype=torch.int32, device=self.device),
            can_run_cuda_graph=False,
            next_draft_input=next_draft_input,
            smc_parent_updates=[],
        )

    def _empty_logits_output(self) -> LogitsProcessorOutput:
        return LogitsProcessorOutput(next_token_logits=None, hidden_states=None)

    def _build_prefill_overlap_input(
        self, batch: ModelWorkerBatch, result: GenerationBatchResult
    ) -> SMCDraftInput:
        assert result.next_token_ids is not None
        return SMCDraftInput(
            last_token_ids=result.next_token_ids.to(dtype=torch.int64),
            new_seq_lens=(batch.seq_lens + 1).to(dtype=torch.int32),
        )

    def _build_overlap_draft_input(
        self, parent_reqs: Sequence[Req], parent_updates: Sequence[SMCParentUpdate]
    ) -> SMCDraftInput:
        last_token_ids: List[int] = []
        new_seq_lens: List[int] = []
        for req, update in zip(parent_reqs, parent_updates, strict=True):
            best_output_ids = (
                update.best_output_ids if update.best_output_ids is not None else req.output_ids
            )
            if best_output_ids:
                last_token_ids.append(best_output_ids[-1])
            else:
                last_token_ids.append(req.origin_input_ids[-1])
            new_seq_lens.append(len(req.origin_input_ids) + len(best_output_ids))

        verify_done = torch.get_device_module(self.device).Event()
        verify_done.record()
        return SMCDraftInput(
            last_token_ids=torch.tensor(
                last_token_ids, dtype=torch.int64, device=self.device
            ),
            new_seq_lens=torch.tensor(
                new_seq_lens, dtype=torch.int32, device=self.device
            ),
            verify_done=verify_done,
        )

    def _ensure_draft_prefix_filled(self, parent_req: Req) -> None:
        state = parent_req.smc_state
        if state is None or state.draft_prefix_materialized:
            return

        reqs: List[Req] = []
        committed_seq_lens: List[int] = []
        for particle in state.particles:
            draft_req = particle.draft_req
            if draft_req.kv_committed_len <= 0:
                continue
            reqs.append(draft_req)
            committed_seq_lens.append(draft_req.kv_committed_len)

        if reqs:
            with self.draft_worker.draft_tp_context(
                self.draft_worker.draft_runner.tp_group
            ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
                self._run_draft_prefix_fill_batch(
                    # This is a one-time draft-prefix KV fill, not an EAGLE-style
                    # recurring draft-extend stage with accepted bonus tokens.
                    reqs,
                    committed_seq_lens,
                    worker=self.draft_worker.draft_worker,
                )

        state.draft_prefix_materialized = True

    def _run_draft_prefix_fill_batch(
        self,
        reqs: Sequence[Req],
        committed_seq_lens: Sequence[int],
        worker,
    ) -> None:
        if not reqs:
            return

        batch = ScheduleBatch.init_new(
            reqs=list(reqs),
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self._internal_tree_cache,
            model_config=worker.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        batch.forward_mode = ForwardMode.EXTEND
        batch.return_logprob = False

        input_ids: List[int] = []
        out_cache_loc: List[torch.Tensor] = []
        seq_lens: List[int] = []
        prefix_lens: List[int] = []
        extend_lens: List[int] = []

        for req, committed_seq_len in zip(reqs, committed_seq_lens, strict=True):
            prompt_len = len(req.origin_input_ids)
            committed_output_len = committed_seq_len - prompt_len
            if committed_output_len < 0 or committed_output_len > len(req.output_ids):
                raise AssertionError(
                    "SMC draft prefix fill received inconsistent lengths: "
                    f"rid={req.rid}, prompt_len={prompt_len}, "
                    f"committed_seq_len={committed_seq_len}, "
                    f"output_len={len(req.output_ids)}"
                )

            fill_ids = req.origin_input_ids + req.output_ids[:committed_output_len]
            if len(fill_ids) != committed_seq_len:
                raise AssertionError(
                    "SMC draft prefix fill built an unexpected fill length: "
                    f"rid={req.rid}, expected={committed_seq_len}, actual={len(fill_ids)}"
                )

            input_ids.extend(fill_ids)
            out_cache_loc.append(
                self.req_to_token_pool.req_to_token[
                    req.req_pool_idx, :committed_seq_len
                ].to(dtype=torch.int64, copy=True)
            )
            seq_lens.append(committed_seq_len)
            prefix_lens.append(0)
            extend_lens.append(committed_seq_len)

        batch.input_ids = torch.tensor(input_ids, dtype=torch.int64, device=self.device)
        batch.req_pool_indices = torch.tensor(
            [req.req_pool_idx for req in reqs], dtype=torch.int64, device=self.device
        )
        batch.seq_lens = torch.tensor(seq_lens, dtype=torch.int64, device=self.device)
        batch.seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)
        batch.seq_lens_sum = sum(seq_lens)
        batch.orig_seq_lens = torch.tensor(
            seq_lens, dtype=torch.int32, device=self.device
        )
        batch.out_cache_loc = torch.cat(out_cache_loc)
        batch.prefix_lens = prefix_lens
        batch.extend_lens = extend_lens
        batch.extend_num_tokens = batch.seq_lens_sum
        batch.extend_logprob_start_lens = [0] * len(reqs)
        batch.extend_input_logprob_token_ids = None
        batch.top_logprobs_nums = None
        batch.token_ids_logprobs = None
        batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
            batch,
            worker.model_config.vocab_size,
        )

        worker.forward_batch_generation(batch.get_model_worker_batch(), is_verify=True)

    def _run_parent_step(self, parent_req: Req) -> bool:
        state = parent_req.smc_state
        assert state is not None

        active_particles = state.active_particles()
        if not active_particles:
            return False

        draft_tokens: Dict[int, List[int]] = {
            particle.target_req.smc_particle_idx: [] for particle in active_particles
        }
        draft_logprobs: Dict[int, float] = {
            particle.target_req.smc_particle_idx: 0.0 for particle in active_particles
        }
        draft_finished: Dict[int, bool] = {
            particle.target_req.smc_particle_idx: False for particle in active_particles
        }
        target_had_pending_token: Dict[int, bool] = {
            particle.target_req.smc_particle_idx: self._req_has_uncommitted_token(
                particle.target_req
            )
            for particle in active_particles
        }

        draft_sampling_info = self._make_sampling_info(
            [particle.draft_req for particle in active_particles],
            self.draft_worker.draft_worker.model_config.vocab_size,
        )
        can_run_cuda_graph = False
        if self._can_use_fused_draft_cuda_graph(active_particles, draft_sampling_info):
            with self.draft_worker.draft_tp_context(
                self.draft_worker.draft_runner.tp_group
            ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
                draft_can_run_cuda_graph = self._run_fused_draft_particles(
                    active_particles,
                    draft_tokens,
                    draft_logprobs,
                    draft_finished,
                    draft_sampling_info,
                )
            can_run_cuda_graph = can_run_cuda_graph or draft_can_run_cuda_graph
        else:
            for _ in range(self.smc_gamma):
                step_particles = [
                    particle for particle in active_particles if not particle.is_finished
                ]
                if not step_particles:
                    break

                with self.draft_worker.draft_tp_context(
                    self.draft_worker.draft_runner.tp_group
                ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
                    (
                        next_token_ids,
                        next_token_logprobs,
                        draft_can_run_cuda_graph,
                    ) = self._run_draft_step_particles(step_particles)
                can_run_cuda_graph = can_run_cuda_graph or draft_can_run_cuda_graph

                for particle, token_id, token_logprob in zip(
                    step_particles, next_token_ids, next_token_logprobs, strict=True
                ):
                    particle_idx = particle.target_req.smc_particle_idx
                    draft_tokens[particle_idx].append(token_id)
                    draft_logprobs[particle_idx] += float(token_logprob)
                    self._append_particle_token(particle, token_id)
                    draft_finished[particle_idx] = particle.is_finished

        particles_to_score = [
            particle
            for particle in active_particles
            if draft_tokens[particle.target_req.smc_particle_idx]
        ]
        if particles_to_score:
            target_result = self._run_score_batch(
                particles_to_score=particles_to_score,
                draft_tokens=draft_tokens,
                draft_logprobs=draft_logprobs,
                draft_finished=draft_finished,
                target_had_pending_token=target_had_pending_token,
            )
            can_run_cuda_graph = (
                can_run_cuda_graph or target_result.can_run_cuda_graph
            )

        state.step_index += 1
        return can_run_cuda_graph

    def _make_sampling_info(
        self,
        reqs: Sequence[Req],
        vocab_size: int,
    ) -> SamplingBatchInfo:
        batch = ScheduleBatch.init_new(
            reqs=list(reqs),
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self._internal_tree_cache,
            model_config=self.draft_worker.draft_worker.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        return SamplingBatchInfo.from_schedule_batch(batch, vocab_size)

    def _can_use_fused_draft_cuda_graph(
        self,
        particles: Sequence[SMCParticleState],
        sampling_info: SamplingBatchInfo,
    ) -> bool:
        runner = getattr(self.draft_worker, "smc_draft_cuda_graph_runner", None)
        return bool(
            runner
            and runner.supports_replay(
                [particle.draft_req for particle in particles], sampling_info
            )
        )

    def _run_fused_draft_particles(
        self,
        particles: Sequence[SMCParticleState],
        draft_tokens: Dict[int, List[int]],
        draft_logprobs: Dict[int, float],
        draft_finished: Dict[int, bool],
        sampling_info: SamplingBatchInfo,
    ) -> bool:
        runner = self.draft_worker.smc_draft_cuda_graph_runner
        draft_reqs = [particle.draft_req for particle in particles]
        base_committed_lens = [req.kv_committed_len for req in draft_reqs]
        token_matrix, logprob_matrix = runner.replay(draft_reqs, sampling_info)

        for particle, token_row, logprob_row, base_committed_len in zip(
            particles,
            token_matrix.tolist(),
            logprob_matrix.tolist(),
            base_committed_lens,
            strict=True,
        ):
            particle_idx = particle.target_req.smc_particle_idx
            committed_steps, proposal_finished = resolve_smc_proposal_length(
                particle.target_req, token_row
            )
            for token_id, token_logprob in zip(
                token_row[:committed_steps],
                logprob_row[:committed_steps],
                strict=True,
            ):
                draft_tokens[particle_idx].append(int(token_id))
                draft_logprobs[particle_idx] += float(token_logprob)
                self._append_particle_token(particle, int(token_id))

            particle.draft_req.kv_committed_len = base_committed_len + committed_steps
            particle.draft_req.decode_batch_idx += committed_steps
            particle.draft_req.prefix_indices = self.req_to_token_pool.req_to_token[
                particle.draft_req.req_pool_idx, : particle.draft_req.kv_committed_len
            ].to(dtype=torch.int64, copy=True)
            draft_finished[particle_idx] = proposal_finished

        return True

    def _append_particle_token(self, particle: SMCParticleState, token_id: int) -> None:
        particle.target_req.output_ids.append(token_id)
        particle.draft_req.output_ids.append(token_id)
        particle.target_req.check_finished(1)
        particle.draft_req.finished_reason = copy.copy(particle.target_req.finished_reason)
        particle.draft_req.finished_len = particle.target_req.finished_len
        particle.draft_req.finished_output = particle.target_req.finished_output
        particle.draft_req.to_finish = copy.copy(particle.target_req.to_finish)

    def _run_score_batch(
        self,
        particles_to_score: Sequence[SMCParticleState],
        draft_tokens: Dict[int, List[int]],
        draft_logprobs: Dict[int, float],
        draft_finished: Dict[int, bool],
        target_had_pending_token: Dict[int, bool],
    ) -> GenerationBatchResult:
        model_worker_batch = self._make_score_model_worker_batch(
            particles_to_score=particles_to_score,
            draft_tokens=draft_tokens,
            draft_logprobs=draft_logprobs,
            draft_finished=draft_finished,
            target_had_pending_token=target_had_pending_token,
        )
        score_input: SMCScoreInput = model_worker_batch.spec_info
        verify_forward_batch, can_run_cuda_graph = score_input.prepare_for_v2_verify(
            self.req_to_token_pool,
            model_worker_batch,
            self.target_worker,
        )
        forward_output = self.target_worker.forward_batch_generation(
            model_worker_batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
            skip_attn_backend_init=True,
        )
        assert forward_output.logits_output is not None
        score_input.sample(
            model_worker_batch,
            forward_output.logits_output,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
        )
        forward_output.can_run_cuda_graph = can_run_cuda_graph
        return forward_output

    def _run_draft_step_particles(
        self, step_particles: Sequence[SMCParticleState]
    ) -> tuple[List[int], List[float], bool]:
        next_token_ids: List[Optional[int]] = [None] * len(step_particles)
        next_token_logprobs: List[Optional[float]] = [None] * len(step_particles)

        decode_result = self._run_decode_batch(
            [particle.draft_req for particle in step_particles],
            worker=self.draft_worker.draft_worker,
        )
        self._fill_draft_step_outputs(
            range(len(step_particles)),
            decode_result,
            next_token_ids,
            next_token_logprobs,
        )

        assert all(token_id is not None for token_id in next_token_ids)
        assert all(logprob is not None for logprob in next_token_logprobs)
        return next_token_ids, next_token_logprobs, decode_result.can_run_cuda_graph

    def _fill_draft_step_outputs(
        self,
        indices: Sequence[int],
        result: GenerationBatchResult,
        next_token_ids: List[Optional[int]],
        next_token_logprobs: List[Optional[float]],
    ) -> None:
        assert result.next_token_ids is not None
        assert result.logits_output is not None
        assert result.logits_output.next_token_logprobs is not None

        token_ids = result.next_token_ids.tolist()
        token_logprobs = result.logits_output.next_token_logprobs.tolist()
        for batch_idx, token_id, token_logprob in zip(
            indices, token_ids, token_logprobs, strict=True
        ):
            next_token_ids[batch_idx] = int(token_id)
            next_token_logprobs[batch_idx] = float(token_logprob)

    def _req_has_uncommitted_token(self, req: Req) -> bool:
        return len(req.origin_input_ids) + len(req.output_ids) > req.kv_committed_len

    def _run_decode_batch(self, reqs: List[Req], worker) -> GenerationBatchResult:
        batch = self._make_decode_batch(reqs, worker.model_config)
        return worker.forward_batch_generation(batch.get_model_worker_batch())

    def _make_score_model_worker_batch(
        self,
        particles_to_score: Sequence[SMCParticleState],
        draft_tokens: Dict[int, List[int]],
        draft_logprobs: Dict[int, float],
        draft_finished: Dict[int, bool],
        target_had_pending_token: Dict[int, bool],
    ) -> ModelWorkerBatch:
        reqs = [particle.target_req for particle in particles_to_score]
        batch = ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self._internal_tree_cache,
            model_config=self.target_worker.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.SMC,
        )
        score_token_num = self.server_args.speculative_num_draft_tokens
        score_start_lens: List[int] = []
        flat_score_tokens: List[int] = []
        draft_lengths: List[int] = []
        draft_logprob_values: List[float] = []
        draft_finished_flags: List[bool] = []

        for particle in particles_to_score:
            req = particle.target_req
            particle_idx = req.smc_particle_idx
            proposal_tokens = draft_tokens[particle_idx]
            if not proposal_tokens:
                raise AssertionError(
                    f"SMC score batch received an empty proposal for particle {particle_idx}"
                )

            had_pending_token = target_had_pending_token[particle_idx]
            score_start_len = req.kv_committed_len if had_pending_token else req.kv_committed_len - 1
            if score_start_len < 0:
                raise AssertionError(
                    "SMC target scoring requires a non-empty anchor prefix: "
                    f"rid={req.rid}, kv_committed_len={req.kv_committed_len}, "
                    f"had_pending={had_pending_token}"
                )

            anchor_offset = len(req.output_ids) - len(proposal_tokens) - 1
            anchor_token = (
                req.output_ids[anchor_offset]
                if anchor_offset >= 0
                else req.origin_input_ids[-1]
            )
            row_tokens = [anchor_token] + proposal_tokens
            if len(row_tokens) > score_token_num:
                raise AssertionError(
                    "SMC score batch exceeded the fixed verify width: "
                    f"rid={req.rid}, width={score_token_num}, actual={len(row_tokens)}"
                )
            row_tokens.extend([row_tokens[-1]] * (score_token_num - len(row_tokens)))

            score_start_lens.append(score_start_len)
            flat_score_tokens.extend(row_tokens)
            draft_lengths.append(len(proposal_tokens))
            draft_logprob_values.append(draft_logprobs[particle_idx])
            draft_finished_flags.append(draft_finished[particle_idx])

        batch.forward_mode = ForwardMode.DECODE
        batch.req_pool_indices = torch.tensor(
            [req.req_pool_idx for req in reqs], dtype=torch.int64, device=self.device
        )
        batch.seq_lens = torch.tensor(
            score_start_lens,
            dtype=torch.int64,
            device=self.device,
        )
        batch.seq_lens_cpu = batch.seq_lens.cpu()
        batch.seq_lens_sum = int(batch.seq_lens_cpu.sum().item())
        batch.orig_seq_lens = torch.tensor(
            [len(req.origin_input_ids) for req in reqs],
            dtype=torch.int32,
            device=self.device,
        )
        batch.top_logprobs_nums = [0] * len(reqs)
        batch.token_ids_logprobs = [None] * len(reqs)
        batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
            batch,
            self.target_worker.model_config.vocab_size,
        )
        batch.spec_info = SMCScoreInput(
            draft_token=torch.tensor(
                flat_score_tokens, dtype=torch.int64, device=self.device
            ),
            draft_lengths=torch.tensor(
                draft_lengths, dtype=torch.int32, device=self.device
            ),
            draft_logprobs=torch.tensor(
                draft_logprob_values, dtype=torch.float32, device=self.device
            ),
            draft_finished=torch.tensor(
                draft_finished_flags, dtype=torch.bool, device=self.device
            ),
            positions=build_smc_positions(batch.seq_lens, score_token_num),
            custom_mask=build_smc_causal_mask(batch.seq_lens, score_token_num),
            draft_token_num=score_token_num,
        )
        return batch.get_model_worker_batch()

    def _make_decode_batch(self, reqs: List[Req], model_config) -> ScheduleBatch:
        batch = ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self._internal_tree_cache,
            model_config=model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        batch.req_pool_indices = torch.tensor(
            [req.req_pool_idx for req in reqs], dtype=torch.int64, device=self.device
        )
        batch.seq_lens = torch.tensor(
            [req.kv_committed_len for req in reqs],
            dtype=torch.int64,
            device=self.device,
        )
        batch.seq_lens_cpu = batch.seq_lens.cpu()
        batch.seq_lens_sum = int(batch.seq_lens_cpu.sum().item())
        batch.orig_seq_lens = torch.tensor(
            [len(req.origin_input_ids) for req in reqs],
            dtype=torch.int32,
            device=self.device,
        )
        batch.output_ids = torch.tensor(
            [
                req.output_ids[-1] if req.output_ids else req.origin_input_ids[-1]
                for req in reqs
            ],
            dtype=torch.int64,
            device=self.device,
        )
        batch.top_logprobs_nums = [0] * len(reqs)
        batch.token_ids_logprobs = [None] * len(reqs)
        batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
            batch,
            model_config.vocab_size,
        )
        batch.prepare_for_decode()
        return batch

    def _maybe_resample(self, state: SMCRequestState) -> None:
        if state is None:
            return

        active_particles = state.active_particles()
        if len(active_particles) <= 1:
            return

        normalized_weights = normalize_log_weights(
            [particle.log_weight for particle in active_particles]
        )
        ess = effective_sample_size(normalized_weights)
        if ess >= len(active_particles) * state.resample_threshold:
            return

        if state.resample_method == "multinomial":
            ancestors = multinomial_resample(normalized_weights)
        else:
            ancestors = systematic_resample(normalized_weights)

        source_snapshots = [
            self._snapshot_particle(active_particles[ancestor_idx])
            for ancestor_idx in ancestors
        ]

        for dst_particle, snapshot in zip(active_particles, source_snapshots, strict=True):
            self._restore_particle_from_snapshot(dst_particle, snapshot)
            dst_particle.log_weight = 0.0

        for snapshot in source_snapshots:
            self._release_snapshot(snapshot)

    def _snapshot_particle(self, particle: SMCParticleState) -> dict:
        return {
            "target_req": self._snapshot_req(particle.target_req),
            "draft_req": self._snapshot_req(particle.draft_req),
            "log_weight": particle.log_weight,
        }

    def _snapshot_req(self, req: Req) -> dict:
        seq_len = req.kv_committed_len
        if seq_len > 0:
            indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :seq_len
            ].to(dtype=torch.int64, copy=True)
            self.token_to_kv_pool_allocator.inc_ref(indices)
        else:
            indices = torch.empty((0,), dtype=torch.int64)

        return {
            "indices": indices,
            "output_ids": list(req.output_ids),
            "finished_reason": copy.copy(req.finished_reason),
            "finished_len": req.finished_len,
            "finished_output": req.finished_output,
            "to_finish": copy.copy(req.to_finish),
            "kv_committed_len": req.kv_committed_len,
            "decoded_text": req.decoded_text,
            "surr_offset": req.surr_offset,
            "read_offset": req.read_offset,
            "cache_protected_len": req.cache_protected_len,
            "logprob_start_len": req.logprob_start_len,
        }

    def _restore_particle_from_snapshot(
        self, particle: SMCParticleState, snapshot: dict
    ) -> None:
        self._restore_req_from_snapshot(particle.target_req, snapshot["target_req"])
        self._restore_req_from_snapshot(particle.draft_req, snapshot["draft_req"])
        particle.log_weight = snapshot["log_weight"]

    def _restore_req_from_snapshot(self, req: Req, snapshot: dict) -> None:
        if req.kv_allocated_len > 0:
            old_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, : req.kv_allocated_len
            ].to(dtype=torch.int64, copy=True)
            self.token_to_kv_pool_allocator.dec_ref_and_free(old_indices)

        indices = snapshot["indices"]
        if indices.numel() > 0:
            self.token_to_kv_pool_allocator.inc_ref(indices)
            self.req_to_token_pool.write(
                (req.req_pool_idx, slice(0, indices.shape[0])),
                indices.to(torch.int32),
            )
            req.prefix_indices = indices.to(dtype=torch.int64, copy=True)
        else:
            req.prefix_indices = torch.empty((0,), dtype=torch.int64)

        req.output_ids = list(snapshot["output_ids"])
        req.finished_reason = copy.copy(snapshot["finished_reason"])
        req.finished_len = snapshot["finished_len"]
        req.finished_output = snapshot["finished_output"]
        req.to_finish = copy.copy(snapshot["to_finish"])
        req.kv_committed_len = snapshot["kv_committed_len"]
        req.kv_allocated_len = snapshot["kv_committed_len"]
        req.decoded_text = snapshot["decoded_text"]
        req.surr_offset = snapshot["surr_offset"]
        req.read_offset = snapshot["read_offset"]
        req.cache_protected_len = snapshot["cache_protected_len"]
        req.logprob_start_len = snapshot["logprob_start_len"]

    def _release_snapshot(self, snapshot: dict) -> None:
        self._release_req_snapshot(snapshot["target_req"])
        self._release_req_snapshot(snapshot["draft_req"])

    def _release_req_snapshot(self, snapshot: dict) -> None:
        indices = snapshot["indices"]
        if indices.numel() > 0:
            self.token_to_kv_pool_allocator.dec_ref_and_free(indices)

    def _build_parent_update(self, parent_req: Req) -> SMCParentUpdate:
        state = parent_req.smc_state
        if state is None:
            return SMCParentUpdate(
                done=parent_req.finished(),
                best_particle_idx=0,
                best_output_ids=list(parent_req.output_ids),
                finish_reason=copy.copy(parent_req.finished_reason),
                finished_len=parent_req.finished_len,
            )

        best_particle = state.get_best_particle()
        return SMCParentUpdate(
            done=state.is_terminal(),
            best_particle_idx=state.best_particle_idx,
            best_output_ids=list(best_particle.output_ids),
            finish_reason=copy.copy(best_particle.target_req.finished_reason),
            finished_len=best_particle.target_req.finished_len,
        )
