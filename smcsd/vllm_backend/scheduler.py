from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, FinishReason
from vllm.v1.request import Request, RequestStatus

if TYPE_CHECKING:
    from vllm.v1.outputs import ModelRunnerOutput


@dataclass
class SMCParticleGroupState:
    parent_req: Request
    n_particles: int
    num_computed_tokens: int  # initial prompt length; kept for group registration
    seed_token_ids: list[int]
    shared_prefix_block_ids: tuple[list[int], ...]   # [kv_group][block]
    decode_block_ids: list[tuple[list[int], ...]]    # [particle][kv_group][block]
    particle_req_ids: list[str] = field(default_factory=list)
    accumulated_tokens: list[list[int]] = field(default_factory=list)  # [N][tokens]
    particle_finished: list[bool] = field(default_factory=list)
    draft_temperature: float = 1.0
    # Cumulative log-weight per particle: sum_t log(p_target(x_t)/p_draft(x_t)).
    log_weights: list[float] = field(default_factory=list)  # [N]
    # Per-particle sequence lengths: only active particles advance each cycle.
    # After resampling, remapped from ancestors so the runner always gets the
    # correct starting position for each particle's KV context.
    particle_num_computed_tokens: list[int] = field(default_factory=list)  # [N]
    target_temperature: float = 1.0


# Scheduler Output

@dataclass
class NewParticleGroupData:
    """New particle group to be registered."""
    group_id: str
    particle_req_ids: list[str]
    prompt_token_ids: list[int]
    num_computed_tokens: int
    sampling_params: SamplingParams | None
    prefix_block_ids: tuple[list[int], ...]        # [kv_group][block]
    decode_block_ids: list[tuple[list[int], ...]]  # [particle][kv_group][block]
    seed_token_ids: torch.Tensor   # [N]
    seq_lens: torch.Tensor         # [N]
    gamma: int
    temperature: float        # draft temperature
    token_counts: list[int]
    stop_token_ids: list[int]
    max_tokens: int | None


@dataclass
class SMCGroupBatch:
    """Ongoing draft cycle — group already registered."""
    group_id: str
    seed_token_ids: torch.Tensor   # [N]
    seq_lens: torch.Tensor         # [N]
    gamma: int
    particle_finished: list[bool] = field(default_factory=list)  # [N]
    temperature: float = 1.0        # draft temperature
    target_temperature: float = 1.0
    token_counts: list[int] = field(default_factory=list)
    stop_token_ids: list[int] = field(default_factory=list)
    max_tokens: int | None = None


@dataclass
class SMCSchedulerOutput(SchedulerOutput):
    new_particle_groups: list[NewParticleGroupData] = field(default_factory=list)
    ongoing_smc_groups: list[SMCGroupBatch] = field(default_factory=list)


class SMCVLLMScheduler(Scheduler):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        smc = self.vllm_config.smc_config
        assert smc is not None, "SMCVLLMScheduler requires smc_config"
        self.smc_n_particles: int = smc.n_particles
        self.smc_gamma: int = smc.gamma
        self.smc_groups: dict[str, SMCParticleGroupState] = {}  # group_id -> state
        # key: parent_req_id -> (accumulated_tokens[N][T], log_weights[N])
        self._completed_groups: dict[str, tuple[list[list[int]], list[float]]] = {}
        # Requests that finished prefill but are waiting for particle-group capacity.
        self._fork_waitlist: list[Request] = []
        self.max_concurrent_groups: int = (
            self.scheduler_config.max_num_seqs // (self.smc_n_particles + 1)
        )
        if self.max_concurrent_groups < 1:
            raise ValueError(
                "SMC vLLM backend requires max_num_seqs >= n_particles + 1 "
                f"(got max_num_seqs={self.scheduler_config.max_num_seqs}, "
                f"n_particles={self.smc_n_particles})."
            )

    def has_requests(self) -> bool:
        return super().has_requests() or bool(self.smc_groups) or bool(self._fork_waitlist)

    def fork_to_particles(
        self,
        parent_req_id: str,
        n_particles: int,
        gamma: int,
        seed_token_id: int,
        temperature: float,
    ) -> NewParticleGroupData:
        """Allocate KV blocks for N particles."""
        parent_req = self.requests[parent_req_id]
        prompt_token_ids = list(parent_req.prompt_token_ids)
        # At this point only the prompt is materialized in KV
        # the sampled seed has not yet been written into the cache
        num_computed_tokens = len(prompt_token_ids)

        particle_req_ids = [
            f"{parent_req_id}__p{i}__{uuid.uuid4().hex[:8]}"
            for i in range(n_particles)
        ]

        sp = parent_req.sampling_params
        max_tokens = sp.max_tokens if (sp is not None and sp.max_tokens is not None) else 128
        visible_max_tokens = sp.max_tokens if sp is not None else None

        # Share only full prefix blocks across particles
        num_full_shared_blocks = num_computed_tokens // self.block_size
        num_shared_tokens = num_full_shared_blocks * self.block_size
        private_tokens_needed = (
            (num_computed_tokens - num_shared_tokens)
            + max_tokens
            + gamma
        )
        n_decode_blocks = math.ceil(private_tokens_needed / self.block_size)

        # Fork blocks: touches prefix N times and allocates decode blocks per particle
        # Free parent's ownership of prefix blocks
        prefix_block_ids, decode_block_ids_flat = self.kv_cache_manager.fork_blocks(
            parent_req_id=parent_req_id,
            particle_req_ids=particle_req_ids,
            n_decode_blocks=n_decode_blocks,
        )

        # prefix_block_ids: list[int] (single kv-group view from fork_blocks)
        num_kv_groups = len(self.kv_cache_config.kv_cache_groups)
        # fork_blocks returns flat block ids; for now assume single group.
        # TODO: extend fork_blocks to return per-group ids for MLA / hybrid.
        shared_prefix_block_ids: tuple[list[int], ...] = tuple(
            list(prefix_block_ids[:num_full_shared_blocks]) for _ in range(num_kv_groups)
        )
        decode_block_ids: list[tuple[list[int], ...]] = [
            tuple(dec for _ in range(num_kv_groups))
            for dec in decode_block_ids_flat
        ]

        seed_token_ids_tensor = torch.full(
            (n_particles,), seed_token_id, dtype=torch.long
        )
        seq_lens = torch.full(
            (n_particles,), num_computed_tokens, dtype=torch.int32
        )

        group_id = parent_req_id
        target_temperature = (
            sp.temperature if sp is not None and sp.temperature is not None else 1.0
        )
        stop_token_ids = list(sp._all_stop_token_ids) if sp is not None else []
        state = SMCParticleGroupState(
            parent_req=parent_req,
            n_particles=n_particles,
            num_computed_tokens=num_computed_tokens,
            seed_token_ids=[seed_token_id] * n_particles,
            shared_prefix_block_ids=shared_prefix_block_ids,
            decode_block_ids=decode_block_ids,
            particle_req_ids=particle_req_ids,
            accumulated_tokens=[[] for _ in range(n_particles)],
            particle_finished=[False] * n_particles,
            draft_temperature=temperature,
            log_weights=[0.0] * n_particles,
            particle_num_computed_tokens=[num_computed_tokens] * n_particles,
            target_temperature=target_temperature,
        )
        self.smc_groups[group_id] = state

        return NewParticleGroupData(
            group_id=group_id,
            particle_req_ids=particle_req_ids,
            prompt_token_ids=prompt_token_ids,
            num_computed_tokens=num_computed_tokens,
            sampling_params=sp,
            prefix_block_ids=shared_prefix_block_ids,
            decode_block_ids=decode_block_ids,
            seed_token_ids=seed_token_ids_tensor,
            seq_lens=seq_lens,
            gamma=gamma,
            temperature=temperature,
            # The parent request has already emitted the seed token;
            # accumulated_tokens stores only post-seed SMC tokens.
            token_counts=[1] * n_particles,
            stop_token_ids=stop_token_ids,
            max_tokens=visible_max_tokens,
        )

    def build_particle_group(self, group_id: str) -> SMCGroupBatch:
        """Build an SMCGroupBatch from current group state."""
        state = self.smc_groups[group_id]
        seq_lens = torch.tensor(
            state.particle_num_computed_tokens, dtype=torch.int32
        )
        seed_tensor = torch.tensor(state.seed_token_ids, dtype=torch.long)
        sp = state.parent_req.sampling_params
        return SMCGroupBatch(
            group_id=group_id,
            seed_token_ids=seed_tensor,
            seq_lens=seq_lens,
            gamma=self.smc_gamma,
            particle_finished=list(state.particle_finished),
            temperature=state.draft_temperature,
            target_temperature=state.target_temperature,
            token_counts=[1 + len(toks) for toks in state.accumulated_tokens],
            stop_token_ids=list(sp._all_stop_token_ids) if sp is not None else [],
            max_tokens=sp.max_tokens if sp is not None else None,
        )

    def finish_particle_group(self, group_id: str) -> None:
        """Remove CPU-side state for a completed group."""
        self.smc_groups.pop(group_id, None)

    def process_prefill_result(
        self,
        parent_req_id: str,
        seed_token_id: int,
        n_particles: int,
        gamma: int,
        temperature: float,
    ) -> NewParticleGroupData:
        return self.fork_to_particles(
            parent_req_id=parent_req_id,
            n_particles=n_particles,
            gamma=gamma,
            seed_token_id=seed_token_id,
            temperature=temperature,
        )

    def schedule(self) -> SMCSchedulerOutput:
        ongoing_smc_groups: list[SMCGroupBatch] = [
            self.build_particle_group(gid)
            for gid in list(self.smc_groups)
        ]

        for req in list(self.running):
            if (len(req.output_token_ids) == 1
                    and req.request_id not in self.smc_groups):
                self.running.remove(req)
                self._fork_waitlist.append(req)

        pending_fork: list[Request] = []
        available = self.max_concurrent_groups - len(self.smc_groups)
        for req in list(self._fork_waitlist):
            if available <= 0:
                break
            self._fork_waitlist.remove(req)
            pending_fork.append(req)
            available -= 1

        # Prefill requests flow through vllm's base scheduler and execute_model
        base_output = super().schedule()

        # Allocate KV blocks for N particles after prefill completes
        new_particle_groups: list[NewParticleGroupData] = []
        for req in pending_fork:
            new_group = self.process_prefill_result(
                parent_req_id=req.request_id,
                seed_token_id=req.output_token_ids[0],
                n_particles=self.smc_n_particles,
                gamma=self.smc_gamma,
                temperature=getattr(req, "smc_draft_temperature", 
                                    req.sampling_params.temperature),
            )
            new_particle_groups.append(new_group)

        return SMCSchedulerOutput(
            scheduled_new_reqs=base_output.scheduled_new_reqs,
            scheduled_cached_reqs=base_output.scheduled_cached_reqs,
            num_scheduled_tokens=base_output.num_scheduled_tokens,
            total_num_scheduled_tokens=base_output.total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=base_output.scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=base_output.scheduled_encoder_inputs,
            num_common_prefix_blocks=base_output.num_common_prefix_blocks,
            finished_req_ids=base_output.finished_req_ids,
            free_encoder_mm_hashes=base_output.free_encoder_mm_hashes,
            preempted_req_ids=base_output.preempted_req_ids,
            has_structured_output_requests=base_output.has_structured_output_requests,
            pending_structured_output_tokens=base_output.pending_structured_output_tokens,
            num_invalid_spec_tokens=base_output.num_invalid_spec_tokens,
            kv_connector_metadata=base_output.kv_connector_metadata,
            ec_connector_metadata=base_output.ec_connector_metadata,
            new_block_ids_to_zero=base_output.new_block_ids_to_zero,
            new_particle_groups=new_particle_groups,
            ongoing_smc_groups=ongoing_smc_groups,
        )

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        from smcsd.vllm_backend.outputs import SMCModelRunnerOutput

        result = super().update_from_output(scheduler_output, model_runner_output)

        if not isinstance(model_runner_output, SMCModelRunnerOutput):
            return result

        for group_id, (draft_token_ids, log_probs, next_seed_ids) in model_runner_output.smc_draft_results.items():
            state = self.smc_groups.get(group_id)
            if state is None:
                continue

            # draft_token_ids: [A, gamma+1]; [:, 0] is the carried seed input.
            # Commits gamma+1 tokens per particle: x_1..x_gamma from draft
            # plus next_seed_ids (the target bonus token), which seeds the next cycle.
            active_ps = [p for p, f in enumerate(state.particle_finished) if not f]
            new_tokens = torch.cat(
                (draft_token_ids[:, 1:], next_seed_ids.unsqueeze(1)),
                dim=1,
            )  # [A, gamma+1]

            sp = state.parent_req.sampling_params
            stop_ids: set[int] = sp._all_stop_token_ids if sp is not None else set()
            max_tokens = sp.max_tokens if sp is not None else None

            for i, p in enumerate(active_ps):
                particle_tokens = new_tokens[i].tolist()
                if max_tokens is not None:
                    remaining = max_tokens - (1 + len(state.accumulated_tokens[p]))
                    if remaining <= 0:
                        state.particle_finished[p] = True
                        continue
                    particle_tokens = particle_tokens[:remaining]

                # Trim at first stop token within the visible committed tokens.
                if stop_ids:
                    stop_pos = next(
                        (j for j, t in enumerate(particle_tokens) if t in stop_ids),
                        -1,
                    )
                    if stop_pos >= 0:
                        particle_tokens = particle_tokens[: stop_pos + 1]
                        state.particle_finished[p] = True
                state.accumulated_tokens[p].extend(particle_tokens)
                if max_tokens is not None and 1 + len(state.accumulated_tokens[p]) >= max_tokens:
                    state.particle_finished[p] = True

            # Accumulate log-weight increments.
            logprob_diff = model_runner_output.smc_logprob_diffs.get(group_id)
            if logprob_diff is not None:
                for i, p in enumerate(active_ps):
                    state.log_weights[p] += float(logprob_diff[i].item())

            # Advance per-particle seq_lens for active particles only.
            for p in active_ps:
                state.particle_num_computed_tokens[p] += self.smc_gamma + 1

            # Apply resampling ancestry over all N particles.
            ancestor_indices_t = model_runner_output.resampled_groups.get(group_id)
            if ancestor_indices_t is not None:
                anc = ancestor_indices_t.tolist()  # [N], indices into 0..N-1
                # Snapshot sources first to avoid aliasing.
                # Surviving particles restart with uniform log weights.
                new_weights   = [0.0 for _ in range(state.n_particles)]
                new_tokens    = [list(state.accumulated_tokens[anc[p]])        for p in range(state.n_particles)]
                new_finished  = [state.particle_finished[anc[p]]              for p in range(state.n_particles)]
                new_seq_lens  = [state.particle_num_computed_tokens[anc[p]]   for p in range(state.n_particles)]
                for p in range(state.n_particles):
                    state.log_weights[p]                    = new_weights[p]
                    state.accumulated_tokens[p]             = new_tokens[p]
                    state.particle_finished[p]              = new_finished[p]
                    state.particle_num_computed_tokens[p]   = new_seq_lens[p]

            if all(state.particle_finished):
                self._finish_group(group_id, state, result)
            else:
                # Build per-particle seeds for this cycle: active particles get the new
                # bonus token; finished particles keep their previous seed.
                full_seeds = list(state.seed_token_ids)
                for i, p in enumerate(active_ps):
                    full_seeds[p] = int(next_seed_ids[i].item())
                # Apply ancestry so each particle uses its ancestor's seed.
                if ancestor_indices_t is not None:
                    anc = ancestor_indices_t.tolist()
                    for p in range(state.n_particles):
                        state.seed_token_ids[p] = full_seeds[anc[p]]
                else:
                    state.seed_token_ids = full_seeds

        return result

    def _finish_group(
        self,
        group_id: str,
        state: SMCParticleGroupState,
        result: dict[int, EngineCoreOutputs],
    ) -> None:
        """Emit a finish signal for the parent request and clean up."""
        self._completed_groups[group_id] = (
            state.accumulated_tokens,
            list(state.log_weights),
        )

        parent_req = state.parent_req
        finish_out = EngineCoreOutput(
            request_id=group_id,
            new_token_ids=[],
            finish_reason=FinishReason.STOP,
        )
        client_idx = parent_req.client_index
        existing = result.get(client_idx)
        if existing is None:
            result[client_idx] = EngineCoreOutputs(outputs=[finish_out])
        else:
            result[client_idx] = EngineCoreOutputs(
                outputs=existing.outputs + [finish_out],
                scheduler_stats=existing.scheduler_stats,
                timestamp=existing.timestamp,
            )

        self.finish_requests(group_id, RequestStatus.FINISHED_STOPPED)
        for p_id in state.particle_req_ids:
            self.kv_cache_manager.coordinator.free(p_id)
        self.finish_particle_group(group_id)
