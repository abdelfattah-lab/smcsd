from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field

import torch

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request


@dataclass
class SMCParticleGroupState:
    parent_req: Request
    n_particles: int
    num_computed_tokens: int               # parent prompt length at fork
    seed_token_ids: list[int]              # per particle; updated after each cycle
    shared_prefix_block_ids: tuple[list[int], ...]   # [kv_group][block]
    decode_block_ids: list[tuple[list[int], ...]]    # [particle][kv_group][block]


# Scheduler Output

@dataclass
class NewParticleGroupData:
    """New particle group to be registered."""
    group_id: str
    particle_req_ids: list[str]
    prompt_token_ids: list[int]
    num_computed_tokens: int
    prefix_block_ids: tuple[list[int], ...]        # [kv_group][block]
    decode_block_ids: list[tuple[list[int], ...]]  # [particle][kv_group][block]
    seed_token_ids: torch.Tensor   # [P]
    kv_slots: torch.Tensor         # [num_kv_groups, P, gamma+1]
    seq_lens: torch.Tensor         # [P]
    gamma: int
    temperature: float


@dataclass
class SMCGroupBatch:
    """Ongoing draft cycle — group already registered."""
    group_id: str
    seed_token_ids: torch.Tensor   # [P]
    kv_slots: torch.Tensor         # [num_kv_groups, P, gamma+1]
    seq_lens: torch.Tensor         # [P]
    gamma: int
    temperature: float


@dataclass
class SMCSchedulerOutput(SchedulerOutput):
    new_particle_groups: list[NewParticleGroupData] = field(default_factory=list)
    ongoing_smc_groups: list[SMCGroupBatch] = field(default_factory=list)


class SMCVLLMScheduler(Scheduler):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        spec = self.vllm_config.speculative_config
        assert spec is not None, "SMCVLLMScheduler requires speculative_config"
        self.smc_n_particles: int = spec.smc_n_particles
        self.smc_gamma: int = spec.num_speculative_tokens
        self.smc_temperature: float = spec.smc_temperature
        self.smc_groups: dict[str, SMCParticleGroupState] = {}  # group_id -> state


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
        num_computed_tokens = parent_req.num_computed_tokens
        prompt_token_ids = list(parent_req.prompt_token_ids)

        particle_req_ids = [
            f"{parent_req_id}__p{i}__{uuid.uuid4().hex[:8]}"
            for i in range(n_particles)
        ]

        n_decode_blocks = math.ceil((gamma + 1) / self.block_size)

        # Fork blocks: touches prefix N times and allocates decode blocks per particle
        # frees parent's ownership of prefix blocks.
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
            prefix_block_ids for _ in range(num_kv_groups)
        )
        decode_block_ids: list[tuple[list[int], ...]] = [
            tuple(dec for _ in range(num_kv_groups))
            for dec in decode_block_ids_flat
        ]

        kv_slots = self._compute_kv_slots(
            decode_block_ids=decode_block_ids,
            num_computed_tokens=num_computed_tokens,
            n_particles=n_particles,
            num_kv_groups=num_kv_groups,
            gamma=gamma,
        )

        seed_token_ids_tensor = torch.full(
            (n_particles,), seed_token_id, dtype=torch.long
        )
        seq_lens = torch.full(
            (n_particles,), num_computed_tokens, dtype=torch.long
        )

        group_id = parent_req_id
        state = SMCParticleGroupState(
            parent_req=parent_req,
            n_particles=n_particles,
            num_computed_tokens=num_computed_tokens,
            seed_token_ids=[seed_token_id] * n_particles,
            shared_prefix_block_ids=shared_prefix_block_ids,
            decode_block_ids=decode_block_ids,
        )
        self.smc_groups[group_id] = state

        return NewParticleGroupData(
            group_id=group_id,
            particle_req_ids=particle_req_ids,
            prompt_token_ids=prompt_token_ids,
            num_computed_tokens=num_computed_tokens,
            prefix_block_ids=shared_prefix_block_ids,
            decode_block_ids=decode_block_ids,
            seed_token_ids=seed_token_ids_tensor,
            kv_slots=kv_slots,
            seq_lens=seq_lens,
            gamma=gamma,
            temperature=temperature,
        )

    def update_particle_group(
        self,
        group_id: str,
        new_seed_token_ids: list[int],
        gamma: int,
        temperature: float,
    ) -> SMCGroupBatch:
        """Build an SMCGroupBatch for an already-registered group.

        Called each scheduling step for groups that have completed their
        previous draft cycle and are ready for the next one.
        """
        state = self.smc_groups[group_id]
        state.seed_token_ids = new_seed_token_ids
        state.num_computed_tokens += gamma + 1

        kv_slots = self._compute_kv_slots(
            decode_block_ids=state.decode_block_ids,
            num_computed_tokens=state.num_computed_tokens,
            n_particles=state.n_particles,
            num_kv_groups=len(state.shared_prefix_block_ids),
            gamma=gamma,
        )
        seq_lens = torch.full(
            (state.n_particles,), state.num_computed_tokens, dtype=torch.long
        )
        seed_tensor = torch.tensor(new_seed_token_ids, dtype=torch.long)

        return SMCGroupBatch(
            group_id=group_id,
            seed_token_ids=seed_tensor,
            kv_slots=kv_slots,
            seq_lens=seq_lens,
            gamma=gamma,
            temperature=temperature,
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

    def process_decode_result(
        self,
        group_id: str,
        draft_token_ids: torch.Tensor,   # [N, gamma+1] — tokens drafted this cycle
        draft_log_probs: torch.Tensor,   # [N, gamma+1, V] — draft log-probs
        target_log_probs: torch.Tensor,  # [N, gamma+1, V] — target log-probs
        gamma: int,
        temperature: float,
    ) -> SMCGroupBatch | None:
        """Called by the engine after a draft+verify cycle completes.

        Responsibilities:
          1. Resample: compute importance weights from draft/target log-prob
             ratio, resample particle indices.
          2. Update: call update_particle_group with new seed tokens and
             updated num_computed_tokens.
          3. Drain: if the group has reached its generation budget or all
             particles have EOS, call finish_particle_group and return None.

        Returns an SMCGroupBatch for the next cycle, or None if the group
        is finished.
        """
        raise NotImplementedError

    def schedule(self) -> SMCSchedulerOutput:
        pending_fork: list[Request] = []
        for req in list(self.running):
            # TODO: double check this logic
            # Detect requests that completed prefill last cycle: 
            # len(output_token_ids) == 1 and not in smc_groups.
            if (len(req.output_token_ids) == 1
                    and req.request_id not in self.smc_groups):
                  # Remove them from self.running before calling super.schedule 
                  # so the base scheduler does not schedule them for normal decode.
                self.running.remove(req)
                pending_fork.append(req)

        # Prefill request flows through vllm's base scheduler and execute_model implementation
        base_output = super().schedule()

        # Allocate KV blocks for N particles after prefill completes.
        new_particle_groups: list[NewParticleGroupData] = []
        for req in pending_fork:
            new_group = self.process_prefill_result(
                parent_req_id=req.request_id,
                seed_token_id=req.output_token_ids[0],
                n_particles=self.smc_n_particles,
                gamma=self.smc_gamma,
                temperature=self.smc_temperature,
            )
            new_particle_groups.append(new_group)

        # Ongoing SMC groups: out of scope until process_decode_result is
        # implemented (requires verify + resample output).
        ongoing_smc_groups: list[SMCGroupBatch] = []

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

    def _compute_kv_slots(
        self,
        decode_block_ids: list[tuple[list[int], ...]],
        num_computed_tokens: int,
        n_particles: int,
        num_kv_groups: int,
        gamma: int,
    ) -> torch.Tensor:
        """Compute kv_slots tensor shaped [num_kv_groups, P, gamma+1].

        slot_id[g, p, s] = decode_block_ids[p][g][block_index] * block_size
                           + block_offset
        where position = num_computed_tokens + s.
        """
        block_size = self.block_size
        slots = torch.zeros(
            (num_kv_groups, n_particles, gamma + 1), dtype=torch.long
        )
        for s in range(gamma + 1):
            position = num_computed_tokens + s
            block_index = position // block_size
            block_offset = position % block_size
            for p in range(n_particles):
                for g in range(num_kv_groups):
                    block_id = decode_block_ids[p][g][block_index]
                    slots[g, p, s] = block_id * block_size + block_offset
        return slots
