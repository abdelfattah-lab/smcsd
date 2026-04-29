from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata, build_slot_mappings_by_layer

from smcsd.vllm_backend.scheduler import (
    NewParticleGroupData,
    SMCGroupBatch,
    SMCSchedulerOutput,
)

if TYPE_CHECKING:
    from vllm.v1.outputs import ModelRunnerOutput


@dataclass
class ParticleGroup:
    """Runner-side handle for one SMC particle group."""
    group_id: str
    particle_rows: list[int]   # row indices into RequestState / BlockTables
    n_particles: int


class SMCGPUModelRunner(GPUModelRunner):
    """Extends GPUModelRunner with SMC particle-group awareness.

    Adds particle lifecycle and the draft+verify cycle.
    Normal prefill requests still flow through super().execute_model().
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.particle_groups: dict[str, ParticleGroup] = {}  # group_id -> ParticleGroup

    def add_particle_group(
        self,
        group_id: str,
        particle_req_ids: list[str],
        prompt_token_ids: list[int],
        num_computed_tokens: int,
        prefix_block_ids: tuple[list[int], ...],          # [kv_group][block]
        decode_block_ids: list[tuple[list[int], ...]],    # [particle][kv_group][block]
    ) -> None:
        """Register N particles into RequestState and BlockTables.

        For each particle:
          1. Claim a free row via req_states.add_request (pops free_indices).
          2. Write block table: prefix_block_ids[g] + decode_block_ids[i][g]
             for each kv group g.
        Flushes staged writes to GPU after all particles are registered.
        """
        prompt_len = len(prompt_token_ids)
        particle_rows: list[int] = []

        for i, p_id in enumerate(particle_req_ids):
            self.req_states.add_request(
                req_id=p_id,
                prompt_len=prompt_len,
                all_token_ids=prompt_token_ids,
                num_computed_tokens=num_computed_tokens,
            )
            req_index = self.req_states.req_id_to_index[p_id]
            particle_rows.append(req_index)

            combined_block_ids = tuple(
                prefix_block_ids[g] + decode_block_ids[i][g]
                for g in range(len(prefix_block_ids))
            )
            self.block_tables.append_block_ids(req_index, combined_block_ids, overwrite=True)

        self.req_states.apply_staged_writes()

        self.particle_groups[group_id] = ParticleGroup(
            group_id=group_id,
            particle_rows=particle_rows,
            n_particles=len(particle_req_ids),
        )

    def remove_particle_group(self, group_id: str, particle_req_ids: list[str]) -> None:
        """Return all particle rows to req_states.free_indices."""
        self.particle_groups.pop(group_id)
        for p_id in particle_req_ids:
            self.req_states.remove_request(p_id)

    def smc_draft_cycle(
        self,
        particle_rows: torch.Tensor,   # [N] — row indices
        seed_token_ids: torch.Tensor,  # [N] — last accepted token per particle
        kv_slots: torch.Tensor,        # [num_kv_groups, N, γ+1] — write slots
        seq_lens: torch.Tensor,        # [N] — num_computed_tokens at cycle start
        gamma: int,
        temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run gamma+1 autoregressive draft steps for all N particles.

        Returns:
            draft_token_ids  [N, gamma+1]    — sampled token at each step
            draft_log_probs  [N, gamma+1, V] — log-softmax logits at each step
        """
        P = particle_rows.shape[0]
        block_tables = self._gather_block_tables(particle_rows)  # once before loop

        draft_token_ids = torch.zeros(P, gamma + 1, dtype=torch.int32, device=self.device)
        draft_log_probs = torch.zeros(P, gamma + 1, self.vocab_size, dtype=torch.float32, device=self.device)
        draft_token_ids[:, 0] = seed_token_ids
        seq_lens_cur = seq_lens.clone()

        for step in range(gamma + 1):
            input_ids = draft_token_ids[:, step]   # [N]
            positions  = seq_lens_cur               # [N]
            slot_ids   = kv_slots[:, :, step]       # [num_kv_groups, N]

            attn_metadata = self._build_draft_attn_metadata(
                block_tables, seq_lens_cur + 1, slot_ids
            )
            slot_mappings_by_layer = build_slot_mappings_by_layer(
                slot_ids, self.kv_cache_config
            )

            with set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=P,
                slot_mapping=slot_mappings_by_layer,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                batch_descriptor=BatchDescriptor(num_tokens=P, has_lora=False),
            ):
                hidden_states = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=None,
                    inputs_embeds=None,
                )  # [P, hidden_dim]

            logits = self.model.compute_logits(hidden_states)   # [N, vocab_size]
            log_probs = torch.log_softmax(logits / temperature, dim=-1)  # [N, V]
            draft_log_probs[:, step] = log_probs

            if step < gamma:
                if temperature > 0:
                    next_tokens = torch.multinomial(log_probs.exp(), num_samples=1).squeeze(-1)
                else:
                    next_tokens = logits.argmax(dim=-1)
                draft_token_ids[:, step + 1] = next_tokens

            seq_lens_cur = seq_lens_cur + 1

        return draft_token_ids, draft_log_probs

    def smc_target_score(
        self,
        particle_rows: torch.Tensor,    # [N]
        draft_token_ids: torch.Tensor,  # [N, gamma+1]
        seq_lens: torch.Tensor,         # [N] — num_computed_tokens at draft start
        kv_slots: torch.Tensor,         # [num_kv_groups, N, gamma+1]
    ) -> torch.Tensor:
        """Score all draft tokens with the target model in one batched pass.

        On single GPU the target model is the same model used for drafting;
        the KV written during smc_draft_cycle is reused here.

        Returns:
            target_log_probs  [P, gamma+1, V]
        """
        raise NotImplementedError
    
    def _gather_block_tables(
        self, particle_rows: torch.Tensor  # [N]
    ) -> tuple[torch.Tensor, ...]:
        """Return dense block tables for the given particle row indices.

        Uses BlockTables.gather_block_tables with particle_rows as idx_mapping.
        Returns tuple of [P, max_blocks] tensors, one per KV cache group.
        """
        return self.block_tables.gather_block_tables(
            particle_rows, num_reqs_padded=particle_rows.shape[0]
        )

    def _build_draft_attn_metadata(
        self,
        block_tables: tuple[torch.Tensor, ...],  # [N, max_blocks] per kv-group
        seq_lens: torch.Tensor,                  # [N] — context length seen by attn
        slot_ids: torch.Tensor,                  # [num_kv_groups, n]
    ) -> dict[str, Any]:
        """Build attention metadata for one decode step over N particles.

        Each particle contributes exactly 1 query token (pure decode).
        seq_lens is already incremented to include the token being written.
        """
        N = seq_lens.shape[0]
        # Each particle has 1 query token: query_start_loc = [0, 1, 2, ..., N]
        query_start_loc = torch.arange(N + 1, dtype=torch.int32, device=self.device)
        query_start_loc_cpu = torch.arange(N + 1, dtype=torch.int32)
        return build_attn_metadata(
            attn_groups=self.attn_groups,
            num_reqs=N,
            num_tokens=N,
            query_start_loc_gpu=query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=1,
            seq_lens=seq_lens,
            max_seq_len=self.max_model_len,
            block_tables=list(block_tables),
            slot_mappings=slot_ids,
            kv_cache_config=self.kv_cache_config,
        )

    def _run_draft_and_verify(self, batch: SMCGroupBatch) -> None:
        """Look up particle rows and run one draft + verify cycle for a group."""
        particle_rows = torch.tensor(
            self.particle_groups[batch.group_id].particle_rows,
            dtype=torch.int32,
            device=self.device,
        )
        self.smc_draft_cycle(
            particle_rows=particle_rows,
            seed_token_ids=batch.seed_token_ids,
            kv_slots=batch.kv_slots,
            seq_lens=batch.seq_lens,
            gamma=batch.gamma,
            temperature=batch.temperature,
        )
        # TODO: target verify

    def execute_model(
        self,
        scheduler_output: SMCSchedulerOutput,
    ) -> ModelRunnerOutput:
        # Prefill parent requests: delegated to base runner
        output = super().execute_model(scheduler_output)

        # Newly forked groups: register particle rows
        for new_group in scheduler_output.new_particle_groups:
            self.add_particle_group(
                group_id=new_group.group_id,
                particle_req_ids=new_group.particle_req_ids,
                prompt_token_ids=new_group.prompt_token_ids,
                num_computed_tokens=new_group.num_computed_tokens,
                prefix_block_ids=new_group.prefix_block_ids,
                decode_block_ids=new_group.decode_block_ids,
            )
            self._run_draft_and_verify(new_group)

        # Ongoing groups: run draft + verify.
        for batch in scheduler_output.ongoing_smc_groups:
            self._run_draft_and_verify(batch)

        return output
