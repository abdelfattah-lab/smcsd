from __future__ import annotations

import copy
from dataclasses import dataclass
from types import SimpleNamespace
from pprint import pformat
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn

from vllm.config.compilation import CUDAGraphMode, CompilationMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.model_executor.model_loader import get_model
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.config import get_config as get_hf_config, get_hf_text_config
from vllm.sequence import IntermediateTensors
from vllm.v1.core.kv_cache_utils import (
    get_kv_cache_config_from_groups,
    get_kv_cache_groups,
)
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.attn_utils import (
    build_slot_mappings_by_layer,
    get_kv_cache_spec,
    init_attn_backend,
    init_kv_cache,
)
from vllm.v1.worker.gpu.input_batch import InputBatch, post_update
from vllm.v1.worker.gpu.sample.sampler import Sampler

from smcsd.vllm_backend.scheduler import (
    NewParticleGroupData,
    SMCGroupBatch,
    SMCSchedulerOutput,
)

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput

from smcsd.vllm_backend.outputs import SMCModelRunnerOutput

if TYPE_CHECKING:
    pass


@dataclass
class ParticleGroup:
    """Runner-side handle for one SMC particle group."""
    group_id: str
    particle_rows: list[int]   # row indices into RequestState / BlockTables
    particle_req_ids: list[str]
    n_particles: int
    debug_dumped: bool = False


class SMCGPUModelRunner(GPUModelRunner):
    """Extends GPUModelRunner with SMC particle-group awareness.

    Adds particle lifecycle and the draft+verify cycle.
    Normal prefill requests still flow through super().execute_model().
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.draft_model: nn.Module
        self.draft_kv_cache_config = None
        self.draft_attn_backends: dict[str, Any] = {}
        self.draft_attn_groups: list[list[Any]] = []
        self.draft_kv_caches: list[torch.Tensor] = []
        self.particle_groups: dict[str, ParticleGroup] = {}  # group_id -> ParticleGroup
        self.target_sampler: Sampler | None = None
        self._draft_attn_metadata_debugged = False

    def load_model(self, *args, **kwargs) -> None:
        super().load_model(*args, **kwargs)
        smc = self.vllm_config.smc_config
        assert smc is not None, "SMCGPUModelRunner requires smc_config"
        draft_model_config = copy.copy(self.vllm_config.model_config)
        draft_model_config.model = smc.draft_model_path
        draft_hf_config = get_hf_config(
            smc.draft_model_path,
            trust_remote_code=draft_model_config.trust_remote_code,
            revision=draft_model_config.revision,
            code_revision=draft_model_config.code_revision,
            config_format=draft_model_config.config_format,
        )
        draft_model_config.hf_config = draft_hf_config
        draft_model_config.hf_text_config = get_hf_text_config(draft_hf_config)
        draft_model_config.attention_chunk_size = getattr(
            draft_model_config.hf_text_config, "attention_chunk_size", None
        )
        draft_model_config.model_arch_config = draft_model_config.get_model_arch_config()

        draft_vllm_config = copy.copy(self.vllm_config)
        draft_vllm_config.model_config = draft_model_config
        draft_compilation_config = copy.copy(self.vllm_config.compilation_config)
        draft_compilation_config.static_forward_context = {}
        draft_compilation_config.mode = CompilationMode.NONE
        draft_compilation_config.cudagraph_mode = CUDAGraphMode.NONE
        draft_vllm_config.compilation_config = draft_compilation_config

        draft_model = get_model(vllm_config=draft_vllm_config)
        draft_model.eval()
        self.draft_model = draft_model
        self.draft_vllm_config = draft_vllm_config

    def initialize_kv_cache(self, *args, **kwargs) -> None:
        super().initialize_kv_cache(*args, **kwargs)
        # Allocate a draft-side KV arena from the draft model's own KV spec.
        draft_kv_cache_spec = get_kv_cache_spec(self.draft_vllm_config)
        draft_kv_cache_groups = get_kv_cache_groups(
            self.draft_vllm_config, draft_kv_cache_spec
        )
        target_kv_bytes = sum(
            kv_cache_tensor.size for kv_cache_tensor in self.kv_cache_config.kv_cache_tensors
        )
        draft_kv_cache_config = get_kv_cache_config_from_groups(
            self.draft_vllm_config,
            draft_kv_cache_groups,
            target_kv_bytes,
            suppress_log=True,
        )
        if draft_kv_cache_config.num_blocks < self.kv_cache_config.num_blocks:
            raise RuntimeError(
                "Draft KV cache config has fewer blocks than target "
                f"({draft_kv_cache_config.num_blocks} < {self.kv_cache_config.num_blocks})."
            )
        if len(draft_kv_cache_config.kv_cache_groups) != len(self.kv_cache_config.kv_cache_groups):
            raise RuntimeError(
                "Draft KV cache groups do not match target KV cache groups "
                f"({len(draft_kv_cache_config.kv_cache_groups)} != {len(self.kv_cache_config.kv_cache_groups)})."
            )
        for draft_group, target_group in zip(
            draft_kv_cache_config.kv_cache_groups,
            self.kv_cache_config.kv_cache_groups,
        ):
            if draft_group.kv_cache_spec.block_size != target_group.kv_cache_spec.block_size:
                raise RuntimeError(
                    "Draft and target block sizes must match for shared logical block ids "
                    f"({draft_group.kv_cache_spec.block_size} != {target_group.kv_cache_spec.block_size})."
                )

        draft_attn_backends, draft_attn_groups, _ = init_attn_backend(
            draft_kv_cache_config,
            self.draft_vllm_config,
            self.device,
        )
        draft_ctx = self.draft_vllm_config.compilation_config.static_forward_context
        init_kv_cache(
            self.draft_kv_caches,
            draft_ctx,
            draft_kv_cache_config,
            draft_attn_backends,
            self.device,
            self.cache_config.cache_dtype,
        )
        self.draft_kv_cache_config = draft_kv_cache_config
        self.draft_attn_backends = draft_attn_backends
        self.draft_attn_groups = draft_attn_groups

        if self.is_last_pp_rank and not self.is_pooling_model:
            self.target_sampler = Sampler(
                max_num_reqs=self.max_num_reqs,
                vocab_size=self.vocab_size,
                device=self.device,
                req_states=self.req_states,
                logprobs_mode=self.model_config.logprobs_mode,
                num_speculative_tokens=self.num_speculative_steps + 1,
            )

    def register_particle_group(
        self,
        group_id: str,
        particle_req_ids: list[str],
        prompt_token_ids: list[int],
        num_computed_tokens: int,
        sampling_params: SamplingParams | None,
        prefix_block_ids: tuple[list[int], ...],          # [kv_group][block]
        decode_block_ids: list[tuple[list[int], ...]],    # [particle][kv_group][block]
        temperature: float,
    ) -> None:
        """Register N particles into RequestState and BlockTables."""
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
            if self.is_last_pp_rank and self.sampler is not None:
                draft_sampling_params = copy.copy(sampling_params)
                draft_sampling_params.temperature = temperature
                self.sampler.add_request(
                    req_index, prompt_len, draft_sampling_params
                )
            if self.is_last_pp_rank and self.target_sampler is not None:
                self.target_sampler.add_request(
                    req_index, prompt_len, sampling_params
                )

        self.req_states.apply_staged_writes()
        self.block_tables.apply_staged_writes()
        if self.sampler is not None:
            self.sampler.apply_staged_writes()
        if self.target_sampler is not None:
            self.target_sampler.apply_staged_writes()

        # if particle_req_ids:
        #     debug_particles = min(2, len(particle_req_ids))
        #     print(f"[smc-debug] register_particle_group group_id={group_id} prompt_len={prompt_len} num_computed_tokens={num_computed_tokens}")
        #     print(f"[smc-debug] shared_prefix_block_ids={tuple(list(x) for x in prefix_block_ids)}")
        #     for i in range(debug_particles):
        #         combined_block_ids = tuple(
        #             list(prefix_block_ids[g]) + list(decode_block_ids[i][g])
        #             for g in range(len(prefix_block_ids))
        #         )
        #         print(
        #             f"[smc-debug] particle[{i}] req_id={particle_req_ids[i]} "
        #             f"row={particle_rows[i]} combined_block_ids={combined_block_ids}"
        #         )

        self.particle_groups[group_id] = ParticleGroup(
            group_id=group_id,
            particle_rows=particle_rows,
            particle_req_ids=list(particle_req_ids),
            n_particles=len(particle_req_ids),
        )

    def _run_batched_draft_prefill(
        self,
        groups: list[NewParticleGroupData],
    ) -> None:
        """Prefill draft KV for all new particle groups."""
        assert self.draft_kv_cache_config is not None

        all_particle_rows: list[int] = []
        all_input_ids: list[int] = []
        all_positions: list[int] = []
        all_seq_lens: list[int] = []
        query_start_loc_list: list[int] = [0]

        for group in groups:
            particle_rows = self.particle_groups[group.group_id].particle_rows
            N = len(particle_rows)
            L = len(group.prompt_token_ids)
            all_particle_rows.extend(particle_rows)
            all_input_ids.extend(group.prompt_token_ids * N)
            all_positions.extend(list(range(L)) * N)
            all_seq_lens.extend([L] * N)
            for _ in range(N):
                query_start_loc_list.append(query_start_loc_list[-1] + L)

        total_N = len(all_particle_rows)
        total_tokens = query_start_loc_list[-1]

        particle_rows_t = torch.tensor(all_particle_rows, dtype=torch.int32, device=self.device)
        input_ids = torch.tensor(all_input_ids, dtype=torch.int32, device=self.device)
        positions = torch.tensor(all_positions, dtype=torch.long, device=self.device)
        seq_lens = torch.tensor(all_seq_lens, dtype=torch.int32, device=self.device)
        query_start_loc = torch.tensor(query_start_loc_list, dtype=torch.int32, device=self.device)
        query_start_loc_np = np.array(query_start_loc_list, dtype=np.int32)

        block_tables = self._gather_block_tables(particle_rows_t)
        slot_ids = self.block_tables.compute_slot_mappings(
            particle_rows_t,
            query_start_loc,
            positions,
            num_tokens_padded=total_tokens,
        )

        input_batch = SimpleNamespace(
            num_reqs=total_N,
            num_tokens=total_tokens,
            num_reqs_after_padding=total_N,
            num_tokens_after_padding=total_tokens,
            query_start_loc=query_start_loc,
            query_start_loc_np=query_start_loc_np,
            num_scheduled_tokens=np.array(all_seq_lens, dtype=np.int32),
            seq_lens=seq_lens,
            seq_lens_cpu_upper_bound=seq_lens.cpu(),
            dcp_local_seq_lens=None,
            positions=positions,
        )
        attn_metadata = self.model_state.prepare_attn(
            input_batch,
            CUDAGraphMode.NONE,
            block_tables,
            slot_ids,
            self.draft_attn_groups,
            self.draft_kv_cache_config,
        )
        slot_mappings_by_layer = build_slot_mappings_by_layer(
            slot_ids, self.draft_kv_cache_config
        )

        with set_forward_context(
            attn_metadata,
            self.draft_vllm_config,
            num_tokens=total_tokens,
            slot_mapping=slot_mappings_by_layer,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            batch_descriptor=BatchDescriptor(num_tokens=total_tokens, has_lora=False),
        ):
            self.draft_model(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=None,
                inputs_embeds=None,
            )

    def _run_batched_target_prefill_tail(
        self,
        groups: list[NewParticleGroupData],
    ) -> None:
        """Fill target KV for the prompt tail per particle."""
        block_size = self.kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size

        all_particle_rows: list[int] = []
        all_input_ids: list[int] = []
        all_positions: list[int] = []
        all_seq_lens: list[int] = []
        all_num_scheduled: list[int] = []
        query_start_loc_list: list[int] = [0]

        for group in groups:
            L = group.num_computed_tokens
            num_shared_tokens = (L // block_size) * block_size
            tail_len = L - num_shared_tokens
            if tail_len == 0:
                continue

            particle_rows = self.particle_groups[group.group_id].particle_rows
            N = len(particle_rows)
            tail_tokens = group.prompt_token_ids[num_shared_tokens:]
            tail_positions = list(range(num_shared_tokens, L))

            all_particle_rows.extend(particle_rows)
            all_input_ids.extend(tail_tokens * N)
            all_positions.extend(tail_positions * N)
            all_seq_lens.extend([L] * N)
            all_num_scheduled.extend([tail_len] * N)
            for _ in range(N):
                query_start_loc_list.append(query_start_loc_list[-1] + tail_len)

        if not all_particle_rows:
            return

        total_N = len(all_particle_rows)
        total_tokens = query_start_loc_list[-1]

        particle_rows_t = torch.tensor(all_particle_rows, dtype=torch.int32, device=self.device)
        input_ids = torch.tensor(all_input_ids, dtype=torch.int32, device=self.device)
        positions = torch.tensor(all_positions, dtype=torch.long, device=self.device)
        seq_lens = torch.tensor(all_seq_lens, dtype=torch.int32, device=self.device)
        query_start_loc = torch.tensor(query_start_loc_list, dtype=torch.int32, device=self.device)
        query_start_loc_np = np.array(query_start_loc_list, dtype=np.int32)

        block_tables = self._gather_block_tables(particle_rows_t)
        slot_ids = self.block_tables.compute_slot_mappings(
            particle_rows_t,
            query_start_loc,
            positions,
            num_tokens_padded=total_tokens,
        )

        input_batch = SimpleNamespace(
            num_reqs=total_N,
            num_tokens=total_tokens,
            num_reqs_after_padding=total_N,
            num_tokens_after_padding=total_tokens,
            query_start_loc=query_start_loc,
            query_start_loc_np=query_start_loc_np,
            num_scheduled_tokens=np.array(all_num_scheduled, dtype=np.int32),
            seq_lens=seq_lens,
            seq_lens_cpu_upper_bound=seq_lens.cpu(),
            dcp_local_seq_lens=None,
            positions=positions,
        )
        attn_metadata = self.model_state.prepare_attn(
            input_batch,
            CUDAGraphMode.NONE,
            block_tables,
            slot_ids,
            self.attn_groups,
            self.kv_cache_config,
        )
        slot_mappings_by_layer = build_slot_mappings_by_layer(slot_ids, self.kv_cache_config)

        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=total_tokens,
            slot_mapping=slot_mappings_by_layer,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            batch_descriptor=BatchDescriptor(num_tokens=total_tokens, has_lora=False),
        ):
            self.model(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=None,
                inputs_embeds=None,
            )

    def remove_particle_group(self, group_id: str) -> None:
        """Return all particle rows to req_states.free_indices."""
        group = self.particle_groups.pop(group_id)
        for p_id in group.particle_req_ids:
            self.req_states.remove_request(p_id)

    @torch.inference_mode()
    def _run_batched_target_verify(
        self,
        batches: list[SMCGroupBatch | NewParticleGroupData],
        draft_results: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """Run target model on [seed, t1..t_gamma] per particle, return bonus tokens.

        Fills target KV at the draft positions and samples one bonus token per
        particle from the target logit at the last draft position (L+gamma).

        Returns:
            dict group_id -> bonus_tokens [A_i]  (int32 on device)
        """
        gamma = batches[0].gamma

        # Collect active particles across all groups
        group_slices: dict[str, tuple[int, int]] = {}
        all_rows: list[int] = []
        all_start_seq_lens: list[int] = []
        all_draft_ids: list[torch.Tensor] = []

        offset = 0
        for batch in batches:
            group = self.particle_groups[batch.group_id]
            finished = getattr(batch, "particle_finished", None) or [False] * len(group.particle_rows)
            active_local = [i for i, f in enumerate(finished) if not f]
            A_i = len(active_local)
            if A_i > 0:
                seq_lens_i = batch.seq_lens[active_local]
                all_rows.extend(group.particle_rows[i] for i in active_local)
                all_start_seq_lens.extend(seq_lens_i.tolist())
                all_draft_ids.append(draft_results[batch.group_id][0])  # [A_i, gamma+1]
            group_slices[batch.group_id] = (offset, offset + A_i)
            offset += A_i

        A_total = offset
        if A_total == 0:
            return {}

        rows = torch.tensor(all_rows, dtype=torch.int32, device=self.device)
        # L_i: seq_len at the start of this draft cycle for each particle
        start_seq_lens = torch.tensor(all_start_seq_lens, dtype=torch.int32, device=self.device)

        # Input ids: [seed, t1..t_gamma] per particle, flattened
        # draft_results[group_id][0] shape: [A_i, gamma+1], [:, 0] is seed
        draft_ids = torch.cat(all_draft_ids, dim=0)               # [A_total, gamma+1]
        input_ids = draft_ids.reshape(-1).to(torch.int32)         # [A_total * (gamma+1)]

        # Positions: [L_i, L_i+1, ..., L_i+gamma] per particle.
        step_offsets = torch.arange(gamma + 1, dtype=torch.long, device=self.device)
        positions = (
            start_seq_lens.long().unsqueeze(1) + step_offsets.unsqueeze(0)
        ).reshape(-1)                                              # [A_total * (gamma+1)]

        # After writing all gamma+1 new tokens the total KV length is L_i + gamma + 1.
        # Target KV at 0..L_i-1 is already present in shared prefix blocks.
        seq_lens_full = start_seq_lens + gamma + 1                # [A_total]

        total_tokens = A_total * (gamma + 1)
        query_start_loc = torch.arange(
            0, total_tokens + 1, gamma + 1, dtype=torch.int32, device=self.device
        )                                                          # [A_total + 1]
        query_start_loc_np = query_start_loc.cpu().numpy()

        # Last token index (position L_i+gamma) in the flat token layout.
        last_token_indices = query_start_loc[1:].long() - 1       # [A_total]

        block_tables = self._gather_block_tables(rows)
        slot_ids = self.block_tables.compute_slot_mappings(
            rows,
            query_start_loc,
            positions,
            num_tokens_padded=total_tokens,
        )

        input_batch_ns = SimpleNamespace(
            num_reqs=A_total,
            num_tokens=total_tokens,
            num_reqs_after_padding=A_total,
            num_tokens_after_padding=total_tokens,
            query_start_loc=query_start_loc,
            query_start_loc_np=query_start_loc_np,
            num_scheduled_tokens=np.full(A_total, gamma + 1, dtype=np.int32),
            seq_lens=seq_lens_full,
            seq_lens_cpu_upper_bound=seq_lens_full.cpu(),
            dcp_local_seq_lens=None,
            positions=positions,
        )

        attn_metadata = self.model_state.prepare_attn(
            input_batch_ns,
            CUDAGraphMode.NONE,
            block_tables,
            slot_ids,
            self.attn_groups,
            self.kv_cache_config,
        )
        slot_mappings_by_layer = build_slot_mappings_by_layer(slot_ids, self.kv_cache_config)

        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=total_tokens,
            slot_mapping=slot_mappings_by_layer,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            batch_descriptor=BatchDescriptor(num_tokens=total_tokens, has_lora=False),
        ):
            model_output = self.model(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=None,
                inputs_embeds=None,
            )

        # Support models that return (hidden_states, aux) tuples (e.g. Eagle3 target).
        hidden_states = model_output[0] if isinstance(model_output, tuple) else model_output

        # Logits at position L_i+gamma for each particle.
        logits = self.model.compute_logits(hidden_states[last_token_indices])  # [A_total, V]

        bonus_input_batch = self._build_draft_input_batch(
            particle_rows=rows,
            input_ids=input_ids[last_token_indices],
            positions=positions[last_token_indices],
            seq_lens=seq_lens_full,
            query_start_loc=torch.arange(A_total + 1, dtype=torch.int32, device=self.device),
            query_start_loc_np=np.arange(A_total + 1, dtype=np.int32),
        )
        sampler_output = self.target_sampler(logits, bonus_input_batch)
        bonus_tokens = sampler_output.sampled_token_ids.squeeze(-1).to(torch.int32)

        # Draft decode wrote next_seeds into last_sampled_tokens at step gamma;
        # overwrite with the correct bonus token.
        self.req_states.last_sampled_tokens[rows] = bonus_tokens.long().unsqueeze(-1)

        # Fix all_token_ids: replace the draft next_seed with the bonus token at L+gamma.
        positions_to_fix = (start_seq_lens + gamma).long()  # [A_total]
        self.req_states.all_token_ids.gpu[rows.long(), positions_to_fix] = bonus_tokens

        if self.sampler.penalties_state.output_bin_counts is not None:
            draft_next_seeds = torch.cat([
                draft_results[batch.group_id][2]
                for batch in batches
                if (group_slices[batch.group_id][1] > group_slices[batch.group_id][0])
            ])  # [A_total]
            bin_counts = self.sampler.penalties_state.output_bin_counts
            bin_counts[rows.long(), draft_next_seeds.long()] -= 1
            bin_counts[rows.long(), bonus_tokens.long()] += 1

        return {
            batch.group_id: bonus_tokens[s:e]
            for batch in batches
            for s, e in [group_slices[batch.group_id]]
            if e > s
        }
    
    def _build_draft_input_batch(
        self,
        particle_rows: torch.Tensor,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        seq_lens: torch.Tensor,
        query_start_loc: torch.Tensor,
        query_start_loc_np: np.ndarray,
    ) -> InputBatch:
        num_reqs = particle_rows.shape[0]
        idx_mapping_np = particle_rows.detach().cpu().numpy().astype(np.int32, copy=False)
        logits_indices = torch.arange(num_reqs, dtype=torch.int32, device=self.device)
        cu_num_logits = torch.arange(num_reqs + 1, dtype=torch.int32, device=self.device)
        cu_num_logits_np = np.arange(num_reqs + 1, dtype=np.int32)
        expanded_local_pos = torch.zeros(num_reqs, dtype=torch.int32, device=self.device)
        return InputBatch(
            req_ids=[f"smc_particle_{int(row)}" for row in idx_mapping_np],
            num_reqs=num_reqs,
            num_reqs_after_padding=num_reqs,
            idx_mapping=particle_rows,
            idx_mapping_np=idx_mapping_np,
            expanded_idx_mapping=particle_rows,
            expanded_local_pos=expanded_local_pos,
            num_scheduled_tokens=np.ones(num_reqs, dtype=np.int32),
            num_tokens=num_reqs,
            num_tokens_after_padding=num_reqs,
            num_draft_tokens=0,
            query_start_loc=query_start_loc,
            query_start_loc_np=query_start_loc_np,
            seq_lens=seq_lens,
            seq_lens_cpu_upper_bound=seq_lens.detach().cpu(),
            dcp_local_seq_lens=None,
            input_ids=input_ids,
            positions=positions,
            logits_indices=logits_indices,
            cu_num_logits=cu_num_logits,
            cu_num_logits_np=cu_num_logits_np,
            has_structured_output_reqs=False,
        )

    def _gather_block_tables(
        self, particle_rows: torch.Tensor  # [N]
    ) -> tuple[torch.Tensor, ...]:
        """Return dense block tables for the given particle row indices."""
        return self.block_tables.gather_block_tables(
            particle_rows, num_reqs_padded=particle_rows.shape[0]
        )

    def _build_draft_attn_metadata(
        self,
        input_batch: InputBatch,
        slot_ids: torch.Tensor,                  # [num_kv_groups, n]
    ) -> dict[str, Any]:
        """Build attention metadata for one decode step over N particles.
        seq_lens is already incremented to include the token being written.
        """
        N = input_batch.num_reqs
        block_tables = self._gather_block_tables(input_batch.idx_mapping)
        attn_metadata = self.model_state.prepare_attn(
            input_batch,
            CUDAGraphMode.NONE,
            block_tables,
            slot_ids,
            self.draft_attn_groups,
            self.draft_kv_cache_config,
        )
    
        # if not self._draft_attn_metadata_debugged and N >= 2:
        #     print("[smc-debug] attn_metadata input")
        #     print(f"[smc-debug] num_reqs={N} num_tokens={N} max_query_len=1")
        #     print(f"[smc-debug] query_start_loc_gpu={input_batch.query_start_loc.tolist()}")
        #     print(f"[smc-debug] query_start_loc_cpu={input_batch.query_start_loc_np.tolist()}")
        #     print(f"[smc-debug] seq_lens={input_batch.seq_lens.tolist()}")
        #     print(f"[smc-debug] positions={input_batch.positions.tolist()}")
        #     print(f"[smc-debug] slot_ids={slot_ids.cpu().tolist()}")
        #     for g, bt in enumerate(block_tables):
        #         rows_to_show = min(2, bt.shape[0])
        #         print(
        #             f"[smc-debug] block_table[{g}] shape={tuple(bt.shape)} "
        #             f"rows={bt[:rows_to_show].cpu().tolist()}"
        #         )
        #     print(f"[smc-debug] attn_metadata_keys={list(attn_metadata.keys())}")
        #     print("[smc-debug] attn_metadata effective")
        #     for layer_name, layer_metadata in attn_metadata.items():
        #         print(f"[smc-debug] layer={layer_name}")
        #         print(pformat(layer_metadata))
        #     self._draft_attn_metadata_debugged = True

        return attn_metadata

    def _run_batched_draft_decode(
        self,
        batches: list[SMCGroupBatch | NewParticleGroupData],
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Run gamma+1 draft decode steps for all groups in one batched forward pass per step.

        Returns:
            dict group_id -> (draft_token_ids [A_i, gamma+1],
                              draft_log_probs  [A_i, gamma+1, V],
                              next_seed_ids    [A_i])
            where A_i is the number of active (unfinished) particles for that group.
        """
        assert self.draft_kv_cache_config is not None
        gamma = batches[0].gamma

        # Collect active particles across all groups
        group_slices: dict[str, tuple[int, int]] = {}
        all_rows: list[int] = []
        all_seeds: list[int] = []
        all_seq_lens: list[int] = []

        offset = 0
        for batch in batches:
            group = self.particle_groups[batch.group_id]
            finished = getattr(batch, "particle_finished", None) or [False] * len(group.particle_rows)
            active_local = [i for i, f in enumerate(finished) if not f]
            A_i = len(active_local)
            if A_i > 0:
                seeds_i = batch.seed_token_ids[active_local]
                seq_lens_i = batch.seq_lens[active_local]
                all_rows.extend(group.particle_rows[i] for i in active_local)
                all_seeds.extend(seeds_i.tolist())
                all_seq_lens.extend(seq_lens_i.tolist())
            group_slices[batch.group_id] = (offset, offset + A_i)
            offset += A_i

        A_total = offset

        # Run gamma+1 decode steps over all A_total particles 
        rows = torch.tensor(all_rows, dtype=torch.int32, device=self.device)
        seq_lens_cur = torch.tensor(all_seq_lens, dtype=torch.int32, device=self.device)
        draft_ids = torch.zeros(A_total, gamma + 1, dtype=torch.int32, device=self.device)
        log_probs = torch.zeros(A_total, gamma + 1, self.vocab_size, dtype=torch.float32, device=self.device)
        draft_ids[:, 0] = torch.tensor(all_seeds, dtype=torch.int32, device=self.device)
        next_seeds = torch.zeros(A_total, dtype=torch.int32, device=self.device)

        for step in range(gamma + 1):
            input_ids = draft_ids[:, step]
            positions = seq_lens_cur.long()
            query_start_loc = torch.arange(A_total + 1, dtype=torch.int32, device=self.device)
            query_start_loc_np = np.arange(A_total + 1, dtype=np.int32)
            input_batch = self._build_draft_input_batch(
                particle_rows=rows,
                input_ids=input_ids,
                positions=positions,
                seq_lens=seq_lens_cur + 1,
                query_start_loc=query_start_loc,
                query_start_loc_np=query_start_loc_np,
            )
            slot_ids = self.block_tables.compute_slot_mappings(
                rows,
                query_start_loc,
                positions,
                num_tokens_padded=A_total,
            )
            attn_metadata = self._build_draft_attn_metadata(input_batch, slot_ids)
            slot_mappings_by_layer = build_slot_mappings_by_layer(
                slot_ids, self.draft_kv_cache_config
            )

            with set_forward_context(
                attn_metadata,
                self.draft_vllm_config,
                num_tokens=A_total,
                slot_mapping=slot_mappings_by_layer,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                batch_descriptor=BatchDescriptor(num_tokens=A_total, has_lora=False),
            ):
                hidden_states = self.draft_model(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=None,
                    inputs_embeds=None,
                )

            logits = self.draft_model.compute_logits(hidden_states)
            log_probs[:, step] = torch.log_softmax(logits, dim=-1)

            assert self.sampler is not None
            sampler_output = self.sampler(logits, input_batch)
            next_tokens = sampler_output.sampled_token_ids.squeeze(-1).to(torch.int32)
            num_sampled = sampler_output.num_sampled
            num_rejected = torch.zeros_like(num_sampled)
            post_update(
                rows,
                self.req_states.num_computed_tokens.gpu,
                self.req_states.last_sampled_tokens,
                self.sampler.penalties_state.output_bin_counts,
                sampler_output.sampled_token_ids,
                num_sampled,
                num_rejected,
                query_start_loc,
                self.req_states.all_token_ids.gpu,
                self.req_states.total_len.gpu,
            )
            self.req_states.num_computed_tokens_np[rows.cpu().numpy()] += 1

            if step < gamma:
                draft_ids[:, step + 1] = next_tokens
            else:
                next_seeds = next_tokens

            seq_lens_cur = seq_lens_cur + 1

        # Slice results back per group 
        results: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for batch in batches:
            s, e = group_slices[batch.group_id]
            results[batch.group_id] = (
                draft_ids[s:e],
                log_probs[s:e],
                next_seeds[s:e],
            )
        return results

    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        is_profile: bool = False,
    ) -> ModelRunnerOutput | IntermediateTensors | None:
        base_output = super().execute_model(
            scheduler_output, intermediate_tensors, dummy_run, skip_attn_for_dummy_run, is_profile
        )

        # Skip SMC logic during _dummy_run profiling pass 
        if not isinstance(scheduler_output, SMCSchedulerOutput) or not isinstance(base_output, ModelRunnerOutput):
            return base_output

        smc_draft_results: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

        # Register all new groups (req_states, block_tables, sampler).
        for new_group in scheduler_output.new_particle_groups:
            self.register_particle_group(
                group_id=new_group.group_id,
                particle_req_ids=new_group.particle_req_ids,
                prompt_token_ids=new_group.prompt_token_ids,
                num_computed_tokens=new_group.num_computed_tokens,
                sampling_params=new_group.sampling_params,
                prefix_block_ids=new_group.prefix_block_ids,
                decode_block_ids=new_group.decode_block_ids,
                temperature=new_group.temperature,
            )

        if scheduler_output.new_particle_groups:
            self._run_batched_draft_prefill(scheduler_output.new_particle_groups)
            self._run_batched_target_prefill_tail(scheduler_output.new_particle_groups)

        all_batches: list[SMCGroupBatch | NewParticleGroupData] = (
            list(scheduler_output.new_particle_groups)
            + list(scheduler_output.ongoing_smc_groups)
        )
        if all_batches:
            smc_draft_results = self._run_batched_draft_decode(all_batches)
            bonus_tokens_per_group = self._run_batched_target_verify(all_batches, smc_draft_results)
            for group_id, bonus in bonus_tokens_per_group.items():
                draft_ids, log_probs, _ = smc_draft_results[group_id]
                smc_draft_results[group_id] = (draft_ids, log_probs, bonus)

        # Free rows for any groups the scheduler finished this step.
        active_ids = (
            {g.group_id for g in scheduler_output.new_particle_groups}
            | {b.group_id for b in scheduler_output.ongoing_smc_groups}
        )
        for group_id in list(self.particle_groups):
            if group_id not in active_ids:
                self.remove_particle_group(group_id)

        return SMCModelRunnerOutput(
            req_ids=base_output.req_ids,
            req_id_to_index=base_output.req_id_to_index,
            sampled_token_ids=base_output.sampled_token_ids,
            logprobs=base_output.logprobs,
            prompt_logprobs_dict=base_output.prompt_logprobs_dict,
            pooler_output=base_output.pooler_output,
            kv_connector_output=base_output.kv_connector_output,
            ec_connector_output=base_output.ec_connector_output,
            num_nans_in_logits=base_output.num_nans_in_logits,
            cudagraph_stats=base_output.cudagraph_stats,
            smc_draft_results=smc_draft_results,
        )

    def shutdown(self) -> None:
        draft_ctx = self.draft_vllm_config.compilation_config.static_forward_context
        for draft_layer in draft_ctx.values():
            draft_layer.kv_cache = None
        draft_ctx.clear()
        self.draft_model = None
