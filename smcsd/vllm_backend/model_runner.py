from __future__ import annotations

import copy
import os
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
from vllm.v1.worker.gpu.model_states import init_model_state
from vllm.v1.worker.gpu.attn_utils import (
    build_slot_mappings_by_layer,
    get_kv_cache_spec,
    init_attn_backend,
    init_kv_cache,
)
from vllm.v1.worker.gpu.input_batch import InputBatch, post_update
from vllm.v1.worker.gpu.sample.sampler import Sampler

from smcsd.vllm_backend.scheduler import (
    COWBlockRepair,
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
    num_full_shared_blocks: int = 0  # prefix blocks shared by all particles; decode blocks start after
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
        self.draft_model_state = None
        self.particle_groups: dict[str, ParticleGroup] = {}  # group_id -> ParticleGroup
        self.target_sampler: Sampler | None = None
        self.log_weights: torch.Tensor  # allocated in initialize_kv_cache
        self._draft_attn_metadata_debugged = False

    @staticmethod
    def _copy_kv_blocks(kv_cache: torch.Tensor, block_ids: torch.Tensor) -> torch.Tensor:
        """Snapshot physical KV blocks across supported vLLM cache layouts."""
        if kv_cache.dim() >= 2 and kv_cache.shape[0] == 2:
            return kv_cache[:, block_ids].clone()
        return kv_cache[block_ids].clone()

    @staticmethod
    def _write_kv_blocks(
        kv_cache: torch.Tensor,
        block_ids: torch.Tensor,
        snapshot: torch.Tensor,
    ) -> None:
        """Write physical KV block snapshots across supported vLLM layouts."""
        if kv_cache.dim() >= 2 and kv_cache.shape[0] == 2:
            kv_cache[:, block_ids] = snapshot
        else:
            kv_cache[block_ids] = snapshot

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
        self.draft_model_state = init_model_state(
            draft_vllm_config, draft_model, None, self.device
        )

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

        self.log_weights = torch.zeros(
            self.max_num_reqs, dtype=torch.float32, device=self.device
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
        # Free parent request runner row
        self._remove_request(group_id)

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
                # Draft must be pure temperature-sampling so that
                # log_softmax(logits/T)[x_t] is the exact log q_draft.
                # Copying target SamplingParams and propagating top-p/k/min-p/
                # penalties would make the actual proposal a truncated/penalized
                # distribution that does not match the log_softmax formula used
                # in _run_batched_draft_decode, biasing importance weights.
                draft_sampling_params = SamplingParams(temperature=temperature)
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

        block_size = self.kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size
        num_full_shared_blocks = num_computed_tokens // block_size

        self.log_weights[particle_rows] = 0.0

        self.particle_groups[group_id] = ParticleGroup(
            group_id=group_id,
            particle_rows=particle_rows,
            particle_req_ids=list(particle_req_ids),
            n_particles=len(particle_req_ids),
            num_full_shared_blocks=num_full_shared_blocks,
        )

    def _run_batched_draft_prefill(
        self,
        groups: list[NewParticleGroupData],
    ) -> None:
        """Prefill draft KV for new particle groups.

        Full block-aligned prefix blocks are shared by every particle in a group. 
        The prompt tail lives in each particle's private decode blocks and is written per particle.
        """
        def run_prefill_chunk(
            particle_rows: list[int],
            input_ids_list: list[int],
            positions_list: list[int],
            seq_lens_list: list[int],
            num_scheduled_list: list[int],
            query_start_loc_list: list[int],
        ) -> None:
            if not particle_rows:
                return

            total_N = len(particle_rows)
            total_tokens = query_start_loc_list[-1]

            particle_rows_t = torch.tensor(
                particle_rows, dtype=torch.int32, device=self.device
            )
            input_ids = torch.tensor(
                input_ids_list, dtype=torch.int32, device=self.device
            )
            positions = torch.tensor(
                positions_list, dtype=torch.long, device=self.device
            )
            seq_lens = torch.tensor(
                seq_lens_list, dtype=torch.int32, device=self.device
            )
            query_start_loc = torch.tensor(
                query_start_loc_list, dtype=torch.int32, device=self.device
            )
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
                num_scheduled_tokens=np.array(num_scheduled_list, dtype=np.int32),
                seq_lens=seq_lens,
                seq_lens_cpu_upper_bound=seq_lens.cpu(),
                dcp_local_seq_lens=None,
                positions=positions,
            )
 
            attn_metadata = self.draft_model_state.prepare_attn(
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
                batch_descriptor=BatchDescriptor(
                    num_tokens=total_tokens, has_lora=False
                ),
            ):
                self.draft_model(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=None,
                    inputs_embeds=None,
                )

        block_size = self.draft_kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size

        shared_rows: list[int] = []
        shared_input_ids: list[int] = []
        shared_positions: list[int] = []
        shared_seq_lens: list[int] = []
        shared_num_scheduled: list[int] = []
        shared_query_start_loc: list[int] = [0]

        tail_rows: list[int] = []
        tail_input_ids: list[int] = []
        tail_positions: list[int] = []
        tail_seq_lens: list[int] = []
        tail_num_scheduled: list[int] = []
        tail_query_start_loc: list[int] = [0]

        for group in groups:
            particle_rows = self.particle_groups[group.group_id].particle_rows
            L = len(group.prompt_token_ids)
            shared_len = (L // block_size) * block_size
            tail_len = L - shared_len

            if shared_len > 0:
                shared_rows.append(particle_rows[0])
                shared_input_ids.extend(group.prompt_token_ids[:shared_len])
                shared_positions.extend(range(shared_len))
                shared_seq_lens.append(shared_len)
                shared_num_scheduled.append(shared_len)
                shared_query_start_loc.append(
                    shared_query_start_loc[-1] + shared_len
                )

            if tail_len > 0:
                tail_tokens = group.prompt_token_ids[shared_len:]
                tail_pos = list(range(shared_len, L))
                for row in particle_rows:
                    tail_rows.append(row)
                    tail_input_ids.extend(tail_tokens)
                    tail_positions.extend(tail_pos)
                    tail_seq_lens.append(L)
                    tail_num_scheduled.append(tail_len)
                    tail_query_start_loc.append(tail_query_start_loc[-1] + tail_len)

        run_prefill_chunk(
            shared_rows,
            shared_input_ids,
            shared_positions,
            shared_seq_lens,
            shared_num_scheduled,
            shared_query_start_loc,
        )
        run_prefill_chunk(
            tail_rows,
            tail_input_ids,
            tail_positions,
            tail_seq_lens,
            tail_num_scheduled,
            tail_query_start_loc,
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

    def _apply_cow_block_repairs(
        self,
        repaired_block_ids: dict[str, list[tuple[int, list[int]]]],
        block_repairs: list[COWBlockRepair],
    ) -> None:
        """Apply scheduler-planned COW block repairs before SMC writes."""
        if repaired_block_ids:
            for group_id, rows in repaired_block_ids.items():
                group = self.particle_groups.get(group_id)
                if group is None:
                    continue
                for particle_index, block_ids in rows:
                    row = group.particle_rows[particle_index]
                    self.block_tables.append_block_ids(
                        row,
                        tuple(
                            list(block_ids)
                            for _ in range(self.block_tables.num_kv_cache_groups)
                        ),
                        overwrite=True,
                    )
            self.block_tables.apply_staged_writes()

        copy_repairs = [repair for repair in block_repairs if repair.copy_required]
        if not copy_repairs:
            return

        src_blocks = torch.tensor(
            [repair.old_block_id for repair in copy_repairs],
            dtype=torch.long,
            device=self.device,
        )
        dst_blocks = torch.tensor(
            [repair.new_block_id for repair in copy_repairs],
            dtype=torch.long,
            device=self.device,
        )
        for kv_cache in list(self.draft_kv_caches) + list(self.kv_caches):
            snapshot = self._copy_kv_blocks(kv_cache, src_blocks)
            self._write_kv_blocks(kv_cache, dst_blocks, snapshot)

    @torch.inference_mode()
    def _run_batched_target_verify(
        self,
        batches: list[SMCGroupBatch | NewParticleGroupData],
        draft_results: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Run target model on [seed, t1..t_gamma] per particle.

        Fills target KV at the draft positions, samples a bonus token from the
        target logit at position L+gamma, and computes logprob_diff.

        Returns:
            bonus_tokens:  dict group_id -> [A_i]  int32
            logprob_diff:  dict group_id -> [A_i]  float32
                           sum_t( log p_target(t_t) - log q_draft(t_t) ) for t=1..gamma
        """
        gamma = batches[0].gamma

        # Collect active particles across all groups
        group_slices: dict[str, tuple[int, int]] = {}
        all_rows: list[int] = []
        all_start_seq_lens: list[int] = []
        all_draft_ids: list[torch.Tensor] = []
        all_draft_log_probs: list[torch.Tensor] = []
        all_token_counts: list[int] = []
        all_max_tokens: list[int | None] = []
        all_stop_token_ids: list[list[int]] = []
        all_target_temperatures: list[float] = []

        offset = 0
        for batch in batches:
            group = self.particle_groups[batch.group_id]
            finished = getattr(batch, "particle_finished", None) or [False] * len(group.particle_rows)
            active_local = [i for i, f in enumerate(finished) if not f]
            A_i = len(active_local)
            if A_i > 0:
                seq_lens_i = batch.seq_lens[active_local]
                token_counts = getattr(batch, "token_counts", [0] * len(group.particle_rows))
                max_tokens = getattr(batch, "max_tokens", None)
                stop_token_ids = list(getattr(batch, "stop_token_ids", []) or [])
                target_temperature = getattr(batch, "target_temperature", None)
                if target_temperature is None:
                    sp = getattr(batch, "sampling_params", None)
                    target_temperature = (
                        sp.temperature
                        if sp is not None and sp.temperature is not None
                        else 1.0
                    )
                all_rows.extend(group.particle_rows[i] for i in active_local)
                all_start_seq_lens.extend(seq_lens_i.tolist())
                all_token_counts.extend(token_counts[i] for i in active_local)
                all_max_tokens.extend([max_tokens] * A_i)
                all_stop_token_ids.extend([stop_token_ids] * A_i)
                all_target_temperatures.extend([float(target_temperature)] * A_i)
                all_draft_ids.append(draft_results[batch.group_id][0])       # [A_i, gamma+1]
                all_draft_log_probs.append(draft_results[batch.group_id][1]) # [A_i, gamma]
            group_slices[batch.group_id] = (offset, offset + A_i)
            offset += A_i

        A_total = offset
        if A_total == 0:
            return {}, {}

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
        # hidden_states: [A_total * (gamma+1), H] — tokens are laid out sequentially,
        # gamma+1 tokens per particle, so we can safely reshape.

        # Compute logits for all gamma+1 positions in one call.
        all_logits = self.model.compute_logits(hidden_states)           # [A_total*(gamma+1), V]
        all_logits_2d = all_logits.reshape(A_total, gamma + 1, -1)     # [A_total, gamma+1, V]

        # Bonus logits from the last position (L_i+gamma) for sampling.
        bonus_logits = all_logits_2d[:, -1, :]                         # [A_total, V]

        # ---- log-weight increment: keep per-position diffs. ----
        draft_ids = torch.cat(all_draft_ids, dim=0)                     # [A_total, gamma+1]
        draft_log_probs = torch.cat(all_draft_log_probs, dim=0)         # [A_total, gamma] scalars
        if gamma > 0:
            # Target log-probs at positions 0..gamma-1 predict tokens t_1..t_gamma.
            target_temperatures = torch.tensor(
                all_target_temperatures, dtype=all_logits_2d.dtype, device=self.device
            ).clamp_min(1e-5)
            target_log_probs = torch.log_softmax(
                all_logits_2d[:, :gamma, :] / target_temperatures.view(-1, 1, 1),
                dim=-1,
            )                                                           # [A_total, gamma, V]
            draft_token_ids = draft_ids[:, 1:gamma + 1].long()         # [A_total, gamma]
            log_p_target = target_log_probs.gather(
                2, draft_token_ids.unsqueeze(2)
            ).squeeze(2)                                                # [A_total, gamma]
            # draft_log_probs is already [A_total, gamma] scalars gathered at sampling time.
            logprob_diff_per_pos = log_p_target - draft_log_probs       # [A_total, gamma]
        else:
            logprob_diff_per_pos = torch.empty(
                A_total, 0, dtype=torch.float32, device=self.device
            )

        bonus_input_batch = self._build_draft_input_batch(
            particle_rows=rows,
            input_ids=input_ids[last_token_indices],
            positions=positions[last_token_indices],
            seq_lens=seq_lens_full,
            query_start_loc=torch.arange(A_total + 1, dtype=torch.int32, device=self.device),
            query_start_loc_np=np.arange(A_total + 1, dtype=np.int32),
        )
        sampler_output = self.target_sampler(bonus_logits, bonus_input_batch)
        bonus_tokens = sampler_output.sampled_token_ids.squeeze(-1).to(torch.int32)

        # Mask out post-stop / post-length draft weights before resampling.
        if gamma > 0:
            accepted_tokens = torch.cat(
                (draft_ids[:, 1:].to(torch.int32), bonus_tokens.unsqueeze(1)),
                dim=1,
            )  # [A_total, gamma+1]
            weight_lens = [gamma] * A_total
            for i, (count, max_tokens) in enumerate(
                zip(all_token_counts, all_max_tokens)
            ):
                if max_tokens is not None:
                    visible_len = max(0, min(gamma + 1, int(max_tokens) - int(count)))
                    weight_lens[i] = min(weight_lens[i], visible_len, gamma)

            accepted_cpu = accepted_tokens.detach().cpu().tolist()
            for i, stop_ids in enumerate(all_stop_token_ids):
                if not stop_ids:
                    continue
                stop_set = set(stop_ids)
                stop_pos = next(
                    (j for j, token_id in enumerate(accepted_cpu[i]) if token_id in stop_set),
                    -1,
                )
                if stop_pos >= 0:
                    weight_lens[i] = min(weight_lens[i], stop_pos + 1, gamma)

            weight_lens_t = torch.tensor(
                weight_lens, dtype=torch.int64, device=self.device
            )
            keep = (
                torch.arange(gamma, device=self.device).unsqueeze(0)
                < weight_lens_t.unsqueeze(1)
            )
            logprob_diff_flat = (logprob_diff_per_pos * keep).sum(dim=1)
        else:
            logprob_diff_flat = torch.zeros(
                A_total, dtype=torch.float32, device=self.device
            )

        # Commit step gamma with the bonus token
        post_update(
            rows,
            self.req_states.num_computed_tokens.gpu,
            self.req_states.last_sampled_tokens,
            self.target_sampler.penalties_state.output_bin_counts,
            sampler_output.sampled_token_ids,
            sampler_output.num_sampled,
            torch.zeros_like(sampler_output.num_sampled),
            torch.arange(A_total + 1, dtype=torch.int32, device=self.device),
            self.req_states.all_token_ids.gpu,
            self.req_states.total_len.gpu,
        )
        self.req_states.num_computed_tokens_np[rows.cpu().numpy()] += 1

        active = {
            batch.group_id: (s, e)
            for batch in batches
            for s, e in [group_slices[batch.group_id]]
            if e > s
        }
        return (
            {gid: bonus_tokens[s:e]     for gid, (s, e) in active.items()},
            {gid: logprob_diff_flat[s:e] for gid, (s, e) in active.items()},
        )
    
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
        assert self.draft_model_state is not None
        attn_metadata = self.draft_model_state.prepare_attn(
            input_batch,
            CUDAGraphMode.NONE,
            block_tables,
            slot_ids,
            self.draft_attn_groups,
            self.draft_kv_cache_config,
        )

        return attn_metadata

    def _run_batched_draft_decode(
        self,
        batches: list[SMCGroupBatch | NewParticleGroupData],
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Run gamma+1 draft decode steps for all groups in one batched forward pass per step.

        Returns:
            dict group_id -> (draft_token_ids [A_i, gamma+1],
                              draft_log_probs  [A_i, gamma, V],
                              next_seed_ids    [A_i])
            draft_log_probs covers steps 0..gamma-1 (the gamma committed tokens).
            Step gamma's log_probs are not computed (that token is discarded).
        """
        assert self.draft_kv_cache_config is not None
        gamma = batches[0].gamma

        # Collect active particles across all groups
        group_slices: dict[str, tuple[int, int]] = {}
        all_rows: list[int] = []
        all_seeds: list[int] = []
        all_seq_lens: list[int] = []
        all_temperatures: list[float] = []

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
                all_temperatures.extend([batch.temperature] * A_i)
            group_slices[batch.group_id] = (offset, offset + A_i)
            offset += A_i

        A_total = offset

        # Run gamma+1 decode steps over all A_total particles
        rows = torch.tensor(all_rows, dtype=torch.int32, device=self.device)
        seq_lens_cur = torch.tensor(all_seq_lens, dtype=torch.int32, device=self.device)
        draft_ids = torch.zeros(A_total, gamma + 1, dtype=torch.int32, device=self.device)
        draft_log_probs = torch.zeros(A_total, gamma, dtype=torch.float32, device=self.device)
        draft_ids[:, 0] = torch.tensor(all_seeds, dtype=torch.int32, device=self.device)
        next_seeds = torch.zeros(A_total, dtype=torch.int32, device=self.device)
        temps = torch.tensor(
            all_temperatures, dtype=torch.float32, device=self.device,
        ).unsqueeze(1)  # [A_total, 1]

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

            assert self.sampler is not None
            sampler_output = self.sampler(logits, input_batch)
            next_tokens = sampler_output.sampled_token_ids.squeeze(-1).to(torch.int32)

            if step < gamma:
                # Gather scalar log q_draft(t_{step+1}) at the sampled token.
                draft_log_probs[:, step] = torch.log_softmax(logits / temps, dim=-1).gather(
                    1, next_tokens.unsqueeze(1).long()
                ).squeeze(1)
                # Commit draft token t_{step+1}
                post_update(
                    rows,
                    self.req_states.num_computed_tokens.gpu,
                    self.req_states.last_sampled_tokens,
                    self.sampler.penalties_state.output_bin_counts,
                    sampler_output.sampled_token_ids,
                    sampler_output.num_sampled,
                    torch.zeros_like(sampler_output.num_sampled),
                    query_start_loc,
                    self.req_states.all_token_ids.gpu,
                    self.req_states.total_len.gpu,
                )
                self.req_states.num_computed_tokens_np[rows.cpu().numpy()] += 1
                draft_ids[:, step + 1] = next_tokens
            else:
                # Step gamma fills draft KV at L+gamma but its token is discarded
                next_seeds = next_tokens

            seq_lens_cur = seq_lens_cur + 1

        results: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for batch in batches:
            s, e = group_slices[batch.group_id]
            results[batch.group_id] = (
                draft_ids[s:e],
                draft_log_probs[s:e],
                next_seeds[s:e],
            )
        return results

    @torch.inference_mode()
    def _run_batched_resample(
        self,
        batches: list[SMCGroupBatch | NewParticleGroupData],
        logprob_diff_per_group: dict[str, torch.Tensor],
        resample_threshold: float,
    ) -> tuple[dict[str, torch.Tensor], dict[str, list[list[int]]]]:
        """Accumulate importance weights; resample groups whose ESS < threshold * N.

        For resampled groups:
          - Runs systematic resampling on GPU → ancestor_indices [N]
          - Copies KV data (draft + target) from src decode blocks to dst decode blocks
          - Copies req_states fields (token history, sampler penalty state)
          - Resets log_weights to 0

        Returns:
            dict group_id -> ancestor_indices [N_i] (CPU int64, resampled groups only)
        """
        block_size = self.kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size
        resampled: dict[str, torch.Tensor] = {}
        resampled_block_ids: dict[str, list[list[int]]] = {}

        for batch in batches:
            group_id = batch.group_id
            group = self.particle_groups[group_id]
            N = len(group.particle_rows)  # total particles including finished

            # All-particle rows for ESS, resampling, KV copy, and weight reset.
            all_rows = torch.tensor(group.particle_rows, dtype=torch.int32, device=self.device)

            # Scatter logprob_diff only into rows that were forwarded this cycle.
            finished = getattr(batch, "particle_finished", None) or [False] * N
            active_local = [i for i, f in enumerate(finished) if not f]
            if active_local and group_id in logprob_diff_per_group:
                active_rows = torch.tensor(
                    [group.particle_rows[i] for i in active_local],
                    dtype=torch.int32, device=self.device,
                )
                self.log_weights[active_rows] += logprob_diff_per_group[group_id]

            # ESS over all N particles.
            lw = self.log_weights[all_rows]                             # [N]
            log_Z = torch.logsumexp(lw, dim=0)
            log_w_norm = lw - log_Z
            ess = (-torch.logsumexp(2.0 * log_w_norm, dim=0)).exp()

            if ess.item() >= resample_threshold * N:
                continue

            # Systematic resampling over all N particles.
            w = log_w_norm.exp()                                        # [N] normalised
            cumw = torch.cumsum(w, dim=0)                               # [N] CDF
            u = torch.rand(1, device=self.device, dtype=torch.float32) / N
            points = u + torch.arange(N, device=self.device, dtype=torch.float32) / N
            ancestor_indices = torch.searchsorted(
                cumw.contiguous(), points.contiguous()
            ).clamp(0, N - 1)                                           # [N] in 0..N-1

            src_rows = all_rows[ancestor_indices]                       # [N]

            # Used ancestor blocks, including a partial tail block, are
            # remapped read-only. The next scheduler step repairs writable
            # shared blocks with lazy COW before any particle writes again.
            all_rows_np = all_rows.cpu().numpy()
            all_seq_lens_np = self.req_states.num_computed_tokens_np[all_rows_np]
            bt_tuple = self._gather_block_tables(all_rows)
            bt = bt_tuple[0]

            new_block_rows: list[list[int]] = []
            ancestor_indices_cpu = ancestor_indices.cpu().tolist()
            for di, si in enumerate(ancestor_indices_cpu):
                dst_row = int(all_rows_np[di])
                src_seq_len = int(all_seq_lens_np[si])
                num_used_blocks = (src_seq_len + block_size - 1) // block_size
                dst_num_blocks = int(self.block_tables.num_blocks.np[0, dst_row])
                src_row = int(all_rows_np[si])
                src_num_blocks = int(self.block_tables.num_blocks.np[0, src_row])

                old_dst_ids = [int(x) for x in bt[di, :dst_num_blocks].cpu().tolist()]
                src_ids = [int(x) for x in bt[si, :src_num_blocks].cpu().tolist()]
                if num_used_blocks > src_num_blocks:
                    raise RuntimeError(
                        f"SMC resample needs {num_used_blocks} used blocks "
                        f"from ancestor {si}, but only {src_num_blocks} are allocated."
                    )
                if num_used_blocks > dst_num_blocks:
                    raise RuntimeError(
                        f"SMC resample needs {num_used_blocks} destination block "
                        f"slots for particle {di}, but only {dst_num_blocks} are allocated."
                    )

                new_ids = src_ids[:num_used_blocks] + old_dst_ids[num_used_blocks:]
                new_block_rows.append(new_ids)

            for row, block_ids in zip(group.particle_rows, new_block_rows):
                self.block_tables.append_block_ids(
                    row,
                    tuple(
                        list(block_ids)
                        for _ in range(self.block_tables.num_kv_cache_groups)
                    ),
                    overwrite=True,
                )
            self.block_tables.apply_staged_writes()
            resampled_block_ids[group_id] = new_block_rows

            # Copy GPU runner bookkeeping for all N particles.
            rows_l = all_rows.long()
            src_rows_l = src_rows.long()
            self.req_states.last_sampled_tokens[rows_l] = (
                self.req_states.last_sampled_tokens[src_rows_l].clone()
            )
            self.req_states.all_token_ids.gpu[rows_l] = (
                self.req_states.all_token_ids.gpu[src_rows_l].clone()
            )
            self.req_states.num_computed_tokens.gpu[rows_l] = (
                self.req_states.num_computed_tokens.gpu[src_rows_l].clone()
            )
            self.req_states.total_len.gpu[rows_l] = (
                self.req_states.total_len.gpu[src_rows_l].clone()
            )
            dst_np = all_rows.cpu().numpy()
            src_np = src_rows.cpu().numpy()
            self.req_states.num_computed_tokens_np[dst_np] = (
                self.req_states.num_computed_tokens_np[src_np].copy()
            )
            # Copy sampler penalty counters for draft and target samplers so
            # repetition/frequency penalties follow the ancestor trajectory.
            if self.sampler is not None:
                bc = self.sampler.penalties_state.output_bin_counts
                if bc is not None:
                    bc[rows_l] = bc[src_rows_l].clone()
            if self.target_sampler is not None:
                bc = self.target_sampler.penalties_state.output_bin_counts
                if bc is not None:
                    bc[rows_l] = bc[src_rows_l].clone()

            # Reset runner-side interval weights for all N particles.
            self.log_weights[all_rows] = 0.0

            resampled[group_id] = ancestor_indices.cpu()               # [N] in 0..N-1

        return resampled, resampled_block_ids

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

        active_ids = (
            {g.group_id for g in scheduler_output.new_particle_groups}
            | {b.group_id for b in scheduler_output.ongoing_smc_groups}
        )
        for group_id in list(self.particle_groups):
            if group_id not in active_ids:
                self.remove_particle_group(group_id)

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

        self._apply_cow_block_repairs(
            scheduler_output.cow_repaired_block_ids,
            scheduler_output.cow_block_repairs,
        )

        if scheduler_output.new_particle_groups:
            self._run_batched_draft_prefill(scheduler_output.new_particle_groups)
            self._run_batched_target_prefill_tail(scheduler_output.new_particle_groups)

        all_batches: list[SMCGroupBatch | NewParticleGroupData] = (
            list(scheduler_output.new_particle_groups)
            + list(scheduler_output.ongoing_smc_groups)
        )
        resampled_groups: dict[str, torch.Tensor] = {}
        resampled_block_ids: dict[str, list[list[int]]] = {}
        smc_logprob_diffs: dict[str, torch.Tensor] = {}
        if all_batches:
            resample_threshold = self.vllm_config.smc_config.resample_threshold
            if os.environ.get("SMC_VLLM_UNBATCH_DRAFT") == "1":
                for batch in all_batches:
                    smc_draft_results.update(
                        self._run_batched_draft_decode([batch])
                    )
            else:
                smc_draft_results = self._run_batched_draft_decode(all_batches)

            if os.environ.get("SMC_VLLM_UNBATCH_TARGET") == "1":
                bonus_tokens_per_group = {}
                logprob_diff_per_group = {}
                for batch in all_batches:
                    bonus, logprob_diff = self._run_batched_target_verify(
                        [batch], smc_draft_results
                    )
                    bonus_tokens_per_group.update(bonus)
                    logprob_diff_per_group.update(logprob_diff)
            else:
                bonus_tokens_per_group, logprob_diff_per_group = self._run_batched_target_verify(
                    all_batches, smc_draft_results
                )
            for group_id, bonus in bonus_tokens_per_group.items():
                draft_ids, log_probs, _ = smc_draft_results[group_id]
                smc_draft_results[group_id] = (draft_ids, log_probs, bonus)
            smc_logprob_diffs = logprob_diff_per_group
            resampled_groups, resampled_block_ids = self._run_batched_resample(
                all_batches, logprob_diff_per_group, resample_threshold
            )

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
            resampled_groups=resampled_groups,
            resampled_block_ids=resampled_block_ids,
            smc_logprob_diffs=smc_logprob_diffs,
        )

    def shutdown(self) -> None:
        draft_ctx = self.draft_vllm_config.compilation_config.static_forward_context
        for draft_layer in draft_ctx.values():
            draft_layer.kv_cache = None
        draft_ctx.clear()
        self.draft_model = None
