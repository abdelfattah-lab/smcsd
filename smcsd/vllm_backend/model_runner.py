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

    def add_particle_group(
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
            if self.is_last_pp_rank and self.sampler is not None:
                draft_sampling_params = copy.copy(sampling_params)
                draft_sampling_params.temperature = temperature
                self.sampler.add_request(
                    req_index, prompt_len, draft_sampling_params
                )

        self.req_states.apply_staged_writes()
        self.block_tables.apply_staged_writes()
        if self.sampler is not None:
            self.sampler.apply_staged_writes()

        if particle_req_ids:
            debug_particles = min(2, len(particle_req_ids))
            print(f"[smc-debug] add_particle_group group_id={group_id} prompt_len={prompt_len} num_computed_tokens={num_computed_tokens}")
            print(f"[smc-debug] shared_prefix_block_ids={tuple(list(x) for x in prefix_block_ids)}")
            for i in range(debug_particles):
                combined_block_ids = tuple(
                    list(prefix_block_ids[g]) + list(decode_block_ids[i][g])
                    for g in range(len(prefix_block_ids))
                )
                print(
                    f"[smc-debug] particle[{i}] req_id={particle_req_ids[i]} "
                    f"row={particle_rows[i]} combined_block_ids={combined_block_ids}"
                )

        self._run_draft_prefill(prompt_token_ids, particle_rows)

        self.particle_groups[group_id] = ParticleGroup(
            group_id=group_id,
            particle_rows=particle_rows,
            particle_req_ids=list(particle_req_ids),
            n_particles=len(particle_req_ids),
        )

    def remove_particle_group(self, group_id: str) -> None:
        """Return all particle rows to req_states.free_indices."""
        group = self.particle_groups.pop(group_id)
        for p_id in group.particle_req_ids:
            self.req_states.remove_request(p_id)

    def smc_draft_cycle(
        self,
        particle_rows: torch.Tensor,   # [N] — row indices
        seed_token_ids: torch.Tensor,  # [N] — last accepted token per particle
        seq_lens: torch.Tensor,        # [N] — num_computed_tokens at cycle start
        gamma: int,
        temperature: float,
        particle_finished: list[bool] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run gamma+1 autoregressive draft steps for active (unfinished) particles.

        Returns:
            draft_token_ids   [N, gamma+1]
            draft_log_probs   [N, gamma+1, V]
            next_seed_ids     [N]
        """
        P = particle_rows.shape[0]
        seed_token_ids = seed_token_ids.to(self.device)
        seq_lens = seq_lens.to(self.device)

        # Build index of active particles
        if particle_finished and any(particle_finished):
            active_idx = torch.tensor(
                [i for i, f in enumerate(particle_finished) if not f],
                dtype=torch.long, device=self.device,
            )
        else:
            active_idx = torch.arange(P, dtype=torch.long, device=self.device)

        A = active_idx.shape[0]
        rows = particle_rows[active_idx]
        seq_lens_cur = seq_lens[active_idx].clone()

        draft_ids = torch.zeros(A, gamma + 1, dtype=torch.int32, device=self.device)
        log_probs = torch.zeros(A, gamma + 1, self.vocab_size, dtype=torch.float32, device=self.device)
        draft_ids[:, 0] = seed_token_ids[active_idx].to(torch.int32)
        next_seeds = torch.zeros(A, dtype=torch.int32, device=self.device)

        for step in range(gamma + 1):
            input_ids = draft_ids[:, step]
            positions = seq_lens_cur.long()
            query_start_loc = torch.arange(A + 1, dtype=torch.int32, device=self.device)
            query_start_loc_np = np.arange(A + 1, dtype=np.int32)
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
                num_tokens_padded=A,
            )
            attn_metadata = self._build_draft_attn_metadata(input_batch, slot_ids)
            assert self.draft_kv_cache_config is not None
            slot_mappings_by_layer = build_slot_mappings_by_layer(
                slot_ids, self.draft_kv_cache_config
            )

            with set_forward_context(
                attn_metadata,
                self.draft_vllm_config,
                num_tokens=A,
                slot_mapping=slot_mappings_by_layer,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                batch_descriptor=BatchDescriptor(num_tokens=A, has_lora=False),
            ):
                hidden_states = self.draft_model(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=None,
                    inputs_embeds=None,
                )  # [A, hidden_dim]

            logits = self.draft_model.compute_logits(hidden_states)   # [A, vocab_size]
            if step == 0 and A >= 2:
                topk = 5
                p0_top_vals, p0_top_idx = torch.topk(logits[0], k=topk)
                p1_top_vals, p1_top_idx = torch.topk(logits[1], k=topk)
                print(
                    "[smc-debug] step0 logits "
                    f"argmax0={int(logits[0].argmax().item())} "
                    f"argmax1={int(logits[1].argmax().item())} "
                    f"equal={bool(torch.equal(logits[0], logits[1]))}"
                )
                print(
                    "[smc-debug] step0 topk particle[0] "
                    f"indices={p0_top_idx.tolist()} values={p0_top_vals.tolist()}"
                )
                print(
                    "[smc-debug] step0 topk particle[1] "
                    f"indices={p1_top_idx.tolist()} values={p1_top_vals.tolist()}"
                )
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

        return draft_ids, log_probs, next_seeds

    def smc_target_score(
        self,
        particle_rows: torch.Tensor,    # [N]
        draft_token_ids: torch.Tensor,  # [N, gamma+1]
        seq_lens: torch.Tensor,         # [N] — num_computed_tokens at draft start
    ) -> torch.Tensor:
        """Score all draft tokens with the target model in one batched pass.

        Returns:
            target_log_probs  [P, gamma+1, V]
        """
        raise NotImplementedError
    
    def _run_draft_prefill(
        self,
        prompt_token_ids: list[int],
        particle_rows: list[int],
    ) -> None:
        """Prefill draft KV for all particles."""
        assert self.draft_kv_cache_config is not None
        N = len(particle_rows)
        prompt_len = len(prompt_token_ids)
        total_tokens = N * prompt_len
        particle_rows_t = torch.tensor(
            particle_rows, dtype=torch.int32, device=self.device
        )
        block_tables = self._gather_block_tables(particle_rows_t)
        input_ids = torch.tensor(
            prompt_token_ids * N, dtype=torch.int32, device=self.device
        )
        positions = torch.arange(prompt_len, dtype=torch.long, device=self.device).repeat(N)
        query_start_loc = torch.arange(
            N + 1, dtype=torch.int32, device=self.device
        ) * prompt_len
        query_start_loc_np = (
            np.arange(N + 1, dtype=np.int32) * prompt_len
        )
        seq_lens = torch.full((N,), prompt_len, dtype=torch.int32, device=self.device)
        slot_ids = self.block_tables.compute_slot_mappings(
            particle_rows_t,
            query_start_loc,
            positions,
            num_tokens_padded=total_tokens,
        )
        seq_lens_cpu_upper_bound = torch.full(
            (N,), prompt_len, dtype=torch.int32
        )
        input_batch = SimpleNamespace(
            num_reqs=N,
            num_tokens=total_tokens,
            num_reqs_after_padding=N,
            num_tokens_after_padding=total_tokens,
            query_start_loc=query_start_loc,
            query_start_loc_np=query_start_loc_np,
            num_scheduled_tokens=np.full(N, prompt_len, dtype=np.int32),
            seq_lens=seq_lens,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
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
    
        if not self._draft_attn_metadata_debugged and N >= 2:
            print("[smc-debug] attn_metadata input")
            print(f"[smc-debug] num_reqs={N} num_tokens={N} max_query_len=1")
            print(f"[smc-debug] query_start_loc_gpu={input_batch.query_start_loc.tolist()}")
            print(f"[smc-debug] query_start_loc_cpu={input_batch.query_start_loc_np.tolist()}")
            print(f"[smc-debug] seq_lens={input_batch.seq_lens.tolist()}")
            print(f"[smc-debug] positions={input_batch.positions.tolist()}")
            print(f"[smc-debug] slot_ids={slot_ids.cpu().tolist()}")
            for g, bt in enumerate(block_tables):
                rows_to_show = min(2, bt.shape[0])
                print(
                    f"[smc-debug] block_table[{g}] shape={tuple(bt.shape)} "
                    f"rows={bt[:rows_to_show].cpu().tolist()}"
                )
            print(f"[smc-debug] attn_metadata_keys={list(attn_metadata.keys())}")
            print("[smc-debug] attn_metadata effective")
            for layer_name, layer_metadata in attn_metadata.items():
                print(f"[smc-debug] layer={layer_name}")
                print(pformat(layer_metadata))
            self._draft_attn_metadata_debugged = True

        return attn_metadata

    def _run_draft_and_verify(
        self,
        batch: SMCGroupBatch | NewParticleGroupData,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one draft cycle for a group; return (draft_token_ids, draft_log_probs)."""
        particle_group = self.particle_groups[batch.group_id]
        particle_rows = torch.tensor(
            particle_group.particle_rows,
            dtype=torch.int32,
            device=self.device,
        )
        if not particle_group.debug_dumped:
            debug_particles = min(2, particle_rows.shape[0])
            print(f"[smc-debug] draft_cycle group_id={batch.group_id}")
            print(f"[smc-debug] seed_token_ids={batch.seed_token_ids[:debug_particles].tolist()}")
            print(f"[smc-debug] seq_lens={batch.seq_lens[:debug_particles].tolist()}")
            print(f"[smc-debug] particle_rows={particle_rows[:debug_particles].tolist()}")
            particle_group.debug_dumped = True
        draft_token_ids, draft_log_probs, next_seed_ids = self.smc_draft_cycle(
            particle_rows=particle_rows,
            seed_token_ids=batch.seed_token_ids,
            seq_lens=batch.seq_lens,
            gamma=batch.gamma,
            temperature=batch.temperature,
            particle_finished=getattr(batch, "particle_finished", None),
        )
        # TODO: target verify
        return draft_token_ids, draft_log_probs, next_seed_ids

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

        # Newly forked groups: register particle rows then run first draft cycle.
        for new_group in scheduler_output.new_particle_groups:
            self.add_particle_group(
                group_id=new_group.group_id,
                particle_req_ids=new_group.particle_req_ids,
                prompt_token_ids=new_group.prompt_token_ids,
                num_computed_tokens=new_group.num_computed_tokens,
                sampling_params=new_group.sampling_params,
                prefix_block_ids=new_group.prefix_block_ids,
                decode_block_ids=new_group.decode_block_ids,
                temperature=new_group.temperature,
            )
            smc_draft_results[new_group.group_id] = self._run_draft_and_verify(new_group)

        # Ongoing groups: run draft + verify.
        for batch in scheduler_output.ongoing_smc_groups:
            smc_draft_results[batch.group_id] = self._run_draft_and_verify(batch)

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
