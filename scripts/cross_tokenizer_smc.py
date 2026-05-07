"""Cross-tokenizer SMC speculative decoding (v4 — persistent draft KV).

Target and draft use *different* tokenizers. The simplest design that
avoids any cross-tokenizer round-trip into the draft model:

  * Each particle keeps its own persistent draft KV across cycles —
    the draft never has any target token round-tripped into its input.
  * Each particle keeps its own persistent target KV across cycles —
    the target only ever sees the *retokenization* of draft-emitted
    text, treated as a normal extend forward.
  * No bonus token. The target never samples or generates new tokens;
    its sole job is to score the draft's proposed blocks at the
    string level.

Per outer SMC cycle, for each live particle:

  1. Draft AR γ+1 steps from its persistent KV → block of γ+1 draft
     tokens (and ``q_draft(block) = sum log p_draft_i`` accumulated
     across the AR loop).
  2. Detokenize the block under the *draft* tokenizer
     (``skip_special_tokens=True``).
  3. Re-encode that string under the *target* tokenizer.
  4. Target forward (extend) over the re-encoded block IDs, advancing
     the target's KV. Sum ``log p_target(token_i | prefix)`` over those
     IDs to get ``p_target(block)``.
  5. Block-level importance weight: ``log p_target − log q_draft``.

ESS check + systematic resample at the end of each cycle. On resample,
deepcopy both the target and draft KV from src particle to dst.

Output is taken from the highest-cumulative-weight particle's *target*
token sequence (these are the canonical retokenizations the target
itself produced KV for).

Usage (Modal H200):
    modal run modal_apps/run_cross_tokenizer_smc.py::run \
        --num-questions 5 --particles 4 --gamma 4
"""

from __future__ import annotations

import argparse
import copy
import re
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_TARGET = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_DRAFT = "Qwen/Qwen3-0.6B"


# ---------------------------------------------------------------------------
# GSM8K helpers (mirror scripts/accuracy_test_gsm8k.py)
# ---------------------------------------------------------------------------


def extract_answer(text: str) -> Optional[str]:
    match = re.search(r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    last_line = lines[-1] if lines else text.strip()
    numbers = re.findall(r"-?\d+(?:,\d+)*(?:\.\d+)?", last_line)
    return numbers[-1].replace(",", "") if numbers else None


def format_instruction(question: str) -> str:
    return (
        "Solve this math problem step by step.\n"
        "At the very end, output ONLY the final numeric answer "
        "on a new line in the exact format:\n"
        "#### <number>\n\n"
        f"Problem:\n{question}\n"
    )


# ---------------------------------------------------------------------------
# Particle state — persistent target *and* draft KV
# ---------------------------------------------------------------------------


@dataclass
class Particle:
    target_ids: torch.Tensor          # (1, T) — target's view of the sequence
    target_past: Any                  # DynamicCache — advances each cycle
    draft_ids: torch.Tensor           # (1, T) — draft's view (different tokenization)
    draft_past: Any                   # DynamicCache — persistent across cycles
    log_weight: float = 0.0
    finished: bool = False


def copy_particle(src: Particle, dst: Particle) -> None:
    dst.target_ids = src.target_ids.clone()
    dst.target_past = copy.deepcopy(src.target_past)
    dst.draft_ids = src.draft_ids.clone()
    dst.draft_past = copy.deepcopy(src.draft_past)
    dst.log_weight = 0.0
    dst.finished = src.finished


# ---------------------------------------------------------------------------
# Forward / sample helpers
# ---------------------------------------------------------------------------


def model_forward(model, input_ids: torch.Tensor, past=None):
    out = model(input_ids=input_ids, past_key_values=past, use_cache=True)
    return out.logits, out.past_key_values


def sample_from_logits(
    logits: torch.Tensor, temperature: float, generator: torch.Generator
) -> Tuple[torch.Tensor, torch.Tensor]:
    if temperature <= 0:
        token = torch.argmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        lp = log_probs.gather(-1, token.unsqueeze(-1)).squeeze(-1)
        return token, lp
    scaled = logits / max(temperature, 1e-5)
    log_probs = F.log_softmax(scaled, dim=-1)
    probs = log_probs.exp()
    token = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
    lp = log_probs.gather(-1, token.unsqueeze(-1)).squeeze(-1)
    return token, lp


# ---------------------------------------------------------------------------
# Cross-tokenizer SMC engine (v4 — persistent draft KV, no bonus)
# ---------------------------------------------------------------------------


class CrossTokenizerSMC:
    def __init__(
        self,
        target_model_id: str,
        draft_model_id: str,
        n_particles: int = 4,
        gamma: int = 4,
        temperature: float = 0.7,
        resample_threshold: float = 0.5,
        max_new_target_tokens: int = 512,
        device: str = "cuda",
        dtype=torch.bfloat16,
        seed: int = 0,
    ):
        self.n = n_particles
        self.gamma = gamma
        self.temperature = temperature
        self.resample_threshold = resample_threshold
        self.max_new_target_tokens = max_new_target_tokens
        self.device = device

        print(f"Loading target {target_model_id} ...", flush=True)
        self.target_tok = AutoTokenizer.from_pretrained(
            target_model_id, trust_remote_code=True
        )
        self.target = AutoModelForCausalLM.from_pretrained(
            target_model_id, torch_dtype=dtype, device_map=device, trust_remote_code=True,
        )
        self.target.eval()

        print(f"Loading draft {draft_model_id} ...", flush=True)
        self.draft_tok = AutoTokenizer.from_pretrained(
            draft_model_id, trust_remote_code=True
        )
        self.draft = AutoModelForCausalLM.from_pretrained(
            draft_model_id, torch_dtype=dtype, device_map=device, trust_remote_code=True,
        )
        self.draft.eval()

        if self.target_tok.pad_token_id is None:
            self.target_tok.pad_token_id = self.target_tok.eos_token_id
        if self.draft_tok.pad_token_id is None:
            self.draft_tok.pad_token_id = self.draft_tok.eos_token_id

        self.target_eos = self.target_tok.eos_token_id
        self.draft_eos = self.draft_tok.eos_token_id
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)

    # ── Prefill — target on prompt; draft also on prompt; both KV captured ──

    @torch.no_grad()
    def _prefill(self, prompt: str) -> List[Particle]:
        # Target prefill — exposes target_logits[-1] but we don't use it
        # (no bonus token; the target only ever scores).
        target_ids = self.target_tok(prompt, return_tensors="pt").input_ids.to(
            self.device
        )
        _, target_past = model_forward(self.target, target_ids, past=None)

        # Draft prefill on the same string under draft's tokenizer.
        draft_ids = self.draft_tok(prompt, return_tensors="pt").input_ids.to(
            self.device
        )
        draft_logits, draft_past = model_forward(self.draft, draft_ids, past=None)

        # Each particle starts from the same prefilled state but with its own
        # *first sampled token* drawn from the draft's prefill last-position
        # distribution — that's where stochastic divergence between particles
        # begins.
        last_draft_logits = draft_logits[0, -1]
        particles: List[Particle] = []
        for _ in range(self.n):
            tok, _ = sample_from_logits(last_draft_logits, self.temperature, self.gen)
            tok_t = tok.view(1, 1)
            # Advance draft KV one step on this sampled token.
            _, p_draft_past = model_forward(
                self.draft, tok_t, past=copy.deepcopy(draft_past)
            )
            p_draft_ids = torch.cat([draft_ids, tok_t], dim=1)

            particles.append(
                Particle(
                    target_ids=target_ids.clone(),
                    target_past=copy.deepcopy(target_past),
                    draft_ids=p_draft_ids,
                    draft_past=p_draft_past,
                    log_weight=0.0,
                )
            )
        return particles

    # ── One outer-cycle: draft AR + target verify (no bonus) ──

    @torch.no_grad()
    def _draft_ar_block(
        self, particle: Particle
    ) -> Tuple[List[int], float, bool]:
        """Sample γ+1 more draft tokens from particle's persistent KV.

        Returns (block_token_ids, sum_log_q_draft, hit_eos).
        """
        sampled: List[int] = []
        sum_lp = 0.0
        hit_eos = False
        # First step: feed the previously-sampled last draft token (already
        # in particle.draft_ids[:, -1] and consumed into draft_past during
        # the prior cycle — except at the very first cycle where prefill +
        # initial sample already appended it). To advance, we sample from
        # the past's last position by feeding a new input — but we don't
        # have the logits cached. So we re-feed the last draft token to
        # get fresh logits, then sample.
        cur = particle.draft_ids[:, -1:]
        past = particle.draft_past
        for step in range(self.gamma + 1):
            logits, past = model_forward(self.draft, cur, past=past)
            tok, lp = sample_from_logits(
                logits[0, -1], self.temperature, self.gen
            )
            sampled.append(int(tok.item()))
            sum_lp += float(lp.item())
            if int(tok.item()) == self.draft_eos:
                hit_eos = True
                # Stop sampling further; the particle's draft sequence
                # ends here. Subsequent steps would just emit padding.
                break
            cur = tok.view(1, 1)
        particle.draft_past = past
        particle.draft_ids = torch.cat(
            [particle.draft_ids, torch.tensor([sampled], device=self.device)],
            dim=1,
        )
        return sampled, sum_lp, hit_eos

    @torch.no_grad()
    def _target_score_block(
        self, particle: Particle, block_text: str
    ) -> Tuple[List[int], float]:
        """Re-encode block_text under target tok; target forward; sum log-probs.

        Advances particle.target_past. Returns (target_token_ids, sum_log_p_target).
        """
        block_ids = self.target_tok(
            block_text, return_tensors="pt", add_special_tokens=False,
        ).input_ids.to(self.device)
        if block_ids.shape[1] == 0:
            # Degenerate (empty retokenization): treat as zero-mass. Don't
            # advance target_past or target_ids.
            return [], 0.0

        logits, past = model_forward(self.target, block_ids, past=particle.target_past)
        log_probs = F.log_softmax(logits[0], dim=-1)  # (m, V)
        token_lp = log_probs.gather(-1, block_ids[0].unsqueeze(-1)).squeeze(-1)
        sum_lp = float(token_lp.sum().item())
        particle.target_past = past
        particle.target_ids = torch.cat([particle.target_ids, block_ids], dim=1)
        return block_ids[0].tolist(), sum_lp

    # ── Resampling ──

    def _maybe_resample(self, particles: List[Particle]) -> bool:
        log_w = np.array([p.log_weight for p in particles], dtype=np.float64)
        max_lw = log_w.max()
        if not np.isfinite(max_lw):
            return False
        w = np.exp(log_w - max_lw)
        w_sum = w.sum()
        if w_sum <= 0:
            return False
        w_norm = w / w_sum
        ess = 1.0 / np.sum(w_norm ** 2)
        if ess >= self.resample_threshold * self.n:
            return False
        u0 = np.random.uniform(0, 1.0 / self.n)
        positions = u0 + np.arange(self.n) / self.n
        cumsum = np.cumsum(w_norm)
        ancestors = (
            np.searchsorted(cumsum, positions).clip(max=self.n - 1).tolist()
        )
        new_states = [particles[a] for a in ancestors]
        for j, src in enumerate(new_states):
            if particles[j] is src and ancestors[j] == j:
                particles[j].log_weight = 0.0
                continue
            copy_particle(src, particles[j])
        return True

    # ── Generate ──

    @torch.no_grad()
    def generate(self, prompt: str) -> Tuple[str, dict]:
        particles = self._prefill(prompt)
        target_prompt_len = particles[0].target_ids.shape[1]
        n_outer_steps = 0

        while True:
            n_outer_steps += 1
            for p in particles:
                if p.finished:
                    continue
                # 1. Draft AR γ+1 steps (persistent KV).
                block_ids, draft_lp, hit_eos = self._draft_ar_block(p)
                if not block_ids:
                    p.finished = True
                    continue

                # 2+3. Detok under draft tok, retok under target tok.
                block_text = self.draft_tok.decode(
                    block_ids, skip_special_tokens=True,
                )
                # 4. Target verify; sum log-probs.
                _, target_lp = self._target_score_block(p, block_text)

                # 5. Block-level importance weight.
                p.log_weight += (target_lp - draft_lp)

                if hit_eos:
                    p.finished = True

            # Resample if ESS low.
            self._maybe_resample(particles)

            # Termination check.
            new_target_lens = [
                p.target_ids.shape[1] - target_prompt_len for p in particles
            ]
            if all(p.finished for p in particles):
                break
            if max(new_target_lens) >= self.max_new_target_tokens:
                break

        # Pick highest cumulative log-weight particle.
        best_j = 0
        best_lw = particles[0].log_weight
        for j, p in enumerate(particles):
            if p.log_weight > best_lw:
                best_lw = p.log_weight
                best_j = j
        best = particles[best_j]
        completion_ids = best.target_ids[0, target_prompt_len:].tolist()
        text = self.target_tok.decode(completion_ids, skip_special_tokens=True)
        info = {
            "outer_steps": n_outer_steps,
            "completion_tokens": len(completion_ids),
            "best_particle": best_j,
        }
        return text, info


# ---------------------------------------------------------------------------
# CLI / driver
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--draft", default=DEFAULT_DRAFT)
    parser.add_argument("--num-questions", type=int, default=10)
    parser.add_argument("--particles", type=int, default=4)
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--resample-threshold", type=float, default=0.5)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--use-chat-template",
        action="store_true",
        help="Apply target's chat template to the prompt (for instruct models). "
             "When off, the prompt is fed as raw text — recommended for base "
             "models or when chat-template special tokens disrupt the cross-"
             "tokenizer round-trip.",
    )
    args = parser.parse_args()

    print(
        f"Cross-tokenizer SMC v4 (persistent draft KV, no bonus): "
        f"target={args.target} draft={args.draft} "
        f"N={args.particles} gamma={args.gamma} T={args.temperature} "
        f"seed={args.seed} use_chat_template={args.use_chat_template}",
        flush=True,
    )

    smc = CrossTokenizerSMC(
        target_model_id=args.target,
        draft_model_id=args.draft,
        n_particles=args.particles,
        gamma=args.gamma,
        temperature=args.temperature,
        resample_threshold=args.resample_threshold,
        max_new_target_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    print("Loading GSM8K ...", flush=True)
    dataset = load_dataset("gsm8k", "main", split="test")

    correct = 0
    total_completion_tokens = 0
    tic = time.perf_counter()
    for i, sample in enumerate(dataset.select(range(args.num_questions))):
        instruction = format_instruction(sample["question"])
        if args.use_chat_template:
            try:
                prompt = smc.target_tok.apply_chat_template(
                    [{"role": "user", "content": instruction}],
                    tokenize=False, add_generation_prompt=True,
                )
            except Exception:
                prompt = instruction
        else:
            prompt = instruction

        gold = extract_answer(sample["answer"])
        text, info = smc.generate(prompt)
        pred = extract_answer(text)
        ok = (pred == gold)
        correct += int(ok)
        total_completion_tokens += info["completion_tokens"]

        elapsed = time.perf_counter() - tic
        print(
            f"[{i+1}/{args.num_questions}] {'OK' if ok else 'X '} "
            f"acc={correct}/{i+1} ({correct/(i+1):.1%}) "
            f"steps={info['outer_steps']} ntok={info['completion_tokens']} "
            f"tps={total_completion_tokens/elapsed:.1f} "
            f"elapsed={elapsed:.0f}s",
            flush=True,
        )
        if i < 2:
            print(f"--- Q{i} pred={pred} gold={gold} ---")
            print(text[:400])
            print()

    elapsed = time.perf_counter() - tic
    print(f"\n{'='*55}")
    print(f"  Cross-tokenizer SMC v4")
    print(f"  Accuracy:          {correct}/{args.num_questions} ({correct/args.num_questions:.1%})")
    print(f"  Output throughput: {total_completion_tokens/elapsed:.1f} tok/s")
    print(f"  Total tokens:      {total_completion_tokens}")
    print(f"  Wall time:         {elapsed:.1f}s")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
