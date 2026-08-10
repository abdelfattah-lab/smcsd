"""GSM8K benchmark for SMC speculative decoding.

Two modes with identical preprocessing for fair comparison:
  - smc_engine: SMC via the dedicated SMCEngine (offline, no tokenizer manager)
  - baseline:   vanilla generation (no speculative decoding)

Usage:
  # SMCEngine (dedicated offline engine)
  python scripts/smc/accuracy_test_gsm8k.py --mode smc_engine -N 8 -g 8

  # Baseline (no speculative decoding)
  python scripts/smc/accuracy_test_gsm8k.py --mode baseline

  # Custom models
  python scripts/smc/accuracy_test_gsm8k.py --mode smc_engine \
      --model meta-llama/Llama-3.1-8B-Instruct \
      --draft-model meta-llama/Llama-3.2-1B-Instruct \
      -N 8 -g 8
"""

import argparse
import json
import os
import re
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_DRAFT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"


# ---------------------------------------------------------------------------
# Shared preprocessing (identical across all modes)
# ---------------------------------------------------------------------------

def normalize_numeric_answer(value: str) -> Optional[str]:
    """Normalize equivalent numeric strings, e.g. 75.00 -> 75."""
    if value is None:
        return None
    try:
        dec = Decimal(str(value).replace(",", ""))
    except (InvalidOperation, ValueError, TypeError):
        return None
    try:
        if dec == dec.to_integral_value():
            # Avoid quantize(Decimal(1)) which raises InvalidOperation when the
            # number has more digits than the default decimal precision (28);
            # format the integral value directly instead.
            return format(dec.to_integral_value(), "f")
        return format(dec.normalize(), "f")
    except (InvalidOperation, ValueError):
        return str(value).replace(",", "").strip()

def extract_answer(text: str) -> Optional[str]:
    """Extract numeric answer from model output or gold answer."""
    match = re.search(r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)", text)
    if match:
        return normalize_numeric_answer(match.group(1))
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    last_line = lines[-1] if lines else text.strip()
    numbers = re.findall(r"-?\d+(?:,\d+)*(?:\.\d+)?", last_line)
    return normalize_numeric_answer(numbers[-1]) if numbers else None


def has_canonical_answer(text: str) -> bool:
    """Whether the response contains the requested ``#### <number>`` marker."""

    return bool(re.search(r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)", text))


def smc_particle_metrics(output: dict, gold: str) -> dict | None:
    """Summarize all terminal particles without changing the sampled output."""

    particle_texts = output.get("smc_particle_texts")
    raw_log_weights = output.get("smc_log_w_tilde")
    if not isinstance(particle_texts, list) or raw_log_weights is None:
        return None
    if hasattr(raw_log_weights, "tolist"):
        raw_log_weights = raw_log_weights.tolist()
    if not isinstance(raw_log_weights, list) or len(raw_log_weights) != len(particle_texts):
        return None
    try:
        log_weights = np.asarray(raw_log_weights, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if log_weights.size == 0:
        return None
    finite_mask = np.isfinite(log_weights)
    positive_inf = np.isposinf(log_weights)
    if np.any(positive_inf):
        normalized_weights = positive_inf.astype(np.float64)
        normalized_weights /= normalized_weights.sum()
        normalization_fallback = "equal_positive_infinity"
    elif np.any(finite_mask):
        normalized_weights = np.zeros_like(log_weights, dtype=np.float64)
        finite_weights = log_weights[finite_mask]
        shifted = np.exp(finite_weights - np.max(finite_weights))
        normalized_weights[finite_mask] = shifted / shifted.sum()
        normalization_fallback = None
    else:
        normalized_weights = np.full(
            log_weights.shape, 1.0 / log_weights.size, dtype=np.float64
        )
        normalization_fallback = "uniform_no_finite_weights"
    particle_preds = [extract_answer(text) for text in particle_texts]
    particle_correct = [pred == gold for pred in particle_preds]
    max_index = int(np.argmax(normalized_weights))
    selected_text = output.get("text")
    selected_particle_indices = [
        index
        for index, particle_text in enumerate(particle_texts)
        if particle_text == selected_text
    ]
    positive_weights = normalized_weights[normalized_weights > 0]
    raw_log_z = output.get("smc_log_Z_hat")
    try:
        log_z = float(raw_log_z) if raw_log_z is not None else None
    except (TypeError, ValueError):
        log_z = None
    if log_z is not None and not np.isfinite(log_z):
        log_z = None
    return {
        "particle_preds": particle_preds,
        "particle_correct": particle_correct,
        "particle_has_canonical_answer": [
            has_canonical_answer(text) for text in particle_texts
        ],
        "particle_log_weights": [
            float(weight) if np.isfinite(weight) else None
            for weight in log_weights
        ],
        "particle_normalized_weights": normalized_weights.tolist(),
        "particle_ess": float(1.0 / np.sum(normalized_weights**2)),
        "particle_weight_entropy": float(
            -np.sum(positive_weights * np.log(positive_weights))
        ),
        "nonfinite_weight_count": int((~finite_mask).sum()),
        "weight_normalization_fallback": normalization_fallback,
        "smc_log_Z_hat": log_z,
        "selected_particle_indices": selected_particle_indices,
        "selected_matches_max_weight": max_index in selected_particle_indices,
        "any_particle_correct": any(particle_correct),
        "correct_particle_count": sum(particle_correct),
        "correct_weight_mass": float(
            sum(
                weight
                for weight, is_correct in zip(normalized_weights, particle_correct)
                if is_correct
            )
        ),
        "max_weight_particle_index": max_index,
        "max_weight_pred": particle_preds[max_index],
        "max_weight_correct": particle_correct[max_index],
        "max_normalized_weight": float(normalized_weights[max_index]),
    }


def aggregate_smc_diagnostics(diagnostics: list[dict]) -> dict:
    """Aggregate durable terminal-particle diagnostics across questions."""

    rows = [row for row in diagnostics if isinstance(row, dict)]
    if not rows:
        return {}

    def mean_of(key: str) -> float | None:
        values = [
            float(row[key])
            for row in rows
            if row.get(key) is not None and np.isfinite(float(row[key]))
        ]
        return float(np.mean(values)) if values else None

    return {
        "questions_with_particle_diagnostics": len(rows),
        "mean_terminal_ess": mean_of("particle_ess"),
        "mean_terminal_ess_fraction": (
            float(
                np.mean(
                    [
                        float(row["particle_ess"])
                        / max(len(row.get("particle_preds", [])), 1)
                        for row in rows
                        if row.get("particle_ess") is not None
                    ]
                )
            )
            if any(row.get("particle_ess") is not None for row in rows)
            else None
        ),
        "mean_max_normalized_weight": mean_of("max_normalized_weight"),
        "mean_correct_weight_mass": mean_of("correct_weight_mass"),
        "any_particle_correct_rate": float(
            np.mean([bool(row.get("any_particle_correct")) for row in rows])
        ),
        "max_weight_correct_rate": float(
            np.mean([bool(row.get("max_weight_correct")) for row in rows])
        ),
        "selected_matches_max_weight_rate": float(
            np.mean(
                [bool(row.get("selected_matches_max_weight")) for row in rows]
            )
        ),
        "nonfinite_weight_count": int(
            sum(int(row.get("nonfinite_weight_count", 0)) for row in rows)
        ),
        "weight_normalization_fallback_count": int(
            sum(
                row.get("weight_normalization_fallback") is not None
                for row in rows
            )
        ),
        "mean_log_Z_hat": mean_of("smc_log_Z_hat"),
    }


def format_instruction(question: str) -> str:
    """Build the instruction prompt for a GSM8K question."""
    return (
        "Solve this math problem step by step.\n"
        "At the very end, output ONLY the final numeric answer "
        "on a new line in the exact format:\n"
        "#### <number>\n\n"
        f"Problem:\n{question}\n"
    )


def load_gsm8k(
    tokenizer,
    num_questions: int,
    *,
    start: int = 0,
    disable_thinking: bool = False,
    question_indices: list[int] | None = None,
):
    """Load GSM8K and build chat-template prompts + gold labels."""
    print("Loading GSM8K dataset...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")

    prompts = []
    labels = []
    selected = (
        dataset.select(question_indices)
        if question_indices is not None
        else dataset.select(range(start, start + num_questions))
    )
    for sample in selected:
        instruction = format_instruction(sample["question"])
        chat_template_kwargs = {}
        if disable_thinking:
            chat_template_kwargs["enable_thinking"] = False
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
            **chat_template_kwargs,
        )
        prompts.append(prompt)
        labels.append(extract_answer(sample["answer"]))
    assert all(l is not None for l in labels), "Some gold labels could not be parsed"
    return prompts, labels


def load_jsonl_questions(
    tokenizer,
    path: str | Path,
    question_indices: list[int],
    *,
    disable_thinking: bool = False,
):
    """Load question-disjoint SeqKD rows without touching GSM8K test."""

    rows = []
    seen = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            question_id = row.get("question_id")
            messages = row.get("sft_messages")
            gold = normalize_numeric_answer(row.get("gold_answer"))
            if question_id in seen:
                continue
            if (
                not isinstance(question_id, str)
                or not isinstance(messages, list)
                or not messages
                or messages[0].get("role") != "user"
                or not isinstance(messages[0].get("content"), str)
                or gold is None
            ):
                raise ValueError(f"Malformed input JSONL row {line_number}")
            seen.add(question_id)
            rows.append((messages[0]["content"], gold))
    prompts, labels = [], []
    for index in question_indices:
        try:
            user_content, gold = rows[index]
        except IndexError as exc:
            raise ValueError(
                f"Question index {index} outside {len(rows)}-row input"
            ) from exc
        kwargs = {}
        if disable_thinking:
            kwargs["enable_thinking"] = False
        prompts.append(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=False,
                add_generation_prompt=True,
                **kwargs,
            )
        )
        labels.append(gold)
    return prompts, labels


def prediction_manifest_path(pred_path: Path) -> Path:
    """Return the effective-configuration manifest path for a prediction file."""
    return pred_path.with_name(f"{pred_path.name}.manifest.json")


def effective_config(args, question_indices: list[int]) -> dict:
    """Capture every result-affecting command-line setting in JSON form."""
    ignored = {"pred_jsonl", "resume_pred_jsonl", "question_indices_jsonl"}
    config = {
        key: value
        for key, value in vars(args).items()
        if key not in ignored
    }
    # Record the resolved values used by the runners, not merely their CLI
    # defaults. This matters when --draft-model or --prompt-tokenizer are unset.
    config["draft_model"] = args.draft_model or DEFAULT_DRAFT_MODEL
    config["prompt_tokenizer"] = args.prompt_tokenizer or args.model
    config["question_indices"] = question_indices
    return config


def write_json_atomically(path: Path, value: dict) -> None:
    """Persist JSON before predictions are emitted."""
    temp_path = path.with_name(f".{path.name}.tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp_path, path)


class PredictionJSONLWriter:
    """Append durable, unique prediction batches and retain resumed rows."""

    def __init__(
        self,
        pred_path: Path,
        *,
        manifest: dict,
        resume: bool,
    ) -> None:
        self.pred_path = pred_path
        self.manifest_path = prediction_manifest_path(pred_path)
        self.rows_by_question_index: dict[int, dict] = {}
        pred_path.parent.mkdir(parents=True, exist_ok=True)

        if resume and pred_path.exists():
            self._load_existing_rows()
            self._validate_manifest(manifest)
            mode = "a"
        else:
            write_json_atomically(self.manifest_path, manifest)
            mode = "w"
        self.file = pred_path.open(mode, encoding="utf-8")

    def _load_existing_rows(self) -> None:
        with self.pred_path.open(encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    question_index = row["question_index"]
                except (json.JSONDecodeError, KeyError) as exc:
                    raise ValueError(
                        f"Invalid prediction row at {self.pred_path}:{line_number}"
                    ) from exc
                if not isinstance(question_index, int) or isinstance(question_index, bool):
                    raise ValueError(
                        f"Invalid question_index at {self.pred_path}:{line_number}"
                    )
                if question_index in self.rows_by_question_index:
                    raise ValueError(
                        "Cannot resume prediction JSONL with duplicate question_index "
                        f"{question_index} ({self.pred_path}:{line_number})"
                    )
                self.rows_by_question_index[question_index] = row

    def _validate_manifest(self, manifest: dict) -> None:
        if not self.manifest_path.exists():
            raise ValueError(
                "Cannot resume prediction JSONL without its effective-config manifest: "
                f"{self.manifest_path}"
            )
        with self.manifest_path.open(encoding="utf-8") as f:
            existing_manifest = json.load(f)
        if existing_manifest != manifest:
            raise ValueError(
                "Prediction JSONL manifest does not match the current effective "
                "configuration; choose a new --pred-jsonl path or omit "
                "--resume-pred-jsonl to replace it."
            )

    @property
    def completed_question_indices(self) -> set[int]:
        return set(self.rows_by_question_index)

    def write_batch(self, rows: list[dict]) -> None:
        question_indices = [row["question_index"] for row in rows]
        if len(question_indices) != len(set(question_indices)):
            raise ValueError("A prediction batch contains duplicate question_index values")
        duplicates = set(question_indices) & self.completed_question_indices
        if duplicates:
            raise ValueError(
                "Refusing to append duplicate question_index values: "
                f"{sorted(duplicates)}"
            )
        for row in rows:
            self.file.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.rows_by_question_index[row["question_index"]] = row
        # Each completed generation batch is recoverable if a later batch or
        # process crashes. fsync makes "flushed" meaningful across a reboot.
        self.file.flush()
        os.fsync(self.file.fileno())

    def close(self) -> None:
        self.file.close()


# ---------------------------------------------------------------------------
# Evaluation runners
# ---------------------------------------------------------------------------


def run_smc_engine_eval(
    args,
    prompts,
    labels,
    question_indices: Optional[list[int]] = None,
    on_batch: Optional[Callable[[int, list, list, list, list], None]] = None,
):
    """Evaluation using the dedicated SMCEngine (offline, no tokenizer manager)."""
    from smcsd.engine import SMCEngine

    if question_indices is None:
        question_indices = list(range(len(prompts)))
    draft_model = args.draft_model or DEFAULT_DRAFT_MODEL
    cross_artifact_path = args.cross_tokenizer_artifact_path
    if (
        getattr(args, "cross_tokenizer", False)
        and args.cross_tokenizer_mode == "hybrid"
        and cross_artifact_path is None
    ):
        raise ValueError(
            "--cross-tokenizer-mode hybrid requires "
            "--cross-tokenizer-artifact-path"
        )
    if getattr(args, "cross_tokenizer", False) and args.cross_tokenizer_stats_interval:
        os.environ.setdefault(
            "SMC_CROSS_STATS_INTERVAL", str(args.cross_tokenizer_stats_interval)
        )
    engine_kwargs = dict(
        model_path=args.model,
        draft_model_path=draft_model,
        n_particles=args.particles,
        gamma=args.gamma,
        draft_temperature=(
            args.draft_temperature
            if args.draft_temperature is not None
            else args.temperature
        ),
        target_temperature=(
            args.target_temperature
            if args.target_temperature is not None
            else args.temperature
        ),
        power_alpha=args.power_alpha,
        final_selection=args.final_selection,
        trust_remote_code=True,
        page_size=1,
        attention_backend=args.attention_backend,
    )
    if args.seed is not None:
        engine_kwargs["random_seed"] = args.seed
    if args.resample_threshold is not None:
        engine_kwargs["resample_threshold"] = args.resample_threshold
    if args.mem_fraction_static is not None:
        engine_kwargs["mem_fraction_static"] = args.mem_fraction_static
    if args.cuda_graph_max_bs is not None:
        engine_kwargs["cuda_graph_max_bs"] = args.cuda_graph_max_bs
    if args.max_running_requests is not None:
        engine_kwargs["max_running_requests"] = args.max_running_requests
    else:
        engine_kwargs["max_running_requests"] = max(args.particles + 4, 16)
    if args.max_total_tokens is not None:
        engine_kwargs["max_total_tokens"] = args.max_total_tokens
    if getattr(args, "dtype", None):
        engine_kwargs["dtype"] = args.dtype
    if getattr(args, "disable_cuda_graph", False):
        engine_kwargs["disable_cuda_graph"] = True
    if getattr(args, "disable_piecewise_cuda_graph", False):
        engine_kwargs["disable_piecewise_cuda_graph"] = True
    if getattr(args, "tp_size", 1) and args.tp_size > 1:
        engine_kwargs["tp_size"] = args.tp_size
    if getattr(args, "disable_custom_all_reduce", False):
        engine_kwargs["disable_custom_all_reduce"] = True
    if getattr(args, "base_gpu_id", None) is not None:
        engine_kwargs["base_gpu_id"] = args.base_gpu_id
    if getattr(args, "cross_tokenizer", False):
        engine_kwargs["cross_tokenizer"] = True
        engine_kwargs["draft_tokenizer_path"] = args.draft_tokenizer_path
        engine_kwargs["cross_tokenizer_artifact_path"] = cross_artifact_path
        engine_kwargs["cross_tokenizer_mode"] = args.cross_tokenizer_mode
    sampling_params = {
        "max_new_tokens": args.max_new_tokens,
        "ignore_eos": args.ignore_eos,
        "temperature": args.temperature,
    }

    diagnostics = []
    response_texts = []
    with SMCEngine(**engine_kwargs) as engine:
        if args.warmup_questions > 0:
            warmup_prompts, _ = load_gsm8k(
                AutoTokenizer.from_pretrained(args.model),
                args.warmup_questions,
                start=args.warmup_start,
                disable_thinking=args.disable_thinking,
            )
            print(
                f"Running warmup: {len(warmup_prompts)} questions "
                f"(start={args.warmup_start}); not counted in metrics."
            )
            for start in range(0, len(warmup_prompts), args.batch_size):
                batch = warmup_prompts[start : start + args.batch_size]
                engine.generate(batch, sampling_params)

        preds = []
        rids = []
        total_output_tokens = 0
        tic = time.perf_counter()
        for start in range(0, len(prompts), args.batch_size):
            batch_result_start = len(preds)
            batch = prompts[start : start + args.batch_size]
            outputs = engine.generate(batch, sampling_params)
            if not isinstance(outputs, list):
                outputs = [outputs]
            for i, output in enumerate(outputs):
                qi = start + i
                if qi < 3:
                    ntok = output["completion_tokens"]
                    print(f"--- Q{question_indices[qi]} ({ntok} tokens) ---")
                    print(output["text"][:400])
                    print()
                preds.append(extract_answer(output["text"]))
                rids.append(output.get("rid"))
                response_texts.append(output["text"])
                diagnostics.append(smc_particle_metrics(output, labels[qi]))
                total_output_tokens += output["completion_tokens"]
            if on_batch is not None:
                on_batch(
                    start,
                    preds[batch_result_start:],
                    rids[batch_result_start:],
                    response_texts[batch_result_start:],
                    diagnostics[batch_result_start:],
                )
            elapsed = time.perf_counter() - tic
            correct = sum(
                p == l for p, l in zip(preds, labels[: len(preds)])
            )
            print(
                f"\r[{len(preds)}/{len(prompts)}] "
                f"acc={correct}/{len(preds)} ({correct / len(preds):.1%}) "
                f"tps={total_output_tokens / elapsed:.0f} "
                f"elapsed={elapsed:.0f}s",
                flush=True,
            )
        latency = time.perf_counter() - tic

    return preds, rids, response_texts, diagnostics, total_output_tokens, latency


def run_baseline_eval(
    args,
    prompts,
    labels,
    question_indices: Optional[list[int]] = None,
    on_batch: Optional[Callable[[int, list, list, list, list], None]] = None,
):
    """Baseline (vanilla generation, no speculative decoding) evaluation."""
    import sglang as sgl

    if question_indices is None:
        question_indices = list(range(len(prompts)))
    engine_kwargs = dict(
        model_path=args.model,
        trust_remote_code=True,
        attention_backend=args.attention_backend,
    )
    if args.seed is not None:
        engine_kwargs["random_seed"] = args.seed
    if args.mem_fraction_static is not None:
        engine_kwargs["mem_fraction_static"] = args.mem_fraction_static
    if args.cuda_graph_max_bs is not None:
        engine_kwargs["cuda_graph_max_bs"] = args.cuda_graph_max_bs
    if args.max_running_requests is not None:
        engine_kwargs["max_running_requests"] = args.max_running_requests
    if args.max_total_tokens is not None:
        engine_kwargs["max_total_tokens"] = args.max_total_tokens
    if getattr(args, "disable_cuda_graph", False):
        engine_kwargs["disable_cuda_graph"] = True
    if getattr(args, "tp_size", 1) and args.tp_size > 1:
        engine_kwargs["tp_size"] = args.tp_size
    if getattr(args, "disable_custom_all_reduce", False):
        engine_kwargs["disable_custom_all_reduce"] = True
    if getattr(args, "base_gpu_id", None) is not None:
        engine_kwargs["base_gpu_id"] = args.base_gpu_id

    sampling_params = {
        "max_new_tokens": args.max_new_tokens,
        "ignore_eos": args.ignore_eos,
        "temperature": args.temperature,
    }

    with sgl.Engine(**engine_kwargs) as engine:
        if args.warmup_questions > 0:
            warmup_prompts, _ = load_gsm8k(
                AutoTokenizer.from_pretrained(args.model),
                args.warmup_questions,
                start=args.warmup_start,
                disable_thinking=args.disable_thinking,
            )
            print(
                f"Running warmup: {len(warmup_prompts)} questions "
                f"(start={args.warmup_start}); not counted in metrics."
            )
            for start in range(0, len(warmup_prompts), args.batch_size):
                batch = warmup_prompts[start : start + args.batch_size]
                engine.generate(batch, sampling_params)

        preds = []
        rids = []
        response_texts = []
        total_output_tokens = 0
        tic = time.perf_counter()
        for start in range(0, len(prompts), args.batch_size):
            batch_result_start = len(preds)
            batch = prompts[start : start + args.batch_size]
            outputs = engine.generate(batch, sampling_params)
            for i, output in enumerate(outputs):
                qi = start + i
                if qi < 3:
                    ntok = output["meta_info"]["completion_tokens"]
                    print(f"--- Q{question_indices[qi]} ({ntok} tokens) ---")
                    print(output["text"][:400])
                    print()
                preds.append(extract_answer(output["text"]))
                rids.append(None)
                response_texts.append(output["text"])
                total_output_tokens += output["meta_info"][
                    "completion_tokens"
                ]
            if on_batch is not None:
                on_batch(
                    start,
                    preds[batch_result_start:],
                    rids[batch_result_start:],
                    response_texts[batch_result_start:],
                    [None] * (len(preds) - batch_result_start),
                )
            elapsed = time.perf_counter() - tic
            correct = sum(
                p == l for p, l in zip(preds, labels[: len(preds)])
            )
            print(
                f"\r[{len(preds)}/{len(prompts)}] "
                f"acc={correct}/{len(preds)} ({correct / len(preds):.1%}) "
                f"tps={total_output_tokens / elapsed:.0f} "
                f"elapsed={elapsed:.0f}s",
                flush=True,
            )
        latency = time.perf_counter() - tic

    return preds, rids, response_texts, [None] * len(preds), total_output_tokens, latency


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args):
    mode_label = {
        "smc_engine": "SMCEngine (dedicated offline)",
        "baseline": "Baseline (vanilla)",
    }
    print(f"Mode: {mode_label[args.mode]} | Model: {args.model}")
    if args.mode == "smc_engine":
        draft = args.draft_model or DEFAULT_DRAFT_MODEL
        print(
            f"  particles={args.particles}, gamma={args.gamma}, "
            f"temperature={args.temperature}, draft={draft}"
        )
    print(
        f"  num_questions={args.num_questions}, eval_offset={args.eval_offset}, "
        f"warmup_questions={args.warmup_questions}, "
        f"max_new_tokens={args.max_new_tokens}, ignore_eos={args.ignore_eos}"
    )
    if args.prompt_tokenizer:
        print(f"  prompt_tokenizer={args.prompt_tokenizer}")
    if args.disable_thinking:
        print("  chat_template: enable_thinking=False")
    print()

    if args.seed is not None:
        np.random.seed(args.seed)

    # Load tokenizer and data (shared across all modes)
    prompt_tokenizer_path = args.prompt_tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(prompt_tokenizer_path)
    question_indices = None
    if args.question_indices_jsonl:
        with Path(args.question_indices_jsonl).open(encoding="utf-8") as f:
            question_indices = [
                int(json.loads(line)["question_index"])
                for line in f
                if line.strip()
            ]
        if not question_indices:
            raise ValueError("--question-indices-jsonl did not contain any indices")
    selected_question_indices = (
        question_indices
        if question_indices is not None
        else list(range(args.eval_offset, args.eval_offset + args.num_questions))
    )
    if len(selected_question_indices) != len(set(selected_question_indices)):
        raise ValueError("Requested evaluation contains duplicate question_index values")
    if args.input_jsonl:
        prompts, labels = load_jsonl_questions(
            tokenizer,
            args.input_jsonl,
            selected_question_indices,
            disable_thinking=args.disable_thinking,
        )
    else:
        prompts, labels = load_gsm8k(
            tokenizer,
            len(selected_question_indices),
            start=args.eval_offset,
            disable_thinking=args.disable_thinking,
            question_indices=selected_question_indices,
        )

    if args.resume_pred_jsonl and not args.pred_jsonl:
        raise ValueError("--resume-pred-jsonl requires --pred-jsonl")

    pred_writer = None
    if args.pred_jsonl:
        pred_path = Path(args.pred_jsonl)
        manifest = {
            "manifest_version": 1,
            "effective_config": effective_config(args, selected_question_indices),
        }
        pred_writer = PredictionJSONLWriter(
            pred_path,
            manifest=manifest,
            resume=args.resume_pred_jsonl,
        )

    pending_positions = list(range(len(prompts)))
    if pred_writer is not None:
        completed_indices = pred_writer.completed_question_indices
        if completed_indices - set(selected_question_indices):
            pred_writer.close()
            raise ValueError(
                "Prediction JSONL contains question indexes outside this evaluation "
                "scope; choose a new --pred-jsonl path."
            )
        pending_positions = [
            position
            for position, question_index in enumerate(selected_question_indices)
            if question_index not in completed_indices
        ]
    pending_prompts = [prompts[position] for position in pending_positions]
    pending_labels = [labels[position] for position in pending_positions]
    pending_question_indices = [
        selected_question_indices[position] for position in pending_positions
    ]

    def persist_batch(start, batch_preds, batch_rids, batch_texts, batch_diagnostics):
        assert pred_writer is not None
        rows = []
        for i, (pred, rid, response_text, diagnostic) in enumerate(
            zip(batch_preds, batch_rids, batch_texts, batch_diagnostics)
        ):
            question_index = pending_question_indices[start + i]
            row = {
                "question_index": question_index,
                "rid": rid,
                "pred": None if pred is None else str(pred),
                "gold": str(pending_labels[start + i]),
                "correct": pred == pending_labels[start + i],
            }
            if args.include_response_text:
                row["response_text"] = response_text
            if diagnostic is not None:
                row["smc"] = diagnostic
            rows.append(row)
        pred_writer.write_batch(rows)

    # Run only unfinished questions. The writer flushes each generated batch
    # before the next one begins, so a later interruption is resumable.
    try:
        if pending_prompts:
            if args.mode == "smc_engine":
                (
                    preds,
                    rids,
                    response_texts,
                    diagnostics,
                    total_tokens,
                    latency,
                ) = run_smc_engine_eval(
                    args,
                    pending_prompts,
                    pending_labels,
                    pending_question_indices,
                    persist_batch if pred_writer is not None else None,
                )
            else:
                (
                    preds,
                    rids,
                    response_texts,
                    diagnostics,
                    total_tokens,
                    latency,
                ) = run_baseline_eval(
                    args,
                    pending_prompts,
                    pending_labels,
                    pending_question_indices,
                    persist_batch if pred_writer is not None else None,
                )
        else:
            print("All requested question indexes already exist; no generation needed.")
            preds, rids, response_texts, diagnostics = [], [], [], []
            total_tokens, latency = 0, 0.0
    finally:
        if pred_writer is not None:
            pred_writer.close()

    if pred_writer is not None:
        preds = [
            normalize_numeric_answer(
                pred_writer.rows_by_question_index[question_index].get("pred")
            )
            for question_index in selected_question_indices
        ]

    # Report
    correct = sum(p == l for p, l in zip(preds, labels))
    invalid = sum(p is None for p in preds)
    n = len(preds)

    # Optional fingerprint dump for bit-identicality / determinism checks.
    _dump_path = os.environ.get("SMC_DUMP_PREDS")
    if _dump_path:
        with open(_dump_path, "w") as _f:
            json.dump([None if p is None else str(p) for p in preds], _f)

    print(f"\n{'=' * 55}")
    print(f"  {mode_label[args.mode]}")
    if args.mode == "smc_engine":
        print(f"  N={args.particles}, γ={args.gamma}, temp={args.temperature}")
    print(f"{'=' * 55}")
    print(f"  Accuracy:          {correct}/{n} ({100 * correct / n:.1f}%)")
    print(f"  Invalid:           {invalid}/{n} ({100 * invalid / n:.1f}%)")
    throughput = total_tokens / latency if latency else 0.0
    print(f"  Output throughput: {throughput:.1f} tok/s")
    print(f"  Total tokens:      {total_tokens}")
    print(f"  Wall time:         {latency:.1f}s")
    persisted_diagnostics = diagnostics
    if pred_writer is not None:
        persisted_diagnostics = [
            pred_writer.rows_by_question_index[question_index].get("smc")
            for question_index in selected_question_indices
        ]
    smc_summary = aggregate_smc_diagnostics(persisted_diagnostics)
    if smc_summary:
        print(
            "  Terminal ESS/N:    "
            f"{smc_summary['mean_terminal_ess_fraction']:.3f}"
        )
        print(
            "  Correct wt mass:   "
            f"{smc_summary['mean_correct_weight_mass']:.3f}"
        )
        print(
            "  Nonfinite weights: "
            f"{smc_summary['nonfinite_weight_count']}"
        )
    print(f"{'=' * 55}")

    if pred_writer is not None:
        summary_path = pred_path.with_name(f"{pred_path.name}.summary.json")
        summary_payload = {
            "summary_version": 1,
            "effective_config": effective_config(
                args, selected_question_indices
            ),
            "accuracy": {
                "correct": correct,
                "total": n,
                "rate": correct / n if n else None,
                "invalid": invalid,
                "invalid_rate": invalid / n if n else None,
            },
            "systems": {
                "output_throughput_tps": throughput,
                "total_tokens": total_tokens,
                "wall_time_seconds": latency,
            },
            "smc_diagnostics": smc_summary,
        }
        summary_path.write_text(
            json.dumps(summary_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"  Summary JSON:      {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Core
    parser.add_argument(
        "--mode",
        choices=["baseline", "smc_engine"],
        default="smc_engine",
        help="baseline = vanilla, smc_engine = dedicated SMCEngine (default: smc_engine)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"target model path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--draft-model",
        type=str,
        default=DEFAULT_DRAFT_MODEL,
        help=f"draft model path (default: {DEFAULT_DRAFT_MODEL})",
    )
    parser.add_argument(
        "--prompt-tokenizer",
        type=str,
        default=None,
        help=(
            "Tokenizer/chat template used to construct benchmark prompts. "
            "Defaults to --model; use the target tokenizer for draft-prompt controls."
        ),
    )

    # SMC parameters (used by smc_engine mode)
    smc_grp = parser.add_argument_group("SMC parameters")
    smc_grp.add_argument("--particles", "-N", type=int, default=4)
    smc_grp.add_argument("--gamma", "-g", type=int, default=4)
    smc_grp.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="draft temperature (default: 0.7)",
    )
    smc_grp.add_argument(
        "--draft-temperature",
        type=float,
        default=None,
        help="override draft proposal temperature (defaults to --temperature)",
    )
    smc_grp.add_argument(
        "--target-temperature",
        type=float,
        default=None,
        help="override target temperature (defaults to --temperature)",
    )
    smc_grp.add_argument(
        "--seed", type=int, default=None, help="numpy seed for reproducibility"
    )
    smc_grp.add_argument(
        "--resample-threshold", type=float, default=None,
        help="ESS resample threshold (default: 0.5, use 0 to disable resampling)",
    )
    smc_grp.add_argument(
        "--power-alpha",
        type=float,
        default=1.0,
        help="SMC target power alpha for p^alpha weighting (default: 1.0)",
    )
    smc_grp.add_argument(
        "--final-selection",
        choices=["posterior_sample", "max_weight"],
        default="posterior_sample",
        help=(
            "Terminal particle policy. posterior_sample preserves SMC sampling; "
            "max_weight is a deterministic quality-oriented ablation."
        ),
    )
    # Benchmark
    bench = parser.add_argument_group("benchmark")
    bench.add_argument("--num-questions", type=int, default=80)
    bench.add_argument(
        "--input-jsonl",
        type=str,
        default=None,
        help=(
            "Optional question-disjoint SeqKD JSONL. When set, local row "
            "indices are evaluated and GSM8K test is not loaded."
        ),
    )
    bench.add_argument(
        "--eval-offset",
        type=int,
        default=0,
        help="First GSM8K test index to include in measured evaluation.",
    )
    bench.add_argument(
        "--warmup-questions",
        type=int,
        default=0,
        help="Run this many unmeasured questions before starting timed evaluation.",
    )
    bench.add_argument(
        "--warmup-start",
        type=int,
        default=0,
        help="First GSM8K test index to use for warmup questions.",
    )
    bench.add_argument("--max-new-tokens", type=int, default=512)
    bench.add_argument("--batch-size", type=int, default=1)
    bench.add_argument(
        "--ignore-eos",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="pass ignore_eos through engine sampling_params for throughput comparisons",
    )
    bench.add_argument(
        "--disable-thinking",
        action="store_true",
        default=False,
        help="Pass enable_thinking=False to Qwen-style chat templates.",
    )
    bench.add_argument(
        "--pred-jsonl",
        type=str,
        default=None,
        help="Optional path for per-question prediction/correctness JSONL.",
    )
    bench.add_argument(
        "--resume-pred-jsonl",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Resume --pred-jsonl by skipping already persisted question indexes "
            "(default: false, which replaces the output)."
        ),
    )
    bench.add_argument(
        "--include-response-text",
        action="store_true",
        help="Include complete generated response text in --pred-jsonl rows.",
    )
    bench.add_argument(
        "--question-indices-jsonl",
        type=str,
        default=None,
        help="Evaluate GSM8K test indices read from JSONL rows with question_index.",
    )

    # Engine overrides (smc_engine / baseline modes)
    eng = parser.add_argument_group("engine overrides (smc_engine/baseline)")
    eng.add_argument("--dtype", type=str, default=None,
                     help="model dtype override (e.g. bfloat16, float16)")
    eng.add_argument("--attention-backend", type=str, default="triton",
                      choices=["triton", "fa3"],
                      help="attention backend for smc_engine mode (default: triton)")
    eng.add_argument("--mem-fraction-static", type=float, default=0.4)
    eng.add_argument("--cuda-graph-max-bs", type=int, default=128)
    eng.add_argument("--max-running-requests", type=int, default=16)
    eng.add_argument(
        "--max-total-tokens",
        type=int,
        default=None,
        help="Cap the KV-token memory pool size (useful for dense hybrid draft smoke tests).",
    )
    eng.add_argument(
        "--disable-cuda-graph",
        action="store_true",
        default=False,
        help="Disable CUDA graphs (faster startup, slower decode; useful for smoke tests).",
    )
    eng.add_argument(
        "--disable-piecewise-cuda-graph",
        action="store_true",
        default=False,
        help="Disable SGLang piecewise CUDA graphs (useful when ninja/JIT is unavailable).",
    )
    eng.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor-parallel size for the target (and draft, if shared).",
    )
    eng.add_argument(
        "--disable-custom-all-reduce",
        action="store_true",
        default=False,
        help="Use NCCL all-reduce instead of the custom kernel (needed for some "
        "multi-GPU TP setups where custom all-reduce fails during CUDA graph capture).",
    )
    eng.add_argument(
        "--base-gpu-id",
        type=int,
        default=0,
        help="Base logical GPU id after CUDA_VISIBLE_DEVICES remapping.",
    )
    eng.add_argument(
        "--cross-tokenizer",
        action="store_true",
        default=False,
        help="Enable cross-tokenizer target/draft execution in smc_engine mode.",
    )
    eng.add_argument("--draft-tokenizer-path", type=str, default=None)
    eng.add_argument("--cross-tokenizer-artifact-path", type=str, default=None)
    eng.add_argument(
        "--cross-tokenizer-mode",
        choices=["live", "hybrid"],
        default="hybrid",
    )
    eng.add_argument(
        "--cross-tokenizer-stats-interval",
        type=int,
        default=1000,
        help="Print mapper aggregate counters every N mapped rows in cross-tokenizer mode.",
    )

    args = parser.parse_args()
    main(args)
