"""Freeze and validate a reproducible Qwen3/Llama-70B SMCSD contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer

from smcsd.cross_tokenizer.artifacts import (
    load_artifact,
    tokenizer_fingerprint,
    tokenizer_semantic_fingerprint,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def architecture_signature(config) -> dict[str, object]:
    """Fields that must remain unchanged by a merged LoRA checkpoint."""
    keys = (
        "model_type",
        "architectures",
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "max_position_embeddings",
        "rope_theta",
        "rope_scaling",
        "tie_word_embeddings",
    )
    return {key: getattr(config, key, None) for key in keys}


def validate_tokenizer_model_compatibility(tokenizer, config, label: str) -> None:
    vocab = tokenizer.get_vocab()
    if not vocab:
        raise ValueError(f"{label} tokenizer has an empty vocabulary.")
    max_token_id = max(int(token_id) for token_id in vocab.values())
    model_vocab_size = int(config.vocab_size)
    if max_token_id >= model_vocab_size:
        raise ValueError(
            f"{label} tokenizer id {max_token_id} exceeds model vocab size "
            f"{model_vocab_size}."
        )


def local_checkpoint_manifest(model_ref: str) -> dict[str, object] | None:
    path = Path(model_ref).expanduser()
    if not path.is_dir():
        return None
    if (path / "adapter_config.json").is_file():
        raise ValueError(
            f"{path} looks like an unmerged PEFT adapter. Pass a merged model "
            "directory as --draft-model."
        )
    config_path = path / "config.json"
    if not config_path.is_file():
        raise ValueError(f"Local checkpoint is missing {config_path}.")

    weight_files = sorted(path.glob("*.safetensors")) + sorted(
        path.glob("pytorch_model*.bin")
    )
    if not weight_files:
        raise ValueError(f"Local checkpoint {path} contains no model weight files.")

    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = path / index_name
        if not index_path.is_file():
            continue
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        referenced = {str(name) for name in payload.get("weight_map", {}).values()}
        missing = sorted(name for name in referenced if not (path / name).is_file())
        if missing:
            raise ValueError(
                f"Checkpoint index {index_path} references missing shards: {missing}"
            )

    files = [config_path, *weight_files]
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        candidate = path / name
        if candidate.is_file():
            files.append(candidate)
    return {
        "path": str(path.resolve()),
        "files": [
            {
                "name": file.relative_to(path).as_posix(),
                "size": file.stat().st_size,
                "sha256": sha256_file(file),
            }
            for file in files
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--target-model", default="meta-llama/Llama-3.1-70B-Instruct"
    )
    parser.add_argument(
        "--draft-model", default="Qwen/Qwen3-4B-Instruct-2507"
    )
    parser.add_argument(
        "--target-tokenizer",
        default=None,
        help="Tokenizer source used by runtime; defaults to --target-model.",
    )
    parser.add_argument(
        "--draft-tokenizer",
        default=None,
        help="Tokenizer source used by runtime; defaults to --draft-model.",
    )
    parser.add_argument(
        "--draft-base-model",
        default=None,
        help=(
            "Base model used to produce a merged local draft checkpoint. "
            "Required when --draft-model is a local directory."
        ),
    )
    parser.add_argument("--particles", type=int, default=8)
    parser.add_argument("--gamma", type=int, default=8)
    parser.add_argument("--target-temperature", type=float, default=0.7)
    parser.add_argument("--draft-temperature", type=float, default=0.9)
    parser.add_argument("--power-alpha", type=float, default=1.0)
    parser.add_argument("--resample-threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.particles != 8 or args.gamma != 8:
        raise ValueError("The Qwen3 SMCSD contract requires N=8 and gamma=8.")
    if not args.artifact.is_file():
        raise FileNotFoundError(args.artifact)

    target_tokenizer_source = args.target_tokenizer or args.target_model
    draft_tokenizer_source = args.draft_tokenizer or args.draft_model
    draft_config = AutoConfig.from_pretrained(
        args.draft_model, trust_remote_code=True
    )
    target_config = AutoConfig.from_pretrained(
        args.target_model, trust_remote_code=True
    )
    draft_tokenizer = AutoTokenizer.from_pretrained(
        draft_tokenizer_source, trust_remote_code=True
    )
    target_tokenizer = AutoTokenizer.from_pretrained(
        target_tokenizer_source, trust_remote_code=True
    )
    validate_tokenizer_model_compatibility(
        draft_tokenizer, draft_config, "Draft"
    )
    validate_tokenizer_model_compatibility(
        target_tokenizer, target_config, "Target"
    )

    draft_is_local = Path(args.draft_model).expanduser().is_dir()
    if draft_is_local and args.draft_base_model is None:
        raise ValueError(
            "A local/merged --draft-model requires --draft-base-model so the "
            "freezer can verify its architecture and tokenizer lineage."
        )
    draft_manifest = local_checkpoint_manifest(args.draft_model)
    base_provenance = None
    if args.draft_base_model is not None:
        base_config = AutoConfig.from_pretrained(
            args.draft_base_model, trust_remote_code=True
        )
        if architecture_signature(draft_config) != architecture_signature(base_config):
            raise ValueError(
                "Draft checkpoint architecture does not match --draft-base-model."
            )
        base_tokenizer = AutoTokenizer.from_pretrained(
            args.draft_base_model, trust_remote_code=True
        )
        if tokenizer_semantic_fingerprint(
            draft_tokenizer
        ) != tokenizer_semantic_fingerprint(base_tokenizer):
            raise ValueError(
                "Runtime draft tokenizer semantics do not match --draft-base-model."
            )
        base_provenance = {
            "model": args.draft_base_model,
            "config_sha256": stable_json_hash(base_config.to_dict()),
            "architecture": architecture_signature(base_config),
            "tokenizer_semantic_fingerprint": tokenizer_semantic_fingerprint(
                base_tokenizer
            ),
            "commit_hash": getattr(base_config, "_commit_hash", None),
        }

    artifact = load_artifact(args.artifact)
    artifact.validate_for(draft_tokenizer, target_tokenizer)

    # Every special mapping must be an actual special token on both sides.
    draft_special = set(draft_tokenizer.all_special_ids)
    target_special = set(target_tokenizer.all_special_ids)
    invalid_specials = {
        draft_id: target_id
        for draft_id, target_id in artifact.special_token_map.items()
        if int(draft_id) not in draft_special or int(target_id) not in target_special
    }
    if invalid_specials:
        raise ValueError(
            "Artifact contains invalid special-token mappings: "
            f"{invalid_specials}"
        )

    contract = {
        "schema_version": 2,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "target_model": args.target_model,
        "draft_model": args.draft_model,
        "target_tokenizer": target_tokenizer_source,
        "draft_tokenizer": draft_tokenizer_source,
        "target_tokenizer_fingerprint": tokenizer_fingerprint(target_tokenizer),
        "draft_tokenizer_fingerprint": tokenizer_fingerprint(draft_tokenizer),
        "target_model_provenance": {
            "config_sha256": stable_json_hash(target_config.to_dict()),
            "architecture": architecture_signature(target_config),
            "commit_hash": getattr(target_config, "_commit_hash", None),
        },
        "draft_model_provenance": {
            "config_sha256": stable_json_hash(draft_config.to_dict()),
            "architecture": architecture_signature(draft_config),
            "commit_hash": getattr(draft_config, "_commit_hash", None),
            "local_checkpoint": draft_manifest,
            "base": base_provenance,
        },
        "artifact": {
            "path": str(args.artifact.resolve()),
            "sha256": sha256_file(args.artifact),
            "metadata": artifact.metadata.__dict__,
            "special_token_map_size": len(artifact.special_token_map),
            "blocked_specials": artifact.stats.blocked_specials,
            "ngram_entries": len(artifact.ngram_map),
        },
        "runtime": {
            "particles": args.particles,
            "gamma": args.gamma,
            "target_temperature": args.target_temperature,
            "draft_temperature": args.draft_temperature,
            "power_alpha": args.power_alpha,
            "resample_threshold": args.resample_threshold,
            "final_selection": "posterior_sample",
            "cross_tokenizer_mode": "hybrid",
            "online_ngram_mutation": False,
            "SMC_CROSS_ONLINE_NGRAM": "0",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Validated and froze contract: {args.output}")


if __name__ == "__main__":
    main()
