"""Offline mapping artifact schema for cross-tokenizer SMC."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


SCHEMA_VERSION = 1


def _stable_json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return str(value)


def _tokenizer_semantic_payload(tokenizer: Any) -> Dict[str, Any]:
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
    normalized_vocab = sorted(
        (str(token), int(token_id)) for token, token_id in vocab.items()
    )
    backend = getattr(tokenizer, "backend_tokenizer", None)
    backend_json = None
    if backend is not None and hasattr(backend, "to_str"):
        backend_json = backend.to_str()
    return {
        "class": tokenizer.__class__.__name__,
        "vocab_size": len(normalized_vocab),
        "vocab_sha256": _stable_json_hash(normalized_vocab),
        "backend_sha256": (
            hashlib.sha256(backend_json.encode("utf-8")).hexdigest()
            if backend_json is not None
            else None
        ),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "special_tokens_map": _jsonable(getattr(tokenizer, "special_tokens_map", None)),
    }


def tokenizer_semantic_fingerprint(tokenizer: Any) -> str:
    """Fingerprint token ids, tokenizer rules, and special-token semantics."""

    return _stable_json_hash(_tokenizer_semantic_payload(tokenizer))


def tokenizer_fingerprint(tokenizer: Any) -> str:
    """Stable tokenizer fingerprint including source and full token-id map."""

    payload = _tokenizer_semantic_payload(tokenizer)
    payload["name_or_path"] = getattr(tokenizer, "name_or_path", None)
    init_kwargs = getattr(tokenizer, "init_kwargs", None)
    if isinstance(init_kwargs, Mapping):
        payload["init_kwargs"] = _jsonable(init_kwargs)
    return _stable_json_hash(payload)


@dataclass
class MappingMetadata:
    draft_tokenizer: str
    target_tokenizer: str
    draft_fingerprint: str
    target_fingerprint: str
    max_single_token_span: int = 4
    max_ngram: int = 8
    top_k: int = 4
    schema_version: int = SCHEMA_VERSION


@dataclass
class MappingStats:
    draft_vocab_size: int = 0
    target_vocab_size: int = 0
    exact_matches: int = 0
    one_to_many: int = 0
    blocked_specials: int = 0
    unmapped: int = 0
    ngram_entries: int = 0


@dataclass
class MappingArtifact:
    metadata: MappingMetadata
    stats: MappingStats
    # JSON stores integer keys as strings; accessors normalize both forms.
    single_map: Dict[str, List[int]] = field(default_factory=dict)
    special_token_map: Dict[str, int] = field(default_factory=dict)
    # Encoded as "1 2 3" -> [target ids].
    ngram_map: Dict[str, List[int]] = field(default_factory=dict)
    canonical_vocab_index: Dict[str, Dict[str, List[int]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._max_ngram_len_cache: Optional[int] = None

    def get_single(self, draft_id: int) -> Optional[List[int]]:
        value = self.single_map.get(str(int(draft_id)))
        return list(value) if value is not None else None

    def get_ngram(self, draft_ids: Iterable[int]) -> Optional[List[int]]:
        key = encode_ngram_key(tuple(int(x) for x in draft_ids))
        value = self.ngram_map.get(key)
        return list(value) if value is not None else None

    def max_ngram_len(self) -> int:
        if not self.ngram_map:
            return 0
        if self._max_ngram_len_cache is None:
            self._max_ngram_len_cache = max(
                len(decode_ngram_key(key)) for key in self.ngram_map
            )
        return self._max_ngram_len_cache

    def validate_for(self, draft_tokenizer: Any, target_tokenizer: Any) -> None:
        draft_fp = tokenizer_fingerprint(draft_tokenizer)
        target_fp = tokenizer_fingerprint(target_tokenizer)
        if draft_fp != self.metadata.draft_fingerprint:
            raise ValueError("Mapping artifact draft tokenizer fingerprint mismatch.")
        if target_fp != self.metadata.target_fingerprint:
            raise ValueError("Mapping artifact target tokenizer fingerprint mismatch.")


def encode_ngram_key(ids: Tuple[int, ...]) -> str:
    return " ".join(str(int(x)) for x in ids)


def decode_ngram_key(key: str) -> Tuple[int, ...]:
    if not key:
        return ()
    return tuple(int(part) for part in key.split())


def save_artifact(artifact: MappingArtifact, path: str | Path) -> None:
    payload = asdict(artifact)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_artifact(path: str | Path) -> MappingArtifact:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    metadata = MappingMetadata(**payload["metadata"])
    if metadata.schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported mapping schema {metadata.schema_version}; expected {SCHEMA_VERSION}."
        )
    return MappingArtifact(
        metadata=metadata,
        stats=MappingStats(**payload["stats"]),
        single_map={str(k): list(v) for k, v in payload.get("single_map", {}).items()},
        special_token_map={
            str(k): int(v) for k, v in payload.get("special_token_map", {}).items()
        },
        ngram_map={str(k): list(v) for k, v in payload.get("ngram_map", {}).items()},
        canonical_vocab_index=payload.get("canonical_vocab_index", {}),
    )
