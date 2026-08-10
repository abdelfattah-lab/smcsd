"""Cross-tokenizer utilities for SMC speculative decoding.

The runtime SMC path keeps target and draft token spaces separate.  This
package contains the tokenizer-facing pieces that can be tested without loading
models: canonicalization, offline mapping artifacts, bounded alignment, and
runtime mapping helpers.
"""

from smcsd.cross_tokenizer.artifacts import (
    MappingArtifact,
    MappingMetadata,
    MappingStats,
    load_artifact,
    save_artifact,
)
from smcsd.cross_tokenizer.mapper import (
    BoundaryState,
    MappingResult,
    TokenMapper,
)

__all__ = [
    "BoundaryState",
    "MappingArtifact",
    "MappingMetadata",
    "MappingResult",
    "MappingStats",
    "TokenMapper",
    "load_artifact",
    "save_artifact",
]
