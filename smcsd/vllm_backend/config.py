from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SMCConfig:
    """Configuration for SMC speculative decoding."""

    draft_model_path: str
    n_particles: int = 4
    gamma: int = 4
    resample_threshold: float = 0.5
    resample_method: str = "systematic"
    # Capture the draft decode loop  as full CUDA graphs.
    draft_cudagraph: bool = True
