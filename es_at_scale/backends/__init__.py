"""Pluggable inference backends for ES.

Import backend implementations LAZILY (only when selected) so that the vLLM and
SGLang dependency trees -- which live in separate conda environments -- are never
imported into the same process.
"""

from es_at_scale.backends.base import (
    Completion,
    ESBackend,
    GenerationOutput,
    SamplingConfig,
)

__all__ = [
    "Completion",
    "ESBackend",
    "GenerationOutput",
    "SamplingConfig",
    "get_backend",
]


def get_backend(name: str, **cfg) -> ESBackend:
    """Construct a backend by name. ``**cfg`` is forwarded to its constructor."""
    name = name.lower()
    if name == "vllm":
        from es_at_scale.backends.vllm_backend import VLLMBackend

        return VLLMBackend(**cfg)
    if name == "sglang":
        from es_at_scale.backends.sglang_backend import SGLangBackend

        return SGLangBackend(**cfg)
    raise ValueError(f"Unknown backend '{name}'. Choose from: vllm, sglang.")
