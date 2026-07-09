"""Backend-neutral interface for ES inference engines.

The trainer holds one :class:`ESBackend` and never imports vllm/sglang. A backend
owns N inference engines (one population member evaluated per engine at a time)
and exposes the small set of operations ES needs: greedy generation, per-member
weight perturbation/restore, the committed ES update, and checkpoint save/load.

Weight-op and generation calls are **async**: each returns an opaque ``Handle``
that :meth:`ESBackend.wait` resolves, mirroring the trainer's existing
"issue to all engines, then ``ray.get``" fan-out. For vLLM a ``Handle`` is a Ray
``ObjectRef``; for SGLang it is a pending-ZMQ-recv token. TensorRT-LLM will wrap
its executor futures the same way.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional

Handle = Any  # opaque per-backend future/token, resolved via ESBackend.wait


@dataclass(frozen=True)
class SamplingConfig:
    """The only five sampling fields ES ever sets. Defaults are greedy."""

    n: int = 1
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 512
    seed: Optional[int] = None


@dataclass
class Completion:
    """One sampled continuation for a prompt."""

    text: str
    token_ids: List[int] = field(default_factory=list)


@dataclass
class GenerationOutput:
    """Normalized per-prompt result. Mirrors the fields the trainer reads off a
    vLLM ``RequestOutput``: ``gen.prompt``, ``gen.outputs[i].text``,
    ``gen.outputs[i].token_ids``."""

    prompt: str
    outputs: List[Completion] = field(default_factory=list)


class ESBackend(ABC):
    """Interface every inference backend implements for ES."""

    # ------------------------------------------------------------------ lifecycle
    @property
    @abstractmethod
    def num_engines(self) -> int:
        ...

    @abstractmethod
    def start(self) -> None:
        """Launch engines and bring them to a ready state. After ``start`` every
        engine holds identical weights."""

    @abstractmethod
    def shutdown(self) -> None:
        """Terminate all engines/subprocesses and free their resources."""

    # -------------------------------------------------------------- async ops
    @abstractmethod
    def generate_async(
        self, engine_idx: int, prompts: List[str], sampling: SamplingConfig
    ) -> Handle:
        """Resolves to ``List[GenerationOutput]`` (one per prompt)."""

    @abstractmethod
    def perturb_async(
        self, engine_idx: int, seed: int, sigma: float, negate: bool = False
    ) -> Handle:
        """Apply ``+/- sigma * randn(seed)`` in place on ``engine_idx``."""

    @abstractmethod
    def restore_async(self, engine_idx: int, seed: int, sigma: float) -> Handle:
        """Undo a preceding :meth:`perturb_async` on ``engine_idx`` (returns the
        engine to its base weights)."""

    @abstractmethod
    def wait(self, handles: List[Handle]) -> list:
        """Block until every handle resolves; return results in order."""

    # --------------------------------------------------------- committed update
    @abstractmethod
    def sync_after_update(
        self, seeds, coeffs, alpha: float, population_size: int
    ) -> None:
        """Commit the ES gradient step and leave EVERY engine at the identical new
        weights. Both backends apply the update on engine 0, then NCCL-broadcast
        engine-0's weights to every engine."""

    # --------------------------------------------------------------- checkpoint
    @abstractmethod
    def save(self, path: str, engine_idx: int = 0) -> None:
        """Save the canonical model weights to ``path`` (raw ES checkpoint)."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load an ES checkpoint into every engine."""

    # ------------------------------------------------------------- drift check
    def checksums(self) -> List[Optional[tuple]]:
        """Per-engine weight signatures for cross-engine drift detection.
        Optional; default returns ``None`` per engine."""
        return [None] * self.num_engines
