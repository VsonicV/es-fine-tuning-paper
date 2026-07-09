"""Backend-agnostic Evolution Strategies weight operations.

This module is the single source of truth for the ES weight math. It is pure
PyTorch -- it imports only ``torch`` (+ stdlib) and MUST NOT import vllm, sglang,
ray, or any backend package -- so it can be installed into every backend's conda
environment (the vLLM trainer env and the SGLang worker env alike) and called
verbatim from each backend's worker.

The perturb / restore-by-subtraction / update math is lifted unchanged from the
original ``es_at_scale/utils/worker_extension.py`` so that training dynamics stay
byte-for-byte identical across backends. :func:`snapshot` / :func:`restore_from_snapshot`
are generic base-weight helpers, and :func:`checksum` is a cheap signature used to
verify two engines hold identical weights.

A ``params_fn`` is a zero-argument callable returning a *fresh* iterable of
``(name, torch.Tensor)`` pairs, e.g. ``lambda: model.named_parameters()``. Every
function re-invokes it, so a one-shot generator is never exhausted across calls.
"""

from __future__ import annotations

import gc
import time
from typing import Callable, Dict, Iterable, Tuple

import torch

ParamsFn = Callable[[], Iterable[Tuple[str, torch.Tensor]]]


def _sync_and_clear() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    torch.cuda.empty_cache()


def _seed_noise(param: torch.Tensor, seed: int) -> torch.Tensor:
    """Gaussian noise shaped like ``param``, regenerated deterministically from
    ``seed`` on the parameter's own device/dtype. Identical across processes and
    engines given the same seed + GPU SKU."""
    gen = torch.Generator(device=param.device)
    gen.manual_seed(int(seed))
    return torch.randn(param.shape, dtype=param.dtype, device=param.device, generator=gen)


def perturb(params_fn: ParamsFn, seed: int, noise_scale: float, negate: bool = False) -> None:
    """In place: ``theta += sign * noise_scale * randn(seed)`` over every param
    (``sign = -1`` when ``negate``). One population member's exploration step."""
    sign = -1.0 if negate else 1.0
    scale = sign * float(noise_scale)
    for _, p in params_fn():
        noise = _seed_noise(p, seed)
        p.data.add_(scale * noise)
        del noise
    _sync_and_clear()


def restore_subtract(params_fn: ParamsFn, seed: int, sigma: float) -> None:
    """Exact inverse of ``perturb(seed, sigma, negate=False)`` -- ``theta -=
    sigma * randn(seed)``. Used by the vLLM backend (whose engine-0 rebroadcast
    each step masks the bf16 rounding drift)."""
    perturb(params_fn, seed, sigma, negate=True)


def apply_update(
    params_fn: ParamsFn,
    seeds,
    coeffs,
    alpha: float,
    population_size: int,
) -> None:
    """Commit the ES gradient step:
    ``theta += (alpha / population_size) * sum_i coeffs[i] * randn(seeds[i])``.

    The per-seed noise is accumulated in float32 before the single ``alpha/N``
    scaling, then cast back to the parameter dtype -- this preserves the tiny
    (~1e-5) update signal that would otherwise be truncated in bf16/fp16.
    ``coeffs[i]`` is the shaped (z-scored) reward of member ``i``.
    """
    seeds = [int(s) for s in seeds]
    coeffs = [float(c) for c in coeffs]
    scale = float(alpha) / float(population_size)
    for _, p in params_fn():
        acc = torch.zeros_like(p.data, dtype=torch.float32)
        for i, seed in enumerate(seeds):
            noise = _seed_noise(p, seed)
            acc.add_(noise.to(torch.float32) * coeffs[i])
            del noise
        acc.mul_(scale)
        p.data.add_(acc.to(p.dtype))
        del acc
    _sync_and_clear()


def snapshot(params_fn: ParamsFn, device=None) -> Dict[str, torch.Tensor]:
    """Detached clone of every parameter, forming a base ``theta`` snapshot.
    ``device=None`` keeps each tensor on its parameter's device (GPU -- fast
    restore); pass ``"cpu"`` to trade restore speed for VRAM on large models."""
    base: Dict[str, torch.Tensor] = {}
    for name, p in params_fn():
        t = p.detach().clone()
        base[name] = t.to(device) if device is not None else t
    return base


def restore_from_snapshot(params_fn: ParamsFn, base: Dict[str, torch.Tensor]) -> None:
    """Bit-exact restore of ``theta`` from a :func:`snapshot` (no rounding drift,
    unlike :func:`restore_subtract`)."""
    for name, p in params_fn():
        p.data.copy_(base[name].to(device=p.device, dtype=p.dtype))
    _sync_and_clear()


def save_to_disk(params_fn: ParamsFn, filepath: str) -> None:
    """Save weights as a raw ``{name: cpu_tensor}`` ``torch.save`` checkpoint
    (matches the original ES checkpoint format)."""
    state_dict = {name: p.detach().cpu() for name, p in params_fn()}
    torch.save(state_dict, filepath)


def load_from_disk(params_fn: ParamsFn, filepath: str, device) -> None:
    """Load an ES ``{name: tensor}`` checkpoint into the live params in place."""
    state_dict = torch.load(filepath, map_location=device)
    for name, p in params_fn():
        p.data.copy_(state_dict[name].to(device))
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    time.sleep(0.1)


def checksum(params_fn: ParamsFn) -> Tuple[int, float, float]:
    """Cheap order-stable signature ``(num_params, sum, sum_sq)`` in float64.
    Two engines whose weights are bit-identical produce an identical tuple; used
    to verify engines match (e.g. after a weight broadcast)."""
    n = 0
    total = 0.0
    total_sq = 0.0
    for _, p in params_fn():
        f = p.detach().double()
        n += 1
        total += f.sum().item()
        total_sq += (f * f).sum().item()
    return (n, total, total_sq)
