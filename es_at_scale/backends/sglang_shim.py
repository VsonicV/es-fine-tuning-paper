"""In-process SGLang shim that gives the ES trainer per-parameter weight ops on a
live SGLang engine WITHOUT forking SGLang.

SGLang's offline ``Engine`` spawns the model-owning ``Scheduler`` as a fresh
(``spawn``) child process via ``mp.Process(target=run_scheduler_process)`` -- the
module-level function referenced in ``sglang.srt.entrypoints.engine`` -- and exposes
a generic ``collective_rpc(method, **kwargs)`` that runs ``getattr(scheduler,
method)(**kwargs)`` on every TP rank + ``barrier()``.

We replace ``engine.run_scheduler_process`` with a module-level wrapper (before
constructing the Engine) that monkeypatches ``es_*`` methods onto ``Scheduler``
*inside the spawned child*, then calls the real entrypoint. Those methods reach
``self.tp_worker.model_runner.model.named_parameters()`` and run the shared
:mod:`es_at_scale.backends.es_ops` math in place on the GPU. Only ints/floats/lists
cross the RPC boundary -- noise is regenerated on-device from seeds.

Weight logic mirrors the vLLM backend exactly: perturb in place, subtract-restore,
apply the committed ES update on engine 0, then NCCL-broadcast engine-0's weights
to every engine (one full-weight copy per iteration; no per-engine snapshot). The
inter-engine group uses SGLang's own ``StatelessProcessGroup`` + ``PyNcclCommunicator``
(a NCCL communicator independent of each engine's TP group), the same primitives
vLLM uses.

Verified against SGLang 0.5.6.post2. If those internals move, the worker's startup
self-test (``es_selftest``) fails loudly rather than silently misbehaving.
"""

from __future__ import annotations

import json

import torch

from es_at_scale.backends import es_ops


def _params_fn(scheduler):
    return lambda: scheduler.tp_worker.model_runner.model.named_parameters()


# ---- Scheduler methods (run inside the spawned child, on each TP rank) ----

def _es_perturb(self, seed, sigma, negate=False):
    es_ops.perturb(_params_fn(self), seed, sigma, negate)


def _es_restore(self, seed, sigma):
    # Undo perturb by subtracting the same seeded noise (exact inverse up to
    # bf16 rounding; the per-iteration engine-0 broadcast re-syncs any drift).
    es_ops.restore_subtract(_params_fn(self), seed, sigma)


def _es_apply_update(self, seeds, coeffs, alpha, population_size):
    # Applied on engine 0 only; the result is then broadcast to every engine.
    es_ops.apply_update(_params_fn(self), seeds, coeffs, alpha, population_size)


def _es_save(self, path):
    es_ops.save_to_disk(_params_fn(self), path)


def _es_load(self, path):
    pf = _params_fn(self)
    device = next(pf())[1].device
    es_ops.load_from_disk(pf, path, device)


def _nccl_so_path():
    # find_nccl_library() falls back to a bare "libnccl.so.2", which the dynamic
    # linker cannot resolve when the lib lives at a pip path and LD_LIBRARY_PATH is
    # unset. Point ctypes at the exact file instead.
    try:
        import os
        import nvidia.nccl  # namespace package: use __path__, not __file__ (which is None)

        base = list(nvidia.nccl.__path__)[0]
        p = os.path.join(base, "lib", "libnccl.so.2")
        return p if os.path.exists(p) else None
    except Exception:
        return None


def _es_init_broadcast_group(self, master_address, master_port, rank, world_size):
    # Inter-engine NCCL group: one rank per engine (TP=1). Independent of each
    # engine's own TP group -- same StatelessProcessGroup/PyNccl pattern vLLM uses.
    self._es_bcast = None
    if int(world_size) <= 1:
        return
    from sglang.srt.distributed.utils import StatelessProcessGroup
    from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator

    so = _nccl_so_path()
    device = next(self.tp_worker.model_runner.model.parameters()).device
    pg = StatelessProcessGroup.create(
        host=str(master_address), port=int(master_port),
        rank=int(rank), world_size=int(world_size),
    )
    comm = PyNcclCommunicator(pg, device=device, library_path=so)
    # `available` is False only if the NCCL library could not be loaded -- fail loudly
    # (a dead broadcast would desync engines and corrupt training). Note the comm
    # starts `disabled=True` by design; we enable it per-broadcast via change_state.
    assert getattr(comm, "available", False), (
        f"inter-engine NCCL communicator unavailable (nccl so={so!r})"
    )
    self._es_bcast = comm


def _es_broadcast_weights(self, src_rank):
    comm = getattr(self, "_es_bcast", None)
    if comm is None:
        return
    with comm.change_state(enable=True):
        for _, p in self.tp_worker.model_runner.model.named_parameters():
            comm.broadcast(p.data, src=int(src_rank), stream=torch.cuda.current_stream())
    torch.cuda.synchronize()


def _es_checksum(self, out_path):
    # collective_rpc returns only success/message, not values, so publish the
    # checksum via a small file the worker reads back. TP rank 0 only.
    if getattr(self, "tp_rank", 0) != 0:
        return
    sig = es_ops.checksum(_params_fn(self))
    with open(out_path, "w") as f:
        json.dump(list(sig), f)


def _es_selftest(self, out_path):
    # Assert the private weight path resolves and report the param count so the
    # worker can cross-check it against the checkpoint.
    if getattr(self, "tp_rank", 0) != 0:
        return
    names = [n for n, _ in _params_fn(self)()]
    assert len(names) > 0, "model.named_parameters() is empty"
    with open(out_path, "w") as f:
        json.dump({"param_count": len(names)}, f)


_ES_METHODS = {
    "es_perturb": _es_perturb,
    "es_restore": _es_restore,
    "es_apply_update": _es_apply_update,
    "es_save": _es_save,
    "es_load": _es_load,
    "es_init_broadcast_group": _es_init_broadcast_group,
    "es_broadcast_weights": _es_broadcast_weights,
    "es_checksum": _es_checksum,
    "es_selftest": _es_selftest,
}


def _patch_scheduler_class():
    from sglang.srt.managers.scheduler import Scheduler

    for name, fn in _ES_METHODS.items():
        setattr(Scheduler, name, fn)


def run_scheduler_with_es(*args, **kwargs):
    # Executed in the spawned child; patch BEFORE Scheduler(...) is constructed,
    # then hand off to the real entrypoint.
    _patch_scheduler_class()
    from sglang.srt.managers.scheduler import run_scheduler_process

    return run_scheduler_process(*args, **kwargs)


def make_es_engine(**engine_kwargs):
    """Construct an ``sglang.Engine`` whose scheduler child carries the ``es_*``
    methods. The engine's ``_launch_subprocesses`` spawns the scheduler with
    ``target=run_scheduler_process`` looked up in the engine module's namespace,
    so replacing that name with our wrapper injects the patch."""
    import sglang.srt.entrypoints.engine as _eng
    from sglang.srt.entrypoints.engine import Engine

    _eng.run_scheduler_process = run_scheduler_with_es
    return Engine(**engine_kwargs)
