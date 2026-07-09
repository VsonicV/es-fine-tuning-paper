"""SGLang implementation of :class:`ESBackend` (trainer side, es-prefix-cache env).

Spawns N out-of-process ES workers with the SGLang env's python (one per engine,
each GPU-pinned), and drives them over ZeroMQ. Only seeds / sampling / prompts /
token-ids cross the wire; the seed-driven perturbation runs on the engine GPU.

Weight logic mirrors the vLLM backend: perturb in place, subtract-restore, apply
the committed ES update on engine 0, then NCCL-broadcast engine-0's weights to every
engine each iteration (via SGLang's StatelessProcessGroup/PyNccl inter-engine group).
Each engine holds only 1x the model weights -- no per-engine snapshot -- which is the
tradeoff that scales to large models.
"""

from __future__ import annotations

import os
import pickle
import socket
import subprocess
import uuid
from typing import List

import zmq

from es_at_scale.backends.base import (
    Completion,
    ESBackend,
    GenerationOutput,
    Handle,
    SamplingConfig,
)

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class _ZmqHandle:
    __slots__ = ("engine_idx", "req_id", "convert")

    def __init__(self, engine_idx, req_id, convert=None):
        self.engine_idx = engine_idx
        self.req_id = req_id
        self.convert = convert


class SGLangBackend(ESBackend):
    def __init__(
        self,
        model_name: str,
        n_engines: int,
        gpus: List[int],
        n_gpu_per_engine: int = 1,
        dtype: str = "bfloat16",
        mem_fraction_static: float = 0.7,
        seed=None,
        sglang_python: str = None,
        disable_cuda_graph: bool = False,
        op_timeout_s: float = 900.0,
        start_timeout_s: float = 1200.0,
    ):
        self.model_name = model_name
        self._n_engines = int(n_engines)
        self.gpus = list(gpus)
        self.n_gpu_per_engine = int(n_gpu_per_engine)
        self.dtype = dtype
        self.mem_fraction_static = mem_fraction_static
        self.seed = seed
        self.disable_cuda_graph = disable_cuda_graph
        self.op_timeout_ms = int(op_timeout_s * 1000)
        self.start_timeout_ms = int(start_timeout_s * 1000)

        self.sglang_python = sglang_python or os.environ.get("ES_SGLANG_PYTHON")
        if not self.sglang_python:
            raise ValueError(
                "SGLang backend needs the sglang-env python: pass sglang_python "
                "(--sglang-python) or set $ES_SGLANG_PYTHON."
            )
        need = self._n_engines * self.n_gpu_per_engine
        if len(self.gpus) < need:
            raise ValueError(
                f"Need {need} GPUs ({self._n_engines} engines x {self.n_gpu_per_engine}), "
                f"got {len(self.gpus)}: {self.gpus}"
            )

        self.ctx = None
        self.sockets = []
        self.procs = []
        self.endpoints = []
        self._id = 0

    @property
    def num_engines(self) -> int:
        return self._n_engines

    def _engine_gpus(self, i):
        s = i * self.n_gpu_per_engine
        return self.gpus[s:s + self.n_gpu_per_engine]

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self.ctx = zmq.Context.instance()
        run_id = uuid.uuid4().hex[:8]
        for i in range(self._n_engines):
            endpoint = f"ipc:///tmp/es-sglang-{os.getpid()}-{run_id}-{i}.sock"
            self.endpoints.append(endpoint)
            sock = self.ctx.socket(zmq.REQ)
            sock.setsockopt(zmq.LINGER, 0)
            sock.connect(endpoint)
            self.sockets.append(sock)

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in self._engine_gpus(i))
            env["PYTHONPATH"] = _REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
            # The worker runs the sglang-env python but inherits our (trainer-env)
            # PATH/LD_LIBRARY_PATH. Put the sglang env's bin first so its build tools
            # (ninja) and console scripts resolve; drop the trainer's LD_LIBRARY_PATH
            # so the sglang torch loads its own bundled CUDA libs; and make a CUDA
            # toolkit's nvcc available for any flashinfer/torch JIT fallback.
            sglang_bin = os.path.dirname(os.path.abspath(self.sglang_python))
            path_parts = [sglang_bin]
            if not env.get("CUDA_HOME"):
                for cand in ("/usr/local/cuda", "/usr/local/cuda-12.9"):
                    if os.path.isdir(cand):
                        env["CUDA_HOME"] = cand
                        break
            if env.get("CUDA_HOME"):
                path_parts.append(os.path.join(env["CUDA_HOME"], "bin"))
            env["PATH"] = os.pathsep.join(path_parts + [env.get("PATH", "")])
            env.pop("LD_LIBRARY_PATH", None)
            cmd = [
                self.sglang_python, "-m", "es_at_scale.backends.sglang_worker",
                "--model", self.model_name,
                "--endpoint", endpoint,
                "--tp-size", str(self.n_gpu_per_engine),
                "--mem-fraction-static", str(self.mem_fraction_static),
                "--dtype", self.dtype,
                "--random-seed", str((self.seed or 42) + i),
            ]
            if self.disable_cuda_graph:
                cmd.append("--disable-cuda-graph")
            self.procs.append(subprocess.Popen(cmd, env=env))

        # Block until every worker's engine is up (model load + graph capture).
        for i in range(self._n_engines):
            h = self._send(i, {"op": "ping"})
            self._recv_one(h, timeout_ms=self.start_timeout_ms)

        # Build the inter-engine NCCL broadcast group (engine 0 -> all). The init
        # is a collective rendezvous, so fire it on every engine before waiting.
        if self._n_engines > 1:
            master_address, master_port = "127.0.0.1", _free_port()
            self.wait([
                self._send(i, {"op": "init_broadcast_group",
                               "master_address": master_address, "master_port": master_port,
                               "rank": i, "world_size": self._n_engines})
                for i in range(self._n_engines)
            ])

    def shutdown(self) -> None:
        for i, sock in enumerate(self.sockets):
            try:
                sock.send(pickle.dumps({"op": "shutdown", "id": self._next_id()}))
                sock.RCVTIMEO = 10000
                sock.recv()
            except Exception:
                pass
        for p in self.procs:
            try:
                p.terminate()
                p.wait(timeout=15)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
        for sock in self.sockets:
            try:
                sock.close(0)
            except Exception:
                pass
        self.sockets, self.procs, self.endpoints = [], [], []

    # -------------------------------------------------------------- messaging
    def _next_id(self):
        self._id += 1
        return self._id

    def _send(self, engine_idx, msg, convert=None) -> _ZmqHandle:
        rid = self._next_id()
        msg = dict(msg)
        msg["id"] = rid
        self.sockets[engine_idx].send(pickle.dumps(msg))
        return _ZmqHandle(engine_idx, rid, convert)

    def _recv_one(self, handle: _ZmqHandle, timeout_ms=None):
        sock = self.sockets[handle.engine_idx]
        if not sock.poll(timeout_ms if timeout_ms is not None else self.op_timeout_ms):
            raise RuntimeError(
                f"SGLang engine {handle.engine_idx} timed out (is the worker alive?)"
            )
        rep = pickle.loads(sock.recv())
        if not rep.get("ok"):
            raise RuntimeError(
                f"SGLang engine {handle.engine_idx} error: {rep.get('error')}\n"
                f"{rep.get('traceback', '')}"
            )
        return handle.convert(rep) if handle.convert is not None else rep

    def wait(self, handles: List[Handle]) -> list:
        return [self._recv_one(h) for h in handles]

    # -------------------------------------------------------------- async ops
    @staticmethod
    def _to_generation_outputs(rep) -> List[GenerationOutput]:
        out = []
        for o in rep["outputs"]:
            comps = [Completion(text=c["text"], token_ids=c["token_ids"])
                     for c in o["completions"]]
            out.append(GenerationOutput(prompt=o["prompt"], outputs=comps))
        return out

    def generate_async(self, engine_idx, prompts, sampling: SamplingConfig) -> Handle:
        return self._send(
            engine_idx,
            {"op": "generate", "prompts": list(prompts),
             "sampling": {"n": sampling.n, "temperature": sampling.temperature,
                          "top_p": sampling.top_p, "max_tokens": sampling.max_tokens}},
            convert=self._to_generation_outputs,
        )

    def perturb_async(self, engine_idx, seed, sigma, negate=False) -> Handle:
        return self._send(engine_idx,
                          {"op": "perturb", "seed": int(seed), "sigma": float(sigma),
                           "negate": bool(negate)})

    def restore_async(self, engine_idx, seed, sigma) -> Handle:
        return self._send(engine_idx, {"op": "restore", "seed": int(seed), "sigma": float(sigma)})

    # --------------------------------------------------------- committed update
    def sync_after_update(self, seeds, coeffs, alpha, population_size) -> None:
        # Mirror the vLLM backend: apply the update on engine 0, then NCCL-broadcast
        # its weights to every engine (one full-weight copy; no per-engine snapshot).
        self.wait([
            self._send(0, {"op": "apply_update",
                           "seeds": [int(s) for s in seeds],
                           "coeffs": [float(c) for c in coeffs],
                           "alpha": float(alpha),
                           "population_size": int(population_size)})
        ])
        if self._n_engines > 1:
            # Broadcast is collective -- fire on all engines before waiting.
            self.wait([
                self._send(i, {"op": "broadcast_weights", "src": 0})
                for i in range(self._n_engines)
            ])

    # --------------------------------------------------------------- checkpoint
    def save(self, path: str, engine_idx: int = 0) -> None:
        self.wait([self._send(engine_idx, {"op": "save", "path": path})])

    def load(self, path: str) -> None:
        self.wait([self._send(i, {"op": "load", "path": path})
                   for i in range(self._n_engines)])

    # ------------------------------------------------------------- drift check
    def checksums(self):
        reps = self.wait([self._send(i, {"op": "checksum"})
                          for i in range(self._n_engines)])
        return [tuple(r["checksum"]) for r in reps]
