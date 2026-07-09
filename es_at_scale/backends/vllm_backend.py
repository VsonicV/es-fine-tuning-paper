"""vLLM implementation of :class:`ESBackend`.

This is the original es-at-scale engine machinery moved behind the backend
interface with its mechanism unchanged: N Ray-actor ``vllm.LLM`` engines, ES
weight ops driven through ``collective_rpc`` into
``es_at_scale.utils.worker_extension.WorkerExtension``, and the committed update
applied on engine-0 then NCCL-broadcast to every engine.
"""

from __future__ import annotations

import os
from typing import List

import ray
import torch
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm import LLM, SamplingParams
from vllm.utils import get_ip, get_open_port

from es_at_scale.backends.base import (
    Completion,
    ESBackend,
    GenerationOutput,
    Handle,
    SamplingConfig,
)


class ESNcclLLM(LLM):
    def __init__(self, *args, **kwargs):
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        super().__init__(*args, **kwargs)


class _RayHandle:
    """Wraps a Ray ``ObjectRef`` plus an optional post-``ray.get`` converter so
    ``wait`` can uniformly resolve weight-op handles (result ignored) and
    generation handles (converted to ``List[GenerationOutput]``)."""

    __slots__ = ("ref", "convert")

    def __init__(self, ref, convert=None):
        self.ref = ref
        self.convert = convert


class VLLMBackend(ESBackend):
    def __init__(
        self,
        model_name: str,
        n_engines: int,
        n_gpu_per_engine: int = 1,
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.7,
        seed=None,
        use_gpus: str = None,
    ):
        self.model_name = model_name
        self._n_engines = int(n_engines)
        self.n_gpu_per_engine = int(n_gpu_per_engine)
        self.dtype = dtype
        self.gpu_memory_utilization = gpu_memory_utilization
        self.seed = seed
        self.use_gpus = use_gpus
        self.engines = []
        self.pgs = []

    @property
    def num_engines(self) -> int:
        return self._n_engines

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self.use_gpus is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.use_gpus
        os.environ.pop("RAY_ADDRESS", None)
        os.environ.pop("RAY_HEAD_IP", None)
        os.environ.pop("RAY_GCS_SERVER_ADDRESS", None)
        ray.init(address="local", include_dashboard=False, ignore_reinit_error=True)

        self.engines, self.pgs = self._launch_engines()

        master_address = get_ip()
        master_port = get_open_port()
        ray.get(
            [
                self.engines[i].collective_rpc.remote(
                    "init_inter_engine_group",
                    args=(master_address, master_port, i, self._n_engines),
                )
                for i in range(self._n_engines)
            ]
        )

    def _launch_engines(self):
        pgs = [
            placement_group(
                [{"GPU": 1, "CPU": 0}] * self.n_gpu_per_engine,
                strategy="PACK",
                lifetime="detached",
            )
            for _ in range(self._n_engines)
        ]
        ray.get([pg.ready() for pg in pgs])

        strategies = [
            PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=0,
            )
            for pg in pgs
        ]

        engines = [
            ray.remote(num_cpus=0, num_gpus=0, scheduling_strategy=strategy)(
                ESNcclLLM
            ).remote(
                model=self.model_name,
                tensor_parallel_size=self.n_gpu_per_engine,
                distributed_executor_backend="ray",
                worker_extension_cls="es_at_scale.utils.worker_extension.WorkerExtension",
                dtype=self.dtype,
                enable_prefix_caching=False,
                enforce_eager=False,
                gpu_memory_utilization=self.gpu_memory_utilization,
            )
            for strategy in strategies
        ]
        return engines, pgs

    def shutdown(self) -> None:
        for llm in self.engines:
            try:
                ray.kill(llm)
            except Exception:
                pass
        for pg in self.pgs:
            try:
                remove_placement_group(pg)
            except Exception:
                pass
        self.engines = []
        self.pgs = []

    # -------------------------------------------------------------- async ops
    def _sampling_params(self, sampling: SamplingConfig) -> SamplingParams:
        return SamplingParams(
            n=sampling.n,
            seed=sampling.seed,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            max_tokens=sampling.max_tokens,
        )

    @staticmethod
    def _to_generation_outputs(request_outputs) -> List[GenerationOutput]:
        out = []
        for ro in request_outputs:
            comps = [
                Completion(text=o.text, token_ids=list(o.token_ids))
                for o in ro.outputs
            ]
            out.append(GenerationOutput(prompt=ro.prompt, outputs=comps))
        return out

    def generate_async(
        self, engine_idx: int, prompts: List[str], sampling: SamplingConfig
    ) -> Handle:
        ref = self.engines[engine_idx].generate.remote(
            prompts, self._sampling_params(sampling), use_tqdm=False
        )
        return _RayHandle(ref, convert=self._to_generation_outputs)

    def perturb_async(
        self, engine_idx: int, seed: int, sigma: float, negate: bool = False
    ) -> Handle:
        ref = self.engines[engine_idx].collective_rpc.remote(
            "perturb_self_weights", args=(int(seed), sigma, negate)
        )
        return _RayHandle(ref)

    def restore_async(self, engine_idx: int, seed: int, sigma: float) -> Handle:
        ref = self.engines[engine_idx].collective_rpc.remote(
            "restore_self_weights", args=(int(seed), sigma)
        )
        return _RayHandle(ref)

    def wait(self, handles: List[Handle]) -> list:
        results = ray.get([h.ref for h in handles])
        return [
            h.convert(r) if h.convert is not None else r
            for h, r in zip(handles, results)
        ]

    # --------------------------------------------------------- committed update
    def sync_after_update(
        self, seeds, coeffs, alpha: float, population_size: int
    ) -> None:
        ray.get(
            self.engines[0].collective_rpc.remote(
                "update_weights_from_seeds",
                args=(seeds, coeffs, alpha, population_size),
            )
        )
        ray.get(
            [
                e.collective_rpc.remote("broadcast_all_weights", args=(0,))
                for e in self.engines
            ]
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # --------------------------------------------------------------- checkpoint
    def save(self, path: str, engine_idx: int = 0) -> None:
        ray.get(
            self.engines[engine_idx].collective_rpc.remote(
                "save_self_weights_to_disk", args=(path,)
            )
        )

    def load(self, path: str) -> None:
        ray.get(
            [
                e.collective_rpc.remote("load_weights_from_disk", args=(path,))
                for e in self.engines
            ]
        )

    # ------------------------------------------------------------- drift check
    def checksums(self):
        # collective_rpc returns one result per TP worker; TP=1 -> take [0].
        refs = [e.collective_rpc.remote("es_checksum") for e in self.engines]
        return [tuple(res[0]) for res in ray.get(refs)]
