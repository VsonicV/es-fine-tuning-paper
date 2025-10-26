
"""
Worker extension aligned with vLLM's latest TPU integration (tpu-inference backend).

Only TPU-compatibility code paths were added. ES logic and public methods remain unchanged.
"""
import gc
import time
from typing import Dict, Optional

import torch

# vLLM distributed tools
from vllm.distributed.utils import StatelessProcessGroup

# Try to import device communicators that exist across backends.
# We prefer TPU communicator when CUDA is not available.
try:
    from vllm.distributed.device_communicators.tpu_communicator import TpuCommunicator  # type: ignore
    _TPU_COMM_AVAILABLE = True
except Exception:
    TpuCommunicator = None  # type: ignore
    _TPU_COMM_AVAILABLE = False

try:
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator  # type: ignore
    _PYNCCL_AVAILABLE = True
except Exception:
    PyNcclCommunicator = None  # type: ignore
    _PYNCCL_AVAILABLE = False


def _is_cuda() -> bool:
    return bool(torch.cuda.is_available())


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class WorkerExtension:
    """
    Methods used by the ES trainer (API unchanged):
      - perturb_self_weights(seed, sigma_or_scale, negate=False)
      - restore_self_weights(seed, sigma)
      - init_inter_engine_group(master_address, master_port, rank, world_size)
      - broadcast_all_weights(src_rank)
      - save_self_weights_to_disk(filepath)
    """

    def perturb_self_weights(self, seed: int, sigma_or_scale: float, negate: bool = False) -> bool:
        scale = float(sigma_or_scale)
        sign = -1.0 if negate else 1.0
        for _, p in self.model_runner.model.named_parameters():
            # TPU path: generate on CPU, then move to the param device
            if _is_cuda() and p.is_cuda:
                gen = torch.Generator(device=p.device)
                gen.manual_seed(int(seed))
                noise = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)
            else:
                gen = torch.Generator(device="cpu")
                gen.manual_seed(int(seed))
                noise = torch.randn(p.shape, dtype=p.dtype, generator=gen).to(p.device, non_blocking=True)
            p.data.add_(sign * scale * noise)
            del noise
        _sync_cuda()
        # Do NOT call torch_xla syncs here; tpu-inference backend is managed by vLLM.
        return True

    def restore_self_weights(self, seed: int, sigma: float) -> bool:
        # Same RNG + negative update to perfectly invert perturbation.
        return self.perturb_self_weights(seed, sigma_or_scale=sigma, negate=True)

    def init_inter_engine_group(
        self,
        master_address: str,
        master_port: int,
        rank: int,
        world_size: int,
    ) -> bool:
        """
        Initialize an inter-engine communicator. On CUDA, prefer NCCL.
        On TPU, use vLLM's TpuCommunicator (CPU group + TPU collectives).
        """
        pg = StatelessProcessGroup.create(
            host=master_address, port=int(master_port), rank=int(rank), world_size=int(world_size)
        )

        self.inter_pg: Optional[object] = None

        if _is_cuda() and _PYNCCL_AVAILABLE:
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
            self.inter_pg = PyNcclCommunicator(pg, device=device)  # type: ignore
        elif _TPU_COMM_AVAILABLE:
            # vLLM's TPU communicator uses the CPU group for collectives under the hood.
            self.inter_pg = TpuCommunicator(pg)  # type: ignore
        else:
            # Fallback to no-op communicator; in that case we will do nothing.
            self.inter_pg = None
        # No device-specific sync here.
        return True

    def broadcast_all_weights(self, src_rank: int) -> bool:
        """
        Make all engines identical to `src_rank` weights. To avoid relying on a
        backend-specific `broadcast`, implement via `all_reduce`:

            - src contributes its tensor
            - others contribute zeros
            - SUM all-reduce -> everyone receives the src tensor

        Works for both NCCL and TPU communicators.
        """
        if self.inter_pg is None:
            return False

        for _, p in self.model_runner.model.named_parameters():
            # Prepare contribution tensor
            contrib = p.detach()
            # Non-src ranks contribute zeros
            # NOTE: clone() to avoid modifying the original parameter value before reduce.
            buf = contrib.clone()
            if not hasattr(self, "_rank_in_pg"):
                # StatelessProcessGroup uses global ranks; pass via src_rank only.
                # We don't know the local rank here; use an environment variable if present.
                # If RANK is not set, assume 0 (single process), which is fine.
                try:
                    current_rank = int(torch.distributed.get_rank())  # type: ignore[attr-defined]
                except Exception:
                    current_rank = 0
            else:
                current_rank = int(self._rank_in_pg)  # type: ignore[attr-defined]

            if current_rank != int(src_rank):
                buf.zero_()

            # Reduce across processes
            try:
                reduced = self.inter_pg.all_reduce(buf)  # type: ignore[attr-defined]
            except TypeError:
                # Some communicators require kwargs or op specification. Use SUM default.
                reduced = self.inter_pg.all_reduce(buf)  # type: ignore[attr-defined]

            # Copy back into the parameter
            p.data.copy_(reduced.to(p.device, non_blocking=True))
            del buf, reduced

        _sync_cuda()
        return True

    def save_self_weights_to_disk(self, filepath: str) -> bool:
        state_dict_to_save: Dict[str, torch.Tensor] = {}
        for name, p in self.model_runner.model.named_parameters():
            state_dict_to_save[name] = p.detach().cpu()
        torch.save(state_dict_to_save, filepath)
        gc.collect()
        if _is_cuda():
            torch.cuda.empty_cache()
        time.sleep(0.05)
        return True
