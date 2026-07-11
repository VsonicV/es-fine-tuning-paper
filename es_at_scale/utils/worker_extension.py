import random

import numpy as np
import torch

from es_at_scale.backends import es_ops


def _stateless_init_process_group(
    master_address, master_port, rank, world_size, device
):
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup

    pg = StatelessProcessGroup.create(
        host=master_address,
        port=master_port,
        rank=rank,
        world_size=world_size
    )
    return PyNcclCommunicator(pg, device=device)


class WorkerExtension:
    """The class for vLLM's worker to inherit from.

    The ES weight math lives in the backend-agnostic ``es_at_scale.backends.es_ops``
    module; the methods below are thin vLLM-specific adapters that (a) supply the
    ``model_runner.model.named_parameters()`` accessor and (b) own the inter-engine
    PyNccl group used to broadcast engine-0's weights to every engine. Method
    signatures are kept identical to the pre-refactor version so the trainer's
    ``collective_rpc(...)`` call sites are unchanged.
    """

    def _params(self):
        return self.model_runner.model.named_parameters()

    def _set_seed(self, seed):
        # set a seed locally on the worker extension for reproducibility
        self.local_seed = seed

        # seeding
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def save_self_initial_weights(self):
        """Save a copy of itself in CPU memory."""
        self.initial_weights = es_ops.snapshot(self._params, device="cpu")
        print("Initial weights saved.")

    def restore_self_weights(self, seed, sigma):
        self._set_seed(seed)
        es_ops.restore_subtract(self._params, seed, sigma)
        return True

    def init_inter_engine_group(self, master_address: str, master_port: int,
                            engine_idx: int, num_engines: int):
            # TP rank within the engine (0..tp_size-1)
            tp_rank = getattr(self, "rank", None)
            tp_size = getattr(self, "world_size", None)

            if tp_rank is None or tp_size is None:
                tp_rank = getattr(self, "local_rank", 0)
                tp_size = getattr(getattr(self, "parallel_config", None), "tensor_parallel_size", None)
                if tp_size is None:
                    raise RuntimeError("Could not determine TP rank/size from vLLM worker attributes.")

            # IMPORTANT: create a group PER tp_rank across engines
            # So: rank is engine_idx, world_size is num_engines
            # Use a different port per tp_rank so each group has its own rendezvous
            port = int(master_port) + int(tp_rank)

            self.inter_pg = _stateless_init_process_group(
                master_address,
                port,
                rank=int(engine_idx),
                world_size=int(num_engines),
                device=self.device,
            )
            return True

    def broadcast_all_weights(self, src_rank: int):
        for _, p in self._params():
            self.inter_pg.broadcast(
                p, src=int(src_rank), stream=torch.cuda.current_stream()
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True

    def update_weights_from_seeds(self, seeds, coeffs, alpha, population_size):
        es_ops.apply_update(self._params, seeds, coeffs, alpha, population_size)
        return True

    def es_checksum(self):
        return es_ops.checksum(self._params)

    def perturb_self_weights(self, seed, noise_scale, negate=False):
        self._set_seed(seed)
        es_ops.perturb(self._params, seed, noise_scale, negate)
        print(f"Weights changed with: negate={negate}; scale={noise_scale}.")

    def save_self_weights_to_disk(self, filepath):
        """Save the current model weights to disk."""
        es_ops.save_to_disk(self._params, filepath)
        print(f"Model weights saved to {filepath}.")

    def load_weights_from_disk(self, filepath):
        es_ops.load_from_disk(self._params, filepath, self.device)
        return True
