import gc
import time

import torch


def _stateless_init_process_group(
    master_address: str, master_port: int, rank: int, world_size: int, device: torch.device
):
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup

    pg = StatelessProcessGroup.create(host=master_address, port=master_port, rank=rank, world_size=world_size)
    return PyNcclCommunicator(pg, device=device)


class WorkerExtension:
    """
    Methods used by the ES trainer:
    - perturb_self_weights(seed, sigma_or_scale, coeff=1.0, negate=False)
    - restore_self_weights(seed, SIGMA)
    - init_inter_engine_group(master_address, master_port, rank, world_size)
    - broadcast_all_weights(src_rank)
    - save_self_weights_to_disk(filepath)
    """

    def perturb_self_weights(self, seed: int, sigma: float, negate: bool = False) -> bool:
        scale = float(sigma)
        sign = -1.0 if negate else 1.0

        for _, param in self.model_runner.model.named_parameters():
            gen = torch.Generator(device=param.device)
            gen.manual_seed(int(seed))
            noise = torch.randn(param.shape, dtype=param.dtype, device=param.device, generator=gen)
            param.data.add_(sign * scale * noise)
            del noise

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        torch.cuda.empty_cache()
        return True

    def restore_self_weights(self, seed: int, sigma: float) -> bool:
        scale = float(sigma)

        for _, param in self.model_runner.model.named_parameters():
            gen = torch.Generator(device=param.device)
            gen.manual_seed(int(seed))
            noise = torch.randn(param.shape, dtype=param.dtype, device=param.device, generator=gen)
            param.data.add_(-scale * noise)
            del noise

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        torch.cuda.empty_cache()
        return True

    def init_inter_engine_group(self, master_address: str, master_port: int, rank: int, world_size: int) -> bool:
        self.inter_pg = _stateless_init_process_group(master_address, master_port, rank, world_size, self.device)
        return True

    def broadcast_all_weights(self, src_rank: int) -> bool:
        """
        Performs a synchronized broadcast of all model parameters across
        distributed workers in a GPU-based training or inference setup
        """
        for _, param in self.model_runner.model.named_parameters():
            # This call synchronizes all model parameters across GPUs or distributed workers by broadcasting
            # each parameter tensor from the source rank (src_rank) to every other device, using the current
            # CUDA stream for efficienct and asynchronous communication
            self.inter_pg.broadcast(param, src=int(src_rank), stream=torch.cuda.current_stream())

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        return True

    def save_self_weights_to_disk(self, filepath: str) -> bool:
        state_dict_to_save = {}

        for name, param in self.model_runner.model.named_parameters():
            state_dict_to_save[name] = param.detach().cpu()

        torch.save(state_dict_to_save, filepath)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        time.sleep(0.1)
        return True
