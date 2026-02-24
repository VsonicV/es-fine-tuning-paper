"""
Utility methods used by ES trainer
"""

import gc
import time

import torch
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup


class WorkerExtension:
    """
    Methods used by the ES trainer:
    - perturb_self_weights(seed, sigma_or_scale, coeff=1.0, negate=False)
    - restore_self_weights(seed, SIGMA)
    - init_inter_engine_group(master_address, master_port, rank, world_size)
    - broadcast_all_weights(src_rank)
    - save_self_weights_to_disk(filepath)
    """

    def __init__(self):
        self.inter_pg = None

    def perturb_self_weights(self, seed: int, sigma: float, negate: bool = False) -> bool:
        """
        Pertubes model weights

        :param seed
            Random number generator seed
        :param sigma
            Scaling factor for perturbation noise
        :param negate
            Flag to flip the sing of the perturbation noise (In case we want to undo the perturbation)

        :return: True
        """
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
        """
        Restores pertubed model weights

        :param seed
            Random number generator seed
        :param sigma
            Scaling factor for perturbation noise

        :return: True
        """
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

    def init_inter_engine_group(self, main_address: str, main_port: int, rank: int, world_size: int) -> bool:
        """
        Performs the necessary setup to create a communication group among all participating actors

        :param master_address
            IP address of the main node

        :param master_port
            Port of the main node

        :param rank
            Rank of the non-main node

        :param world_size
            Total number of nodes including main node

        :return: True
        """
        pg = StatelessProcessGroup.create(host=main_address, port=main_port, rank=rank, world_size=world_size)
        self.inter_pg = PyNcclCommunicator(pg, device=self.device)
        return True

    def broadcast_all_weights(self, src_rank: int) -> bool:
        """
        Performs a synchronized broadcast of all model parameters across
        distributed workers in a GPU-based training or inference setup

        :param src_rank
            Rank of the node from which the parameters are broadcasted to other nodes

        :return: True
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
        """
        Save model weights to a file on disk

        :param filepath
            Path of the file to save the weights to

        :return: True
        """
        state_dict_to_save = {}

        for name, param in self.model_runner.model.named_parameters():
            state_dict_to_save[name] = param.detach().cpu()

        torch.save(state_dict_to_save, filepath)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        time.sleep(0.1)
        return True
