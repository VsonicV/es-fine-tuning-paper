import gc
import time
import random
import numpy as np
import torch
import os

def _stateless_init_process_group(master_address, master_port, rank, world_size, device):
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup
    pg = StatelessProcessGroup.create(
        host=master_address, port=master_port, rank=rank, world_size=world_size
    )
    return PyNcclCommunicator(pg, device=device)

class WorkerExtension:
    """
    Methods used by the ES trainer:
    - perturb_self_weights(seed, sigma_or_scale, coeff=1.0, negate=False)
    - restore_self_weights(seed, SIGMA)
    - update_weights_from_seeds(seeds, coeffs)  <-- NEW METHOD
    - init_inter_engine_group(master_address, master_port, rank, world_size)
    - broadcast_all_weights(src_rank)
    - save_self_weights_to_disk(filepath)
    """
    def _set_seed(self, seed):
        # set a seed locally on the worker extension for reproducibility
        self.local_seed = seed

        # seeding
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def perturb_self_weights(self, seed, noise_scale, negate=False):
        self._set_seed(seed)
        scale = float(noise_scale)
        sign = -1.0 if negate else 1.0
        for _, p in self.model_runner.model.named_parameters():
            gen = torch.Generator(device=p.device)
            gen.manual_seed(int(seed))
            noise = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)
            p.data.add_(sign * scale * noise)
            del noise
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return True

    def restore_self_weights(self, seed, SIGMA):
        self._set_seed(seed)
        for _, p in self.model_runner.model.named_parameters():
            gen = torch.Generator(device=p.device)
            gen.manual_seed(int(seed))
            noise = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)
            p.data.add_(-float(SIGMA) * noise)
            del noise
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return True

    def update_weights_from_seeds(self, seeds, coeffs, alpha, population_size):
        """
        Mimics the Original implementation's update loop structure:
        Iterate Param -> Iterate Seeds -> Accumulate -> Single Update.
        """
        # seeds and coeffs should be lists of equal length
        # coeffs[i] should be: (alpha / population_size) * normalized_reward
        
        for _, p in self.model_runner.model.named_parameters():
            # float32
            update_accumulator = torch.zeros_like(p.data, dtype=torch.float32)
            
            for i, seed in enumerate(seeds):
                self._set_seed(seed)
                gen = torch.Generator(device=p.device)
                gen.manual_seed(int(seed))
                
                # Generate noise (in native precision, usually float16/bfloat16)
                noise = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)
                
                # Better update stability
                term = noise.to(torch.float32) * coeffs[i]
                
                # Accumulate in FP32
                update_accumulator.add_(term)
            
            # div by population_size multiply by alpha (scalar)
            update_accumulator.div_(population_size)
            update_accumulator.mul_(alpha)
            # Apply final update to weight (cast back to model dtype at the very end)
            p.data.add_(update_accumulator.to(p.dtype))
            
            del update_accumulator
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return True

    def init_inter_engine_group(self, master_address: str, master_port: int, rank: int, world_size: int):
        self.inter_pg = _stateless_init_process_group(
            master_address, master_port, rank, world_size, self.device
        )
        return True

    def broadcast_all_weights(self, src_rank: int):
        for _, p in self.model_runner.model.named_parameters():
            self.inter_pg.broadcast(p, src=int(src_rank), stream=torch.cuda.current_stream())
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True

    def save_self_weights_to_disk(self, filepath):
        state_dict_to_save = {}
        for name, p in self.model_runner.model.named_parameters():
            state_dict_to_save[name] = p.detach().cpu()
        torch.save(state_dict_to_save, filepath)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(0.1)
        return True
    
    def dump_noise_for_seed(self, seed: int, out_dir: str):
        """
        Generate per-parameter noise using the same method as perturb/restore
        and save them to disk for determinism comparison.
        """
        os.makedirs(out_dir, exist_ok=True)
        noise_state = {}
        for name, p in self.model_runner.model.named_parameters():
            gen = torch.Generator(device=p.device)
            gen.manual_seed(int(seed))
            noise = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)
            noise_state[name] = noise.detach().cpu()
            del noise
        torch.save(noise_state, os.path.join(out_dir, f"noise_seed_{int(seed)}.pt"))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        return True
    
    # debug
    def print_model_weights_stats(self):
        for name, p in self.model_runner.model.named_parameters():
            print(f"Param: {name}, Shape: {p.shape}")
        return True