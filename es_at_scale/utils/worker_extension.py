import gc
import time
import torch
import random
import numpy as np


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
    """
    The class for vLLM's worker to inherit from.
    """

    def _set_seed(self, seed):
        # set a seed locally on the worker extension for reproducibility
        self.local_seed = seed

        # seeding
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def save_self_initial_weights(
        self,
    ):
        """Save a copy of itself in the CPU memory."""
        self.initial_weights = {}
        for name, p in self.model_runner.model.named_parameters():
            self.initial_weights[name] = p.detach().clone().cpu()
        print("Initial weights saved.")


    def restore_self_weights(self, seed, sigma):
        self._set_seed(seed)
        for _, p in self.model_runner.model.named_parameters():
            gen = torch.Generator(device=p.device)
            gen.manual_seed(int(seed))
            noise = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)
            p.data.add_(-float(sigma) * noise)
            del noise
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
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
        for _, p in self.model_runner.model.named_parameters():
            self.inter_pg.broadcast(
                p, src=int(src_rank), stream=torch.cuda.current_stream()
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
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

            for i, (seed) in enumerate(seeds):
                gen = torch.Generator(device=p.device)
                gen.manual_seed(int(seed))

                # Generate noise (in native precision, usually float16/bfloat16)
                noise = torch.randn(
                    p.shape, dtype=p.dtype, device=p.device, generator=gen
                )

                # FIXED: Convert noise to float32 BEFORE multiplication.
                # Previous code: noise.to(torch.float16) * coeffs[i]
                # This caused the tiny update signal (1e-5) to be truncated by FP16 precision limits.
                term = noise.to(torch.float32) * coeffs[i]

                # Accumulate in FP32
                update_accumulator.add_(term)

            # multiply with scale (alpha/population_size) once at the end to preserve precision
            scale = float(alpha) / float(population_size)
            update_accumulator.mul_(scale)
            # Apply final update to weight (cast back to model dtype at the very end)
            p.data.add_(update_accumulator.to(p.dtype))

            del update_accumulator

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return True

    def perturb_self_weights(self, seed, noise_scale, negate=False):
        """
        Add noise(seed) scaled by sigma_or_scale * coeff (or subtract when negate=True).
        - For exploration:  perturb_self_weights(seed, SIGMA, 1.0, False)
          and restore with restore_self_weights(seed, SIGMA) as before.
        - For ES update:   perturb_self_weights(seed, 1.0, coeff, False)
          where coeff = ALPHA/POPULATION_SIZE * norm_reward.
        """
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
        print(f"Weights changed with: sign={sign}; scale={sign * scale}.")

    def save_self_weights_to_disk(self, filepath):
        """Save the current model weights to disk."""
        state_dict_to_save = {}
        for name, p in self.model_runner.model.named_parameters():
            state_dict_to_save[name] = p.detach().cpu()
        torch.save(state_dict_to_save, filepath)
        print(f"Model weights saved to {filepath}.")

    def load_weights_from_disk(self, filepath):
        state_dict = torch.load(filepath, map_location=self.device)
        for name, p in self.model_runner.model.named_parameters():
            p.data.copy_(state_dict[name].to(self.device))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(0.1)
        return True
