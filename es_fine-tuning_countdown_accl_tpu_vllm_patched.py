#!/usr/bin/env python3
import argparse
from datetime import datetime
import gc
import json
import os
import random
import shutil
import signal
import sys
import time
import importlib

import numpy as np
import ray
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# JAX Pallas compatibility shim
# ---------------------------------------------------------------------------
# vLLM 0.11.0's TPU Pallas attention path expects:
#   jax.experimental.pallas.tpu.TPUMemorySpace
# Newer JAX has:
#   jax.experimental.pallas.tpu.MemorySpace
# We alias the new name to the old before importing vLLM modules that bind to it.
def _apply_jax_pallas_memoryspace_compat():
    try:
        pltpu = importlib.import_module("jax.experimental.pallas.tpu")
    except Exception:
        # If pallas.tpu isn't present, do nothing; vLLM will choose a different path.
        return
    try:
        ms = getattr(pltpu, "MemorySpace", None)
        # Provide both legacy aliases if missing
        for alias in ("TPUMemorySpace", "TpuMemorySpace"):
            if ms is not None and not hasattr(pltpu, alias):
                setattr(pltpu, alias, ms)
    except Exception:
        # Best-effort shim; if anything else changes upstream, we defer to vLLM's own checks.
        pass

_apply_jax_pallas_memoryspace_compat()

# Import vLLM AFTER applying the shim
from vllm import LLM, SamplingParams
from vllm.utils import get_ip, get_open_port

from countdown.countdown_task import reward_function

# Default Hyperparameters
SIGMA = 0.001
ALPHA = 0.0005
POPULATION_SIZE = 30
NUM_ENGINES = 4
NUM_ITERATIONS = 1000
EXPERIMENT_DIR = "es-ft-experiment"


def parse_args():
    parser = argparse.ArgumentParser(
        description="ES Fine-tuning for Countdown Task with multi-engine sync (GPU/TPU)"
    )
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--sigma", type=float, default=SIGMA)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--population_size", type=int, default=POPULATION_SIZE)
    parser.add_argument("--num_engines", type=int, default=NUM_ENGINES)
    parser.add_argument("--num_iterations", type=int, default=NUM_ITERATIONS)
    parser.add_argument("--experiment_dir", type=str, default=EXPERIMENT_DIR)
    parser.add_argument("--cuda_devices", type=str, default="0,1,2,3")
    parser.add_argument("--global_seed", type=int, help="Global random seed")
    parser.add_argument("--tpu_chips", type=str, default=None,
                        help="Comma-separated TPU chip ids to use, e.g., 0,1,2,3 for v6e-4.")
    args = parser.parse_args()

    # Optional CUDA scoping; TPU actors ignore this (we clear it inside actors).
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    # set global random seed
    if args.global_seed is not None:
        random.seed(args.global_seed)
        np.random.seed(args.global_seed)
        torch.manual_seed(args.global_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.global_seed)

    return args


# ------------------- vLLM engine subclasses (GPU + TPU) --------------------
class ESNcclLLM(LLM):
    """GPU path: keep user's NCCL behavior unchanged."""
    def __init__(self, *args, **kwargs):
        # Let Ray/PG determine the actual visible device in the actor
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        super().__init__(*args, **kwargs)

    def es_call(self, method_name, *args):
        """Unified interface for ES operations (GPU version uses collective_rpc)."""
        return self.collective_rpc(method_name, args=args)


class ESTpuLLM(LLM):
    """TPU path: ES ops implemented in-class to avoid MRO conflicts with TPU workers.
       Adds small helpers to dump/load state for driver-mediated broadcast.
    """
    def __init__(self, *args, **kwargs):
        # Ensure TPU backend selection happens via Engine args and env.
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        # vLLM V1 uses multiprocessing by default; keep this off to avoid extra forks in our Ray actors.
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        # Optional hint; harmless if ignored by your vLLM build.
        os.environ.setdefault("VLLM_DEVICE", "tpu")
        super().__init__(*args, **kwargs)
        # Store inter-engine info
        self.rank = None
        self.world_size = None

    # -------- internal helpers --------
    def _get_model_params(self):
        """Access model parameters from the vLLM engine (V1 then V0)."""
        if hasattr(self.llm_engine, 'engine_core'):
            # V1 engine path
            model_executor = self.llm_engine.engine_core.model_executor
            if hasattr(model_executor, 'driver_worker'):
                return model_executor.driver_worker.model_runner.model.named_parameters()
        elif hasattr(self.llm_engine, 'model_executor'):
            # V0 engine path
            model_executor = self.llm_engine.model_executor
            if hasattr(model_executor, 'driver_worker'):
                return model_executor.driver_worker.model_runner.model.named_parameters()
        raise RuntimeError("Could not access model parameters from vLLM engine")

    # -------- ES operations --------
    def perturb_self_weights(self, seed: int, sigma_or_scale: float, negate: bool = False):
        """Add scaled Gaussian noise to model parameters, deterministically by seed."""
        scale = float(sigma_or_scale)
        sign = -1.0 if negate else 1.0
        for _, p in self._get_model_params():
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(seed))
            noise = torch.randn(p.shape, dtype=p.dtype, generator=gen).to(p.device, non_blocking=True)
            p.data.add_(sign * scale * noise)
            del noise
        return True

    def restore_self_weights(self, seed: int, sigma: float):
        """Restore weights by subtracting the same noise (deterministic by seed)."""
        return self.perturb_self_weights(seed, sigma_or_scale=sigma, negate=True)

    def dump_state_dict(self):
        """Return a CPU state_dict for driver-mediated broadcast."""
        sd = {}
        for name, p in self._get_model_params():
            sd[name] = p.detach().to("cpu")
        return sd

    def load_state_dict(self, state_dict):
        """Load a CPU state_dict (moved to device with non_blocking copy)."""
        # Assume keys match exactly (we saved from an identical model).
        for name, p in self._get_model_params():
            src = state_dict[name]
            p.data.copy_(src.to(p.device, non_blocking=True))
        return True

    def init_inter_engine_group(self, master_address: str, master_port: int, rank: int, world_size: int):
        """Record rank/world_size for logging; TPU collectives are driver-mediated in this script."""
        self.rank = int(rank)
        self.world_size = int(world_size)
        print(f"[TPU] inter-engine group init: rank={self.rank}, world_size={self.world_size}")
        return True

    # Kept for compatibility with existing call sites; does nothing here.
    def broadcast_all_weights(self, src_rank: int):
        print(f"[TPU] broadcast_all_weights(src={src_rank}) is driver-mediated in this script.")
        return True

    def save_self_weights_to_disk(self, filepath: str):
        """Save model weights to disk (always via CPU for portability)."""
        state_dict = {}
        for name, p in self._get_model_params():
            state_dict[name] = p.detach().cpu()
        torch.save(state_dict, filepath)
        gc.collect()
        print(f"Saved weights to {filepath}")
        return True

    def es_call(self, method_name, *args):
        """Unified interface for ES operations (TPU version calls methods directly)."""
        method = getattr(self, method_name)
        return method(*args)


# ------------------------- GPU engine launch (unchanged) --------------------
def launch_engines(num_engines, model_name):
    # Strict 1-GPU isolation via PGs
    pgs = [placement_group([{"GPU": 1, "CPU": 0}], lifetime="detached") for _ in range(num_engines)]
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
        ray.remote(num_cpus=0, num_gpus=0, scheduling_strategy=strategy)(ESNcclLLM).remote(
            model=model_name,
            tensor_parallel_size=1,
            distributed_executor_backend="ray",
            worker_extension_cls="utils.worker_extn.WorkerExtension",
            dtype="float16",
            enable_prefix_caching=False,
            enforce_eager=False,
        )
        for strategy in strategies
    ]
    return engines, pgs


# ------------------------- TPU engine launch (new) --------------------------
def launch_engines_tpu(num_engines, model_name, tpu_chips_csv: str | None = None):
    """
    Launch one vLLM engine per TPU chip using vLLM's TPU backend.
    Isolate chips per actor via TPU_VISIBLE_CHIPS and set BF16.
    """
    # Parse chip list; default to 0..(num_engines-1) if not specified.
    if not tpu_chips_csv or tpu_chips_csv.strip() == "":
        chips = [str(i) for i in range(max(1, num_engines))]
    else:
        chips = [c.strip() for c in tpu_chips_csv.split(",") if c.strip() != ""]
    if len(chips) < num_engines:
        raise ValueError(f"Need at least {num_engines} TPU chips, got {len(chips)} from --tpu_chips={tpu_chips_csv!r}")

    engines = []
    for i in range(num_engines):
        chip_id = chips[i]
        runtime_env = {"env_vars": {
            # Limit process-visible chip(s) for this actor.
            "TPU_VISIBLE_CHIPS": chip_id,
        }}
        # We do not request GPU resources on TPU.
        actor = ray.remote(num_cpus=0, scheduling_strategy="DEFAULT", runtime_env=runtime_env)(ESTpuLLM).remote(
            model=model_name,
            tensor_parallel_size=1,  # one chip per actor
            distributed_executor_backend="ray",
            dtype="bfloat16",
            enable_prefix_caching=False,
            enforce_eager=False,
        )
        engines.append(actor)
    return engines


# ------------------------- Evaluation + ES loop -----------------------------
def evaluate_countdown_handle(llm, task_datas):
    prompts = [d["context"] for d in task_datas]
    sampling_params = SamplingParams(
        temperature=0.0,
        seed=42,
        max_tokens=1024,
    )
    handle = llm.generate.remote(prompts, sampling_params, use_tqdm=False)
    return handle, time.time()


def _postprocess_outputs(outputs, task_datas):
    rewards = []
    avg_rewards = []
    for output, data in zip(outputs, task_datas):
        response = output.outputs[0].text
        r = reward_function(response, data["numbers"], data["target"])
        rewards.append(r)
        avg_rewards.append(r["reward"])
    return {
        "rewards": rewards,
        "avg_reward": float(np.mean(avg_rewards)) if avg_rewards else 0.0,
    }


def main(args):
    # Ensure local Ray
    os.environ.pop("RAY_ADDRESS", None)
    os.environ.pop("RAY_HEAD_IP", None)
    os.environ.pop("RAY_GCS_SERVER_ADDRESS", None)
    ray.init(address="local", include_dashboard=False, ignore_reinit_error=True)

    # Logging
    logging_dir = f"{args.experiment_dir}/countdown_{'tpu' if args.tpu_chips else 'nccl'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=logging_dir)

    # Prepare an HF checkpoint for vLLM to load
    model_saves_dir = f"{logging_dir}/model_saves"
    os.makedirs(model_saves_dir, exist_ok=True)

    # Save base model/tokenizer (bf16 on TPU, fp16 on GPU)
    torch_dtype = torch.bfloat16 if args.tpu_chips else torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch_dtype
    ).to("cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    base_model_path = f"{model_saves_dir}/base_model"
    if os.path.exists(base_model_path):
        shutil.rmtree(base_model_path)
    os.makedirs(base_model_path, exist_ok=True)
    tokenizer.save_pretrained(base_model_path)
    base_model.save_pretrained(base_model_path)
    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Load data
    data_path = "countdown/data/countdown.json"
    with open(data_path, "r") as f:
        task_datas = json.load(f)
    task_datas = task_datas[:200]

    # Launch engines
    if args.tpu_chips is not None:
        # TPU path
        engines = launch_engines_tpu(args.num_engines, base_model_path, args.tpu_chips)
        pgs = None
    else:
        # GPU path
        engines, pgs = launch_engines(args.num_engines, base_model_path)

    # Init inter-engine communicator once
    master_address = get_ip()
    master_port = get_open_port()
    ray.get([
        engines[i].es_call.remote(
            "init_inter_engine_group", master_address, master_port, i, args.num_engines
        )
        for i in range(args.num_engines)
    ])

    def cleanup():
        for llm in engines:
            try:
                ray.kill(llm)
            except Exception:
                pass
        if pgs is not None:
            for pg in pgs:
                try:
                    remove_placement_group(pg)
                except Exception:
                    pass
        ray.shutdown()

    def sig_handler(sig, frame):
        cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    # Engines start with identical weights (loaded from the same HF checkpoint)
    # For each iteration:
    # - Explore: per-seed add noise -> eval -> subtract noise
    # - Compute ES update on engine 0 only
    # - Broadcast weights from engine 0 to all engines
    for i in range(args.num_iterations):
        print(f"\n\n=== Generation {i} ===")
        total_iter_start = time.time()

        # Random seeds for population
        seeds = [random.randint(0, 1_000_000) for _ in range(args.population_size)]
        seeds_perf = {}

        # Round-robin scheduling
        seed_iter = iter(seeds)
        inflight = {}
        results_this_gen = []

        # Kick off an eval on each engine
        for eng_idx, llm in enumerate(engines):
            try:
                seed = next(seed_iter)
            except StopIteration:
                break
            # Add exploration noise
            ray.get(llm.es_call.remote("perturb_self_weights", seed, args.sigma, False))
            handle, start_ts = evaluate_countdown_handle(llm, task_datas)
            inflight[handle] = {
                "engine": llm,
                "engine_idx": eng_idx,
                "seed": seed,
                "start_ts": start_ts,
            }

        while inflight:
            done, _ = ray.wait(list(inflight.keys()), num_returns=1)
            h = done[0]
            meta = inflight.pop(h)

            outputs = ray.get(h)
            metrics = _postprocess_outputs(outputs, task_datas)
            elapsed = time.time() - meta["start_ts"]

            seeds_perf[meta["seed"]] = metrics
            results_this_gen.append(
                {"seed": meta["seed"], "avg_reward": metrics["avg_reward"], "time": elapsed}
            )

            llm = meta["engine"]
            # Remove exploration noise
            ray.get(llm.es_call.remote("restore_self_weights", meta["seed"], args.sigma))

            # Schedule next seed on this engine
            try:
                next_seed = next(seed_iter)
            except StopIteration:
                continue

            ray.get(llm.es_call.remote("perturb_self_weights", next_seed, args.sigma, False))
            handle, start_ts = evaluate_countdown_handle(llm, task_datas)
            inflight[handle] = {
                "engine": llm,
                "engine_idx": meta["engine_idx"],
                "seed": next_seed,
                "start_ts": start_ts,
            }
            print(f"Scheduled seed {next_seed} on engine {meta['engine_idx']}")

        # Normalize rewards
        all_avg_rewards = [v["avg_reward"] for v in seeds_perf.values()]
        mean_reward = float(np.mean(all_avg_rewards)) if all_avg_rewards else 0.0
        std_reward = float(np.std(all_avg_rewards)) if all_avg_rewards else 0.0
        min_reward = float(np.min(all_avg_rewards)) if all_avg_rewards else 0.0
        max_reward = float(np.max(all_avg_rewards)) if all_avg_rewards else 0.0

        print(f"Mean reward: {mean_reward}, std: {std_reward}, min: {min_reward}, max: {max_reward}")
        for k in seeds_perf:
            seeds_perf[k]["norm_reward"] = (seeds_perf[k]["avg_reward"] - mean_reward) / (std_reward + 1e-8)
            print(f"Seed {k} normalized reward: {seeds_perf[k]['norm_reward']}")

        writer.add_scalar("reward/mean", mean_reward, i)
        writer.add_scalar("reward/std", std_reward, i)
        writer.add_scalar("reward/min", min_reward, i)
        writer.add_scalar("reward/max", max_reward, i)

        # Compute ES update ONLY on engine 0 (baseline is already current weights)
        per_seed_coeffs = [
            (seed, (args.alpha / args.population_size) * float(seeds_perf[seed]["norm_reward"]))
            for seed in seeds
        ]

        perturb_start = time.time()
        handles = []
        for seed, coeff in per_seed_coeffs:
            # Use sigma_or_scale=1.0 so the applied scale is `coeff`
            handles.append(engines[0].es_call.remote("perturb_self_weights", seed, coeff, False))
        ray.get(handles)
        print(f"Applied perturbations in {time.time() - perturb_start}s")
        writer.add_scalar("time/perturbation_application", time.time() - perturb_start, i)

        # Broadcast updated weights
        broadcast_start = time.time()
        if args.tpu_chips is not None:
            # TPU path: driver-mediated broadcast via Ray object store
            cpu_state = ray.get(engines[0].es_call.remote("dump_state_dict"))
            # fan-out to all engines (including src; idempotent copy)
            ray.get([e.es_call.remote("load_state_dict", cpu_state) for e in engines])
        else:
            # GPU path: in-engine communicator (NCCL) via worker extension
            ray.get([e.es_call.remote("broadcast_all_weights", 0) for e in engines])
        print(f"Broadcasted updated weights in {time.time() - broadcast_start}s")
        writer.add_scalar("time/broadcast", time.time() - broadcast_start, i)

        # Logging per-result and timing
        for res_idx, res in enumerate(results_this_gen):
            print(f"IDX:{res_idx} Seed {res['seed']} avg_reward: {res['avg_reward']}, time: {res['time']}s")
        writer.add_scalar("time/iteration", time.time() - total_iter_start, i)
        print(f"=== Generation {i} finished ===\n")

    # Save final model weights (all engines are in sync; save from engine 0)
    final_model_path = f"{model_saves_dir}/final_model_iteration_{args.num_iterations}"
    os.makedirs(final_model_path, exist_ok=True)
    ray.get(
        engines[0].es_call.remote(
            "save_self_weights_to_disk", f"{final_model_path}/pytorch_model.pth"
        )
    )
    print(f"Final model weights saved to {final_model_path}.")

    cleanup()


if __name__ == "__main__":
    args = parse_args()
    main(args)
