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
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer

# We intentionally DO NOT import vLLM or JAX at module scope.
# They are imported *inside* the Ray actor so the engine can live
# in its own runtime_env with a fresh libtpu/JAX stack.

# ----------------------------- CLI ------------------------------------------
SIGMA = 0.001
ALPHA = 0.0005
POPULATION_SIZE = 30
NUM_ENGINES = 4
NUM_ITERATIONS = 1000
EXPERIMENT_DIR = "es-ft-experiment"

def parse_args():
    p = argparse.ArgumentParser(description="ES fine-tuning on TPU v6e with isolated vLLM/JAX actor envs")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--sigma", type=float, default=SIGMA)
    p.add_argument("--alpha", type=float, default=ALPHA)
    p.add_argument("--population_size", type=int, default=POPULATION_SIZE)
    p.add_argument("--num_engines", type=int, default=NUM_ENGINES)
    p.add_argument("--num_iterations", type=int, default=NUM_ITERATIONS)
    p.add_argument("--experiment_dir", type=str, default=EXPERIMENT_DIR)
    p.add_argument("--global_seed", type=int, default=None)
    p.add_argument("--tpu_chips", type=str, default=None,
                   help="Comma-separated TPU chip ids per engine, e.g. '0,1,2,3'")
    # Per-actor pip pins (override defaults if needed)
    p.add_argument("--engine_vllm", type=str, default="vllm==0.11.0")
    p.add_argument("--engine_jax", type=str, default="jax[tpu]==0.7.2")
    p.add_argument("--engine_jaxlib", type=str, default="jaxlib==0.7.2")
    p.add_argument("--engine_libtpu", type=str, default="libtpu==0.0.27")  # Latest (Oct 2025)
    p.add_argument("--engine_torch", type=str, default="torch==2.8.0")
    p.add_argument("--engine_transformers", type=str, default="transformers>=4.30.0")
    p.add_argument("--engine_numpy", type=str, default="numpy>=1.21.0")
    return p.parse_args()

# ------------------------- Countdown task reward ----------------------------
# We import the reward function lazily to keep this file self-contained if desired.
def _load_reward():
    try:
        from countdown.countdown_task import reward_function
        return reward_function
    except Exception as e:
        raise RuntimeError("Could not import countdown.reward_function. Ensure countdown/ is on PYTHONPATH.") from e

# ----------------------------- Actor ----------------------------------------
# We define an actor class that imports vLLM *inside* the actor process.
# That way, the driver does not need vLLM/JAX/libtpu installed at all.

@ray.remote
class TpuEngineActor:
    def __init__(self, model_dir: str, dtype: str = "bfloat16"):
        # The driver has provisioned a per-actor pip env via runtime_env.pip.
        # Apply the JAX Pallas compat shim BEFORE importing vLLM.
        self._apply_jax_pallas_memoryspace_compat()

        # Import vLLM now that JAX is patched
        from vllm import LLM, SamplingParams  # noqa: WPS433

        # Optional envs to keep vLLM stable in actors
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        os.environ.setdefault("VLLM_DEVICE", "tpu")

        # Instantiate the engine
        self._SamplingParams = SamplingParams
        self._llm = LLM(
            model=model_dir,
            tensor_parallel_size=1,          # one chip per actor
            distributed_executor_backend="ray",
            dtype=dtype,
            enable_prefix_caching=False,
            enforce_eager=False,
        )

        # Discover params accessors for ES ops (V1 then V0)
        self._named_params_iter = self._resolve_param_accessor()

    # -------------------- internal: pallas compat shim --------------------
    @staticmethod
    def _apply_jax_pallas_memoryspace_compat():
        try:
            pltpu = importlib.import_module("jax.experimental.pallas.tpu")
            ms = getattr(pltpu, "MemorySpace", None)
            for alias in ("TPUMemorySpace", "TpuMemorySpace"):
                if ms is not None and not hasattr(pltpu, alias):
                    setattr(pltpu, alias, ms)
        except Exception:
            # Best-effort; some builds don't ship pallas.tpu
            pass

    # -------------------- internal: params accessor -----------------------
    def _resolve_param_accessor(self):
        # vLLM V1 path
        if hasattr(self._llm, "llm_engine") and hasattr(self._llm.llm_engine, "engine_core"):
            model_executor = self._llm.llm_engine.engine_core.model_executor
            if hasattr(model_executor, "driver_worker"):
                return model_executor.driver_worker.model_runner.model.named_parameters
        # vLLM V0 path
        if hasattr(self._llm, "llm_engine") and hasattr(self._llm.llm_engine, "model_executor"):
            model_executor = self._llm.llm_engine.model_executor
            if hasattr(model_executor, "driver_worker"):
                return model_executor.driver_worker.model_runner.model.named_parameters
        raise RuntimeError("Could not locate model parameters in vLLM engine")

    # -------------------------- Inference ---------------------------------
    def generate(self, prompts, temperature: float = 0.0, seed: int = 42, max_tokens: int = 1024):
        sampling_params = self._SamplingParams(temperature=temperature, seed=seed, max_tokens=max_tokens)
        return self._llm.generate(prompts, sampling_params, use_tqdm=False)

    # -------------------------- ES ops ------------------------------------
    def perturb_self_weights(self, seed: int, sigma_or_scale: float, negate: bool = False):
        scale = float(sigma_or_scale)
        sign = -1.0 if negate else 1.0
        for _, p in self._named_params_iter():
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(seed))
            noise = torch.randn(p.shape, dtype=p.dtype, generator=gen).to(p.device, non_blocking=True)
            p.data.add_(sign * scale * noise)
            del noise
        return True

    def restore_self_weights(self, seed: int, sigma: float):
        return self.perturb_self_weights(seed, sigma_or_scale=sigma, negate=True)

    def dump_state_dict(self):
        sd = {}
        for name, p in self._named_params_iter():
            sd[name] = p.detach().to("cpu")
        return sd

    def load_state_dict(self, state_dict):
        for name, p in self._named_params_iter():
            src = state_dict[name]
            p.data.copy_(src.to(p.device, non_blocking=True))
        return True

    def save_self_weights_to_disk(self, filepath: str):
        state_dict = {}
        for name, p in self._named_params_iter():
            state_dict[name] = p.detach().cpu()
        torch.save(state_dict, filepath)
        gc.collect()
        return True

# -------------------- TPU launcher with isolated actor env ------------------
def launch_tpu_engines(num_engines: int, model_dir: str, tpu_chips_csv: str | None,
                       pip_versions: dict[str, str]):
    # Parse chip list; default 0..N-1
    if not tpu_chips_csv or tpu_chips_csv.strip() == "":
        chips = [str(i) for i in range(max(1, num_engines))]
    else:
        chips = [c.strip() for c in tpu_chips_csv.split(",") if c.strip() != ""]
    if len(chips) < num_engines:
        raise ValueError(f"Need at least {num_engines} TPU chips, got {len(chips)} from --tpu_chips={tpu_chips_csv!r}")

    pip_list = [
        pip_versions["engine_vllm"],
        pip_versions["engine_jax"],
        pip_versions["engine_jaxlib"],
        pip_versions["engine_libtpu"],
        pip_versions["engine_torch"],           # CPU-only torch is fine
        pip_versions["engine_transformers"],
        pip_versions["engine_numpy"],
        "tensorboard>=2.20.0",
        "psutil>=5.8.0",
    ]

    engines = []
    for i in range(num_engines):
        chip_id = chips[i]
        runtime_env = {
            "env_vars": {
                "TPU_VISIBLE_CHIPS": chip_id,
                "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
                "VLLM_DEVICE": "tpu",
                "HF_HUB_DISABLE_TELEMETRY": "1",
            },
            "pip": pip_list,
        }
        actor = TpuEngineActor.options(num_cpus=0, scheduling_strategy="DEFAULT",
                                       runtime_env=runtime_env).remote(
            model_dir=model_dir, dtype="bfloat16"
        )
        engines.append(actor)
    return engines

# ------------------------- Evaluation helpers -------------------------------
def evaluate_countdown_handle(actor, task_datas):
    prompts = [d["context"] for d in task_datas]
    return actor.generate.remote(prompts, temperature=0.0, seed=42, max_tokens=1024), time.time()

def _postprocess_outputs(outputs, task_datas, reward_fn):
    rewards, avg_rewards = [], []
    for output, data in zip(outputs, task_datas):
        response = output.outputs[0].text
        r = reward_fn(response, data["numbers"], data["target"])
        rewards.append(r)
        avg_rewards.append(r["reward"])
    return {"rewards": rewards, "avg_reward": float(np.mean(avg_rewards)) if avg_rewards else 0.0}

# --------------------------------- Main -------------------------------------
def main(args):
    # No GPU/torch_xla support in this script by design.
    # Driver env should not import JAX/vLLM—engines handle that in their own env.

    if args.global_seed is not None:
        random.seed(args.global_seed)
        np.random.seed(args.global_seed)
        torch.manual_seed(args.global_seed)

    # Ray local
    os.environ.pop("RAY_ADDRESS", None)
    os.environ.pop("RAY_HEAD_IP", None)
    os.environ.pop("RAY_GCS_SERVER_ADDRESS", None)
    ray.init(address="local", include_dashboard=False, ignore_reinit_error=True)

    reward_fn = _load_reward()

    # Logging
    logging_dir = f"{args.experiment_dir}/countdown_tpu_isolated_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=logging_dir)

    # Prepare HF checkpoint for vLLM to load
    model_saves_dir = f"{logging_dir}/model_saves"
    os.makedirs(model_saves_dir, exist_ok=True)

    base_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16).to("cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    base_model_path = f"{model_saves_dir}/base_model"
    if os.path.exists(base_model_path):
        shutil.rmtree(base_model_path)
    os.makedirs(base_model_path, exist_ok=True)
    tokenizer.save_pretrained(base_model_path)
    base_model.save_pretrained(base_model_path)
    del base_model
    gc.collect()

    # Load task data
    data_path = "countdown/data/countdown.json"
    with open(data_path, "r") as f:
        task_datas = json.load(f)
    task_datas = task_datas[:200]

    # Launch TPU engines with per-actor pip env (fresh libtpu/JAX/vLLM)
    pip_versions = {
        "engine_vllm": args.engine_vllm,
        "engine_jax": args.engine_jax,
        "engine_jaxlib": args.engine_jaxlib,
        "engine_libtpu": args.engine_libtpu,
        "engine_torch": args.engine_torch,
        "engine_transformers": args.engine_transformers,
        "engine_numpy": args.engine_numpy,
    }
    engines = launch_tpu_engines(args.num_engines, base_model_path, args.tpu_chips, pip_versions)

    # ES loop
    for i in range(args.num_iterations):
        print(f"\n=== Generation {i} ===")
        t0 = time.time()

        # Seeds for population
        seeds = [random.randint(0, 1_000_000) for _ in range(args.population_size)]
        seeds_perf = {}

        # Round-robin scheduling across engines
        seed_iter = iter(seeds)
        inflight = {}
        results_this_gen = []

        # Kick off one eval per engine
        for eng_idx, actor in enumerate(engines):
            try:
                seed = next(seed_iter)
            except StopIteration:
                break
            ray.get(actor.perturb_self_weights.remote(seed, args.sigma, False))
            handle, st = evaluate_countdown_handle(actor, task_datas)
            inflight[handle] = {"actor": actor, "eng_idx": eng_idx, "seed": seed, "start_ts": st}

        while inflight:
            done, _ = ray.wait(list(inflight.keys()), num_returns=1)
            h = done[0]
            meta = inflight.pop(h)

            outputs = ray.get(h)
            metrics = _postprocess_outputs(outputs, task_datas, reward_fn)
            elapsed = time.time() - meta["start_ts"]

            seeds_perf[meta["seed"]] = metrics
            results_this_gen.append({"seed": meta["seed"], "avg_reward": metrics["avg_reward"], "time": elapsed})

            actor = meta["actor"]
            ray.get(actor.restore_self_weights.remote(meta["seed"], args.sigma))

            try:
                next_seed = next(seed_iter)
            except StopIteration:
                continue
            ray.get(actor.perturb_self_weights.remote(next_seed, args.sigma, False))
            handle, st = evaluate_countdown_handle(actor, task_datas)
            inflight[handle] = {"actor": actor, "eng_idx": meta["eng_idx"], "seed": next_seed, "start_ts": st}
            print(f"Scheduled seed {next_seed} on engine {meta['eng_idx']}")

        # Normalize
        all_avg = [v["avg_reward"] for v in seeds_perf.values()]
        mean_r = float(np.mean(all_avg)) if all_avg else 0.0
        std_r = float(np.std(all_avg)) if all_avg else 0.0
        min_r = float(np.min(all_avg)) if all_avg else 0.0
        max_r = float(np.max(all_avg)) if all_avg else 0.0
        print(f"Mean reward: {mean_r}, std: {std_r}, min: {min_r}, max: {max_r}")

        for k in seeds_perf:
            seeds_perf[k]["norm_reward"] = (seeds_perf[k]["avg_reward"] - mean_r) / (std_r + 1e-8)
            print(f"Seed {k} normalized reward: {seeds_perf[k]['norm_reward']}")

        writer.add_scalar("reward/mean", mean_r, i)
        writer.add_scalar("reward/std", std_r, i)
        writer.add_scalar("reward/min", min_r, i)
        writer.add_scalar("reward/max", max_r, i)

        # ES update on engine 0
        per_seed_coeffs = [(seed, (args.alpha / args.population_size) * float(seeds_perf[seed]["norm_reward"]))
                           for seed in seeds]

        t1 = time.time()
        ray.get([engines[0].perturb_self_weights.remote(seed, coeff, False) for seed, coeff in per_seed_coeffs])
        writer.add_scalar("time/perturbation_application", time.time() - t1, i)

        # Broadcast updated weights: driver-mediated fan-out via Ray object store
        t2 = time.time()
        cpu_state = ray.get(engines[0].dump_state_dict.remote())
        ray.get([e.load_state_dict.remote(cpu_state) for e in engines])
        writer.add_scalar("time/broadcast", time.time() - t2, i)

        # Log per-result timing
        for idx, res in enumerate(results_this_gen):
            print(f"IDX:{idx} Seed {res['seed']} avg_reward:{res['avg_reward']:.4f} time:{res['time']:.3f}s")
        writer.add_scalar("time/iteration", time.time() - t0, i)

    # Save final weights
    final_dir = f"{model_saves_dir}/final_model_iteration_{args.num_iterations}"
    os.makedirs(final_dir, exist_ok=True)
    ray.get(engines[0].save_self_weights_to_disk.remote(f"{final_dir}/pytorch_model.pth"))
    print(f"Final model weights saved to {final_dir}.")

    ray.shutdown()

if __name__ == "__main__":
    args = parse_args()
    main(args)
