"""
Accelerated version of ES fine tuning of LLMs
"""

import argparse
import gc
import json
import os
import random
import shutil

# signal module lets your program handle Unix-style signals — special messages that the
# operating system (or other processes) send to a running program to tell it to do something
# (like stop, pause, reload, or clean up before exiting)
import signal
import sys
import time
from datetime import datetime
from typing import List
from typing import Tuple

import numpy as np

# ray module is a distributed computing framework for Python. It lets you parallelize
# and scale functions, classes, and workflows across CPUs, GPUs, and multiple machines
import ray
import torch
from ray.util.placement_group import placement_group
from ray.util.placement_group import remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

# vLLM is a fast LLM inference engine designed to serve large transformer models efficiently,
# often achieving 2–4× higher throughput than traditional frameworks such as PyTorch or Hugging
# Face Transformers in standard inference setups
from vllm import LLM
from vllm import SamplingParams
from vllm.utils import get_ip
from vllm.utils import get_open_port

from countdown.countdown_task import reward_function

# Default Hyperparameters
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"  # The name of the model to fine tune
NUM_ITERATIONS = 1000  # Number of ES iterations (generations)
POPULATION_SIZE = 30  # Population size (number of perturbations per iteration)
SIGMA = 0.001  # Standard deviation for weight perturbations (noise scale)
ALPHA = 0.0005  # Learning rate
NUM_ENGINES = 4
EXPERIMENT_DIR = "es-ft-experiment"  # Directory for log files, saved models, etc.
CUDA_DEVICES = "0,1,2,3"
TRAINING_DATA_PATH = "countdown/data/countdown.json"
NUMBER_OF_TRAINING_DATA = 200
SEED_LOWER_BOUND = 0
SEED_UPPER_BOUND = 1_000_000
SAMPLING_TEMPERATURE = (0.0,)
SAMPLING_SEED = (42,)
SAMPLING_MAX_TOKENS = (1024,)


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments and return them
    """
    parser = argparse.ArgumentParser(description="ES Fine-tuning for Countdown Task with multi-engine NCCL sync")

    parser.add_argument("--model_name", type=str, default=MODEL_NAME, help="Name of the model to fine-tune")
    parser.add_argument(
        "--sigma", type=float, default=SIGMA, help="Standard deviation for weight perturbations (noise scale)"
    )
    parser.add_argument("--alpha", type=float, default=ALPHA, help="Learning rate")
    parser.add_argument(
        "--population_size",
        type=int,
        default=POPULATION_SIZE,
        help="Population size (number of perturbations per iteration)",
    )
    parser.add_argument("--num_engines", type=int, default=NUM_ENGINES, help="")
    parser.add_argument(
        "--num_iterations", type=int, default=NUM_ITERATIONS, help="Number of ES iterations (generations)"
    )
    parser.add_argument(
        "--experiment_dir", type=str, default=EXPERIMENT_DIR, help="Directory for log files, saved models, etc."
    )
    parser.add_argument("--cuda_devices", type=str, default=CUDA_DEVICES, help="GPU indices")
    parser.add_argument("--verbose", action="store_true", help="Print verbose logs")
    parser.add_argument(
        "--global_seed",
        type=int,
        help="Global random seed",
    )

    args = parser.parse_args()

    # Optional: scope host visibility; vLLM actors will ignore
    # it and pick device from Process Group (PG)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    # set global random seed
    if args.global_seed is not None:
        random.seed(args.global_seed)
        np.random.seed(args.global_seed)
        # Set the random seed for CPU and the current GPU
        torch.manual_seed(args.global_seed)
        # Set the random seed for all GPUs
        torch.cuda.manual_seed_all(args.global_seed)

    return args


# Define a custom subclass of LLM, which is from the vLLM library
# When you train or serve large models on multiple GPUs, those GPUs need
# to exchange data. For example: synchronizing gradients, broadcasting model
# weights, etc. NCCL handles these operations efficiently at the hardware level
class ESNcclLLM(LLM):
    """
    The class that is run by Ray on a cluster
    """

    def __init__(self, *args, **kwargs):
        # Remove the CUDA_VISIBLE_DEVICES environment variable if it’s set
        # Hand off GPU assignment control to Ray’s internal runtime
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        # Disable a newer (v1) multiprocessing mode in vLLM
        # Use the legacy multiprocessing / single-process-per-GPU mode
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

        super().__init__(*args, **kwargs)


def launch_engines(
    num_engines: int, model_name: str
) -> Tuple[List[ray.actor.ActorHandle], List[ray.util.placement_group.PlacementGroup]]:
    """
    Launches ESNcclLLM class on the Ray cluster

    :param numn_engines
        Number of nodes in the cluster

    : model_name:
        Name of the model

    :return:
        Returns handles and placement gropu lists
    """
    # Create a list of Ray placement groups (PGs), each reserving 1 GPU (and 0 CPUs)
    pgs = [
        # In Ray, a PG is a way to pre-reserve cluster resources. Think of it as a resource container
        # Each dictionary in the list describes a resource bundle: set of resources that should be co-located on 1 node
        # By default, placement groups are deleted when the creating job or task finishes. lifetime="detached" makes
        # the placement group persistent, i.e. it stays alive even after the creating job ends. You must later remove
        # them manually
        placement_group([{"GPU": 1, "CPU": 0}], lifetime="detached")
        for _ in range(num_engines)
    ]

    # When you create a placement group, it’s not ready immediately
    # pg.ready() blocks until resources for PG are reserved
    # ray.get() blocks until the resources for every PG in pgs are ready
    ray.get([pg.ready() for pg in pgs])

    strategies = [
        # Normally, when you start a Ray task/actor, Ray decides where to schedule it based on available resources
        # By passing a PlacementGroupSchedulingStrategy, you override that default, explicitly telling Ray:
        # “Run this task/actor inside this specific placement group”
        PlacementGroupSchedulingStrategy(
            # Tells Ray which placement group to use. All resources (e.g. GPUs, CPUs) required by this task/actor
            # will come from that specific placement group
            placement_group=pg,
            # Ensures that child tasks or sub-actors created by this actor/task are also scheduled inside
            # the same placement group
            placement_group_capture_child_tasks=True,
            # Placement groups can contain multiple bundles — each bundle is a resource set (like one GPU per engine)
            # This parameter specifies which bundle inside the group to use
            placement_group_bundle_index=0,
        )
        for pg in pgs
    ]

    engines = [
        # ray.remote() marks a class/function to run remotely on the Ray cluster — not on the local Python process
        # Request 0 CPUs/GPUs as the allocation is already controlled by the placement group
        ray.remote(num_cpus=0, num_gpus=0, scheduling_strategy=strategy)(ESNcclLLM).remote(
            # remote() creates an instance of remote actor —> actually launches ESNcclLLM process on the Ray cluster
            # All the keyword arguments below are passed to the actor’s __init__ method
            # The model name or path to load
            model=model_name,
            # Number of GPUs to use for tensor parallelism — 1 means no sharding
            tensor_parallel_size=1,
            # Tells the model to use Ray for distributed coordination
            distributed_executor_backend="ray",
            # A custom worker class to extend model behavior (like logging, metrics, etc.)
            worker_extension_cls="utils.worker_extn.WorkerExtension",
            # Loads model weights in half-precision to save memory
            dtype="float16",
            # Controls caching of prefill states (used in LLM inference)
            enable_prefix_caching=False,
            # Disables “eager execution” for performance — likely uses compiled graph mode
            enforce_eager=False,
        )
        for strategy in strategies
    ]

    return engines, pgs


def evaluate_countdown_handle(llm, task_data):
    """
    Evaluate the LLM on the training dataset

    :param llm
        The LLM to evaluate
    :param task_data
        The training data for a specific task

    :return:
        The handle to the generated response and the current time
    """
    prompts = [data["context"] for data in task_data]

    sampling_params = SamplingParams(
        temperature=SAMPLING_TEMPERATURE,
        seed=SAMPLING_SEED,
        max_tokens=SAMPLING_MAX_TOKENS,
    )
    handle = llm.generate.remote(prompts, sampling_params, use_tqdm=False)
    return handle, time.time()


def _postprocess_outputs(outputs, task_data):
    rewards = []
    avg_rewards = []

    for output, data in zip(outputs, task_data):
        response = output.outputs[0].text
        reward_dict = reward_function(response, data["numbers"], data["target"])
        rewards.append(reward_dict)
        avg_rewards.append(reward_dict["reward"])

    return {
        "rewards": rewards,
        "avg_reward": float(np.mean(avg_rewards)) if avg_rewards else 0.0,
    }


def main(args):  # pylint: disable=too-many-branches, too-many-statements, too-many-locals
    """
    Main method for accelerated version of ES fine tuning of LLMs
    """
    # Ensure local Ray
    os.environ.pop("RAY_ADDRESS", None)  # The IP address of the Ray cluster to connect to
    os.environ.pop("RAY_HEAD_IP", None)  # The IP address of the Ray head node in a cluster
    # The IP address of the Global Control Store (GCS) server used internally by Ray for cluster coordination
    os.environ.pop("RAY_GCS_SERVER_ADDRESS", None)

    # Initialize a Ray runtime — essentially start (or connect to) a Ray cluster
    # "local" means run everything on this machine, including the scheduler, object store, and workers.
    # Disable the Ray web dashboard, which normally runs at http://127.0.0.1:8265
    # Calling ray.init() twice in the same Python session raises an error — Ray doesn’t allow multiple initializations
    # by default. Setting ignore_reinit_error=True suppresses that, so code can run even if Ray is already initialized.
    ray.init(address="local", include_dashboard=False, ignore_reinit_error=True)

    # Set the logging directory
    logging_dir = f"{args.experiment_dir}/countdown_nccl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Create a TensorBoard summary writer that logs training metrics and other data to a directory
    writer = SummaryWriter(log_dir=logging_dir)

    # Prepare an HF checkpoint for vLLM to load
    model_saves_dir = f"{logging_dir}/model_saves"
    os.makedirs(model_saves_dir, exist_ok=True)

    # Load the pretrained causal language model
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16).to("cpu")

    # Load the tokenizer associated with a pretrained causal model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Create a directory for base model. Delete the directory if it already exists
    base_model_path = f"{model_saves_dir}/base_model"
    if os.path.exists(base_model_path):
        shutil.rmtree(base_model_path)
    os.makedirs(base_model_path, exist_ok=True)

    # Save the tokenizer’s files (vocabulary, merges, configs, etc.) to a local directory
    tokenizer.save_pretrained(base_model_path)
    # Save the model’s weights and configuration to a local directory
    base_model.save_pretrained(base_model_path)
    # Delete the variable base_model from memory, effectively freeing up the Python object
    del base_model

    # Free CPU memory
    gc.collect()
    if torch.cuda.is_available():
        # Free GPU memory
        torch.cuda.empty_cache()

    # Load data
    with open(TRAINING_DATA_PATH, "r", encoding="utf-8") as f:
        task_data = json.load(f)
    task_data = task_data[:NUMBER_OF_TRAINING_DATA]

    # Launch engines
    engines, placement_groups = launch_engines(args.num_engines, base_model_path)

    # Init inter-engine communicator once
    # Get the current machine’s IP address
    master_address = get_ip()
    # Find a currently unused TCP port on the current machine that can be bound for communication
    master_port = get_open_port()

    # Initialize a collective communication group among a list of Ray actors
    ray.get(
        [
            # This calls a remote method named collective_rpc on each Ray actor handle in the engines list
            # The use of .remote() means the call is asynchronous and returns immediately with an ObjectRef
            engines[idx].collective_rpc.remote(
                # This is the name of the specific function being called within the remote actor. Its purpose is to
                # perform the necessary setup to create a communication group among all participating actors
                "init_inter_engine_group",
                # These arguments are passed to the remote init_inter_engine_group function:
                args=(master_address, master_port, idx, args.num_engines),
            )
            for idx in range(args.num_engines)
        ]
        # The list comprehension returns a list of ObjectRefs, each representing the pending result of the asynchronous
        # collective_rpc call on its respective engine. These can later be retrieved using ray.get() to ensure all
        # engines have successfully joined the group
    )

    def cleanup():
        """
        Safely and gracefully shut down all distributed Ray resources that were created earlier
        including model actors (engines), placement groups, and finally the Ray runtime itself
        """
        for engine in engines:
            try:
                # Forcibly terminate the remote actor
                # Free GPU/CPU resources used by each engine
                ray.kill(engine)
            except Exception as e:  # pylint: disable=broad-exception-caught
                print(f"ray.kill() for engine {engine} resulted in exception {type(e).__name__}: {e}")
        for pg in placement_groups:
            try:
                # Release the placement_group resource reservations
                remove_placement_group(pg)
            except Exception as e:  # pylint: disable=broad-exception-caught
                print(f"remove_placement_group for placement group {pg} resulted in exception {type(e).__name__}: {e}")

        # End the Ray session cleanly
        ray.shutdown()

    def sig_handler():
        cleanup()
        sys.exit(0)

    # Register the signal handlers. This tells Python’s signal module that when the program receives
    #     SIGINT → usually sent when you press Ctrl + C in the terminal, or
    #     SIGTERM → usually sent by the OS or process manager (like Docker, Kubernetes, systemd) to request termination
    # Run sig_handler instead of quitting immediately. So instead of abruptly killing your Ray workers and leaving GPUs
    # occupied, your program will run cleanup() to free all Ray resources then exits cleanly with sys.exit(0).
    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    # Engines start with identical weights (loaded from the same HF checkpoint)
    # For each iteration:
    # - Explore: per-seed add noise -> eval -> subtract noise (GPU-only)
    # - Compute ES update on engine 0 only
    # - Broadcast weights from engine 0 to all engines (NCCL)
    for idx in range(args.num_iterations):
        print(f"\n\n=== Generation {idx} ===")
        total_iter_start = time.time()

        # Random seeds for population
        seeds = [random.randint(SEED_LOWER_BOUND, SEED_UPPER_BOUND) for _ in range(args.population_size)]
        seeds_perf = {}

        # Round-robin scheduling
        seed_iter = idx(seeds)
        inflight = {}
        results_this_gen = []

        # Kick off an eval on each engine
        for eng_idx, llm in enumerate(engines):
            try:
                seed = next(seed_iter)
            except StopIteration:
                break

            # Add exploration noise
            ray.get(llm.collective_rpc.remote("perturb_self_weights", args=(seed, args.sigma, False)))

            finished_task_handle, start_ts = evaluate_countdown_handle(llm, task_data)

            # inflight is a dictionary used to keep track of currently running jobs or requests
            # Keys are unique request identifiers (handle), often returned when a new job starts
            # Values hold metadata about that job.
            inflight[finished_task_handle] = {
                "engine": llm,
                "engine_idx": eng_idx,
                "seed": seed,
                "start_ts": start_ts,
            }

        # Create a loop that continues running as long as the inflight dictionary is not empty,
        # meaning there are still active or ongoing tasks being tracked
        while inflight:
            # Block until num_returns (set to 1) of the given Ray object references are ready (i.e., task is finished)
            finished_task, _ = ray.wait(list(inflight.keys()), num_returns=1)
            # Extract the handle (the completed Ray object reference) of the finished task
            finished_task_handle = finished_task[0]
            # Remove the completed task from the inflight tracking dictionary
            metadata = inflight.pop(finished_task_handle)

            # finished_task_handle is a Ray object reference (from an async task or actor call).
            # get(handle) retrieves the actual result of that remote computation — blocking until it’s finished
            outputs = ray.get(finished_task_handle)

            metrics = _postprocess_outputs(outputs, task_data)

            elapsed_time = time.time() - metadata["start_ts"]

            seeds_perf[metadata["seed"]] = metrics

            results_this_gen.append(
                {
                    "seed": metadata["seed"],
                    "avg_reward": metrics["avg_reward"],
                    "time": elapsed_time,
                }
            )

            llm = metadata["engine"]
            # Remove exploration noise
            ray.get(llm.collective_rpc.remote("restore_self_weights", args=(metadata["seed"], args.sigma)))

            # Schedule next seed on this engine
            try:
                next_seed = next(seed_iter)
            except StopIteration:
                continue

            ray.get(llm.collective_rpc.remote("perturb_self_weights", args=(next_seed, args.sigma, False)))
            finished_task_handle, start_ts = evaluate_countdown_handle(llm, task_data)

            inflight[finished_task_handle] = {
                "engine": llm,
                "engine_idx": metadata["engine_idx"],
                "seed": next_seed,
                "start_ts": start_ts,
            }
            if args.verbose:
                print(f"Scheduled seed {next_seed} on engine {metadata['engine_idx']}")

        # Normalize rewards
        all_avg_rewards = [metric["avg_reward"] for metric in seeds_perf.values()]
        mean_reward = float(np.mean(all_avg_rewards)) if all_avg_rewards else 0.0
        std_reward = float(np.std(all_avg_rewards)) if all_avg_rewards else 0.0
        min_reward = float(np.min(all_avg_rewards)) if all_avg_rewards else 0.0
        max_reward = float(np.max(all_avg_rewards)) if all_avg_rewards else 0.0

        print(f"Mean reward: {mean_reward}, std: {std_reward}, min: {min_reward}, max: {max_reward}")

        # Standardize rewards by subtracting mean and dividing by standard deviation
        for a_seed, a_metric in seeds_perf.items():
            a_metric["norm_reward"] = (a_metric["avg_reward"] - mean_reward) / (std_reward + 1e-8)

            if args.verbose:
                print(f"Seed {a_seed} normalized reward: {seeds_perf[a_seed]['norm_reward']}")

        writer.add_scalar("reward/mean", mean_reward, idx)
        writer.add_scalar("reward/std", std_reward, idx)
        writer.add_scalar("reward/min", min_reward, idx)
        writer.add_scalar("reward/max", max_reward, idx)

        # Compute ES update ONLY on engine 0 (baseline is already current weights)
        per_seed_coeffs = [
            (
                seed,
                (args.alpha / args.population_size) * float(seeds_perf[seed]["norm_reward"]),
            )
            for seed in seeds
        ]

        perturb_start = time.time()

        handles = []
        for seed, coeff in per_seed_coeffs:
            # Use sigma_or_scale=1.0 so the applied scale is `coeff`
            handles.append(engines[0].collective_rpc.remote("perturb_self_weights", args=(seed, coeff, False)))

        ray.get(handles)

        if args.verbose:
            print(f"Applied perturbations in {time.time() - perturb_start}s")
        writer.add_scalar("time/perturbation_application", time.time() - perturb_start, idx)

        # Broadcast updated weights from engine 0 to all engines (avoid CPU copies)
        broadcast_start = time.time()

        ray.get([engine.collective_rpc.remote("broadcast_all_weights", args=(0,)) for engine in engines])

        if args.verbose:
            print(f"Broadcasted updated weights in {time.time() - broadcast_start}s")
        writer.add_scalar("time/broadcast", time.time() - broadcast_start, idx)

        # Logging per-result and timing
        if args.verbose:
            for result_idx, result in enumerate(results_this_gen):
                print(
                    f"IDX:{result_idx} Seed {result['seed']} avg_reward: {result['avg_reward']}, "
                    f"time: {result['time']}s"
                )

        total_iter_end = time.time()
        writer.add_scalar("time/iteration", total_iter_end - total_iter_start, idx)
        print(f"wall clock time for iteration {idx}: {total_iter_end - total_iter_start}s")
        print(f"=== Generation {idx} finished ===\n")

    # Save final model weights (all engines are in sync; save from engine 0)
    final_model_path = f"{model_saves_dir}/final_model_iteration_{args.num_iterations}"
    os.makedirs(final_model_path, exist_ok=True)

    ray.get(
        engines[0].collective_rpc.remote("save_self_weights_to_disk", args=(f"{final_model_path}/pytorch_model.pth",))
    )
    print(f"Final model weights saved to {final_model_path}.")

    cleanup()


if __name__ == "__main__":
    cli_args = parse_args()
    main(cli_args)
