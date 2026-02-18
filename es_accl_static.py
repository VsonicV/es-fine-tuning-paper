import argparse
from datetime import datetime
import gc
import json
import os
import random
import shutil
import time
from typing import List, Dict, Any

import numpy as np
import ray
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams, TokensPrompt
from vllm.utils import get_ip, get_open_port
import tempfile

from countdown.countdown_task import reward_function

# Default Hyperparameters
SIGMA = 0.001
ALPHA = 0.0005
POPULATION_SIZE = 30
NUM_ENGINES = 4
NUM_ITERATIONS = 800
EXPERIMENT_DIR = "es-ft-experiment"

# GRPOZero align
SYSTEM_MESSAGE = (
    "You are a helpful assistant. You first think about the reasoning process "
    "in your mind and then provide the user with the answer."
)
USER_TEMPLATE = (
    "Using the numbers {numbers}, create an equation that equals {target}. "
    "You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. "
    "Show your work in <think> </think> tags. "
    "And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>."
)
RESPONSE_PROMPT = "Let me solve this step by step.\n<think>"

def parse_args():
    parser = argparse.ArgumentParser(
        description="ES Fine-tuning for Countdown Task with static scheduling"
    )
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--sigma", type=float, default=SIGMA)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--population_size", type=int, default=POPULATION_SIZE)
    parser.add_argument("--num_engines", type=int, default=NUM_ENGINES)
    parser.add_argument("--num_iterations", type=int, default=NUM_ITERATIONS)
    parser.add_argument("--experiment_dir", type=str, default=EXPERIMENT_DIR)
    parser.add_argument("--cuda_devices", type=str, default="0,1,2,3")
    parser.add_argument("--verbose", action="store_true", help="Print verbose logs")
    parser.add_argument("--global_seed", type=int, help="Global random seed")
    parser.add_argument("--precision", type=str, choices=["float16", "bfloat16", "float32"],
                        default="float16", help="Precision for model weights")

    # Evaluation args
    parser.add_argument("--eval_data_path", type=str,
                        default="countdown/data/countdown.json",
                        help="Path to evaluation JSON (same schema as train)")
    parser.add_argument("--eval_interval", type=int, default=25,
                        help="Evaluate every N iterations (0 disables)")
    parser.add_argument("--eval_batch_size", type=int, default=512,
                        help="Batch size for evaluation generation")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    if args.global_seed is not None:
        random.seed(args.global_seed)
        np.random.seed(args.global_seed)
        torch.manual_seed(args.global_seed)
        torch.cuda.manual_seed_all(args.global_seed)
        os.environ["PYTHONHASHSEED"] = str(args.global_seed)
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

    return args

# Align with GRPOZero
def _process_context(task_data, tokenizer):
    numbers = task_data["numbers"]
    target = task_data["target"]
    user_content = USER_TEMPLATE.format(numbers=numbers, target=target)
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_content}
    ]

    messages = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    ) + RESPONSE_PROMPT

    prompts = tokenizer(
        messages,
    )

    return TokensPrompt(prompt_token_ids=prompts['input_ids'])


class ESNcclLLM(LLM):
    def __init__(self, *args, **kwargs):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        # Avoid adding non-serializable objects; rely on worker-side ops
        super().__init__(*args, **kwargs)

def launch_engines(num_engines, model_name, precision="bfloat16"):
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
            dtype=precision,
            enable_prefix_caching=False,
            enforce_eager=False, # NOTE: THIS IS IMPORTANT FOR ACCELERATION
            gpu_memory_utilization=0.9
        )
        for strategy in strategies
    ]
    return engines, pgs

def evaluate_countdown(llm, prompts, seed: int):
    sampling_params = SamplingParams(temperature=0.0, seed=seed, max_tokens=1024)
    return llm.generate.remote(
        prompts, 
        sampling_params, 
        use_tqdm=False
    )

def _postprocess_outputs(outputs, task_datas):
    rewards = []
    avg_rewards = []
    format_rewards = []
    answer_rewards = []
    for output, data in zip(outputs, task_datas):
        response = RESPONSE_PROMPT + output.outputs[0].text
        r = reward_function(response, data["numbers"], data["target"])
        rewards.append(r)
        avg_rewards.append(r["reward"])
        if "reward_info" in r:
            format_rewards.append(r["reward_info"].get("format_reward", 0.0))
            answer_rewards.append(r["reward_info"].get("answer_reward", 0.0))
    avg_format = float(np.mean(format_rewards)) if format_rewards else 0.0
    avg_answer = float(np.mean(answer_rewards)) if answer_rewards else 0.0
    accuracy = (sum(1 for a in answer_rewards if a > 0) / len(answer_rewards) * 100.0) if answer_rewards else 0.0
    return {
        "rewards": rewards,
        "avg_reward": float(np.mean(avg_rewards)) if avg_rewards else 0.0,
        "avg_format": avg_format,
        "avg_answer": avg_answer,
        "accuracy": accuracy,
    }

def evaluate_model(llm, eval_task_datas: List[Dict[str, Any]], writer, step: int, args):
    if not eval_task_datas:
        return
    batch_size = max(1, args.eval_batch_size)
    eval_seed = args.global_seed if args.global_seed is not None else 999
    sampling_params = SamplingParams(temperature=0.0, seed=eval_seed, max_tokens=1024)

    all_rewards = []
    format_rewards = []
    answer_rewards = []
    start = time.time()

    for b in range(0, len(eval_task_datas), batch_size):
        batch = eval_task_datas[b:b+batch_size]
        prompts = [d["context"] for d in batch]

        # prompts = [_process_context(d, llm.tokenizer) for d in batch]

        outputs = ray.get(
            llm.generate.remote(
                prompts, 
                sampling_params, 
                use_tqdm=False
            )
        )
        
        for idx,(out, data) in enumerate(zip(outputs, batch)):
            # for inspection
            if idx == 0:
                print(f"Eval Sample Response:\n{RESPONSE_PROMPT + out.outputs[0].text}\n---")
                print(f"Target Answer: {data['target']}\n===")
                print(f"The rewards: {reward_function(RESPONSE_PROMPT + out.outputs[0].text, data['numbers'], data['target'])}\n===")
            response = RESPONSE_PROMPT + out.outputs[0].text
            r = reward_function(response, data["numbers"], data["target"])
            all_rewards.append(r["reward"])
            if "reward_info" in r:
                format_rewards.append(r["reward_info"].get("format_reward", 0.0))
                answer_rewards.append(r["reward_info"].get("answer_reward", 0.0))
    elapsed = time.time() - start

    avg_reward = float(np.mean(all_rewards)) if all_rewards else 0.0
    std_reward = float(np.std(all_rewards)) if all_rewards else 0.0
    min_reward = float(np.min(all_rewards)) if all_rewards else 0.0
    max_reward = float(np.max(all_rewards)) if all_rewards else 0.0
    avg_format = float(np.mean(format_rewards)) if format_rewards else 0.0
    avg_answer = float(np.mean(answer_rewards)) if answer_rewards else 0.0
    accuracy = (sum(1 for a in answer_rewards if a > 0) / len(answer_rewards) * 100.0) if answer_rewards else 0.0

    print(f"[Eval @ step {step}] avg_reward={avg_reward:.4f} ± {std_reward:.4f} "
          f"range=[{min_reward:.4f},{max_reward:.4f}] format={avg_format:.4f} "
          f"answer={avg_answer:.4f} acc={accuracy:.1f}% time={elapsed:.2f}s")

    writer.add_scalar("eval/avg_reward", avg_reward, step)
    writer.add_scalar("eval/std_reward", std_reward, step)
    writer.add_scalar("eval/min_reward", min_reward, step)
    writer.add_scalar("eval/max_reward", max_reward, step)
    writer.add_scalar("eval/format_reward", avg_format, step)
    writer.add_scalar("eval/answer_reward", avg_answer, step)
    writer.add_scalar("eval/accuracy", accuracy, step)
    writer.add_scalar("eval/time", elapsed, step)


def main(args):
    os.environ.pop("RAY_ADDRESS", None)
    os.environ.pop("RAY_HEAD_IP", None)
    os.environ.pop("RAY_GCS_SERVER_ADDRESS", None)

    unique_dir = tempfile.mkdtemp(prefix=f"ray_temp_session_{int(time.time())}_")

    ray.init(
        address="local",
        include_dashboard=False,
        ignore_reinit_error=True,
        _temp_dir=unique_dir,
        dashboard_port=None
    )

    logging_dir = f"{args.experiment_dir}/countdown_nccl_static_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=logging_dir)

    # dump the args into a json file
    with open(f"{logging_dir}/args.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    model_saves_dir = f"{logging_dir}/model_saves"
    os.makedirs(model_saves_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    gc.collect()

    data_path = "countdown/data/countdown.json"
    with open(data_path, "r") as f:
        task_datas = json.load(f)
    task_datas = task_datas[:200]

    eval_task_datas = []
    if args.eval_interval > 0 and os.path.exists(args.eval_data_path):
        with open(args.eval_data_path, "r") as f:
            eval_task_datas = json.load(f)[200:]
        print(f"Loaded {len(eval_task_datas)} eval samples from {args.eval_data_path}")

        # pre-process eval contexts
        for d in eval_task_datas:
            d["context"] = _process_context(d, tokenizer)

    engines, pgs = launch_engines(args.num_engines, args.model_name, precision=args.precision)

    master_address = get_ip()
    master_port = get_open_port()
    ray.get([
        engines[i].collective_rpc.remote(
            "init_inter_engine_group", args=(master_address, master_port, i, args.num_engines)
        )
        for i in range(args.num_engines)
    ])

    def cleanup():
        for llm in engines:
            try: ray.kill(llm)
            except: pass
        for pg in pgs:
            try: remove_placement_group(pg)
            except: pass
        ray.shutdown()

    prompts = [_process_context(d, tokenizer) for d in task_datas]

    for i in range(args.num_iterations):
        print(f"\n\n=== Generation {i} (static scheduling) ===")
        total_iter_start = time.time()

        # Deterministic per-iteration seed list
        loop_rng = np.random.default_rng(seed=(args.global_seed or 42) + i)
        seeds = loop_rng.integers(0, 2**30, size=args.population_size, dtype=np.int64).tolist()

        seeds_perf: Dict[int, Dict[str, Any]] = {}

        # Static batching: issue exactly one seed per engine per batch in fixed order, then wait
        for b in range(0, len(seeds), args.num_engines):
            batch = seeds[b:b+args.num_engines]
            # 1) Perturb
            ray.get([
                engines[eng_idx].collective_rpc.remote("perturb_self_weights", args=(int(seed), args.sigma, False))
                for eng_idx, seed in enumerate(batch)
            ])

            # 2) Generate with fixed generation seed tied to iteration
            gen_seed = (args.global_seed or 42) + i
            handles = [
                evaluate_countdown(engines[eng_idx], prompts, seed=gen_seed)
                for eng_idx, _ in enumerate(batch)
            ]
            # 3) Collect outputs in the same order
            outputs_per_engine = ray.get(handles)
            # 4) Restore weights
            ray.get([
                engines[eng_idx].collective_rpc.remote("restore_self_weights", args=(int(seed), args.sigma))
                for eng_idx, seed in enumerate(batch)
            ])
            # 5) Score and record
            for eng_idx, seed in enumerate(batch):
                metrics = _postprocess_outputs(outputs_per_engine[eng_idx], task_datas)
                seeds_perf[int(seed)] = metrics

        # Aggregate
        all_avg_rewards = [v["avg_reward"] for v in seeds_perf.values()]
        mean_reward = float(np.mean(all_avg_rewards)) if all_avg_rewards else 0.0
        std_reward = float(np.std(all_avg_rewards)) if all_avg_rewards else 0.0
        min_reward = float(np.min(all_avg_rewards)) if all_avg_rewards else 0.0
        max_reward = float(np.max(all_avg_rewards)) if all_avg_rewards else 0.0

        # Also aggregate format and answer rewards and accuracy over seeds
        all_avg_formats = [v.get("avg_format", 0.0) for v in seeds_perf.values()]
        all_avg_answers = [v.get("avg_answer", 0.0) for v in seeds_perf.values()]
        all_accuracies = [v.get("accuracy", 0.0) for v in seeds_perf.values()]
        mean_format = float(np.mean(all_avg_formats)) if all_avg_formats else 0.0
        mean_answer = float(np.mean(all_avg_answers)) if all_avg_answers else 0.0
        mean_accuracy = float(np.mean(all_accuracies)) if all_accuracies else 0.0

        print(f"Mean reward: {mean_reward:.4f}, std: {std_reward:.4f}, format: {mean_format:.4f}, answer: {mean_answer:.4f}, acc: {mean_accuracy:.1f}%")
        for k in seeds_perf:
            seeds_perf[k]["norm_reward"] = (seeds_perf[k]["avg_reward"] - mean_reward) / (std_reward + 1e-8)

        writer.add_scalar("reward/mean", mean_reward, i)
        writer.add_scalar("reward/std", std_reward, i)
        writer.add_scalar("reward/min", min_reward, i)
        writer.add_scalar("reward/max", max_reward, i)
        # Training-time format and answer rewards + accuracy
        writer.add_scalar("train/format_reward", mean_format, i)
        writer.add_scalar("train/answer_reward", mean_answer, i)
        writer.add_scalar("train/accuracy", mean_accuracy, i)

        # Update weights on engine 0 using batched method
        coeffs = [float(seeds_perf[seed]["norm_reward"]) for seed in seeds]
        perturb_start = time.time()
        ray.get(engines[0].collective_rpc.remote(
            "update_weights_from_seeds",
            args=(seeds, coeffs, args.alpha, args.population_size)
        ))
        if args.verbose:
            print(f"Applied updates in {time.time() - perturb_start}s")

        # Broadcast from engine 0
        ray.get([e.collective_rpc.remote("broadcast_all_weights", args=(0,)) for e in engines])
        torch.cuda.synchronize()
        print(f"Iteration {i} finished in {time.time() - total_iter_start:.2f}s")

        if (args.eval_interval > 0 and (i % args.eval_interval == 0)) or (i == args.num_iterations - 1):
            evaluate_model(engines[0], eval_task_datas, writer, i, args)

    cleanup()

if __name__ == "__main__":
    args = parse_args()
    main(args)