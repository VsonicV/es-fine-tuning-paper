# Example training script: ES fine-tuning for math reasoning (MATH500).
#
# This file is intentionally task-specific. It wires together the math reward
# function, Qwen prompt template, and HuggingFace on-disk datasets to show a
# complete working configuration. It is NOT a general-purpose entry point.
#
# To use ES for your own task, copy this file and replace:
#   - reward_function  ->  your grader
#   - template_function -> your prompt formatter
#   - The DataLoader setup -> your data source
#
# The trainer itself (EvolutionStrategiesTrainer) is fully task-agnostic.

import argparse
from datetime import datetime
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from es_at_scale.trainer.es_trainer import EvolutionStrategiesTrainer

def collate_fn(batch):
    prompts = [item["problem"] for item in batch]
    answer = [item["answer"] for item in batch]
    return prompts, answer


def countdown_collate_fn(batch):
    prompts = [item["context"] for item in batch]
    targets = [
        {"numbers": item["numbers"], "target": item["target"]}
        for item in batch
    ]
    return prompts, targets


def set_seed(seed_value=42):
    random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)


def main():
    parser = argparse.ArgumentParser(
        description="Training script for ES finetuning experiments."
    )

    parser.add_argument("--task", type=str, default="countdown", choices=["math", "countdown"],
                        help="Selects reward function, prompt template, and dataset loader.")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--sigma", type=float, default=0.001)
    parser.add_argument("--alpha", type=float, default=-1)
    parser.add_argument("--reward-shaping", type=str, default="z-scores")
    parser.add_argument("--population-size", type=int, default=30)
    parser.add_argument("--n-iterations", type=int, default=300)
    parser.add_argument("--eval-freq", type=int, default=5)
    parser.add_argument(
        "--train-dataset", type=str, default="datasets/train/countdown"
    )

    parser.add_argument("--eval-dataset", type=str, default="datasets/evaluation_suite/countdown/")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--mini-batch-size", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--n-vllm-engines", type=int, default=8)
    parser.add_argument("--n-gpu-per-vllm-engine", type=int, default=1)
    parser.add_argument("--logging", type=str, default="wandb")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-gpus", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--reward-function-timeout", type=int, default=10)
    parser.add_argument("--output-directory", type=str, default="./experiments/")
    parser.add_argument("--save-best-models", type=bool, default=False)
    parser.add_argument("--experiment-name", type=str, default=None,
                        help="Experiment name for logging and checkpoints. Auto-generated from hyperparams if not set.")
    parser.add_argument("--wandb-project", type=str, default="es-finetuning")

    args = parser.parse_args()
    print(args)

    if args.task == "countdown":
        from es_at_scale.reward_function.countdown_grader import countdown_reward_fn
        from es_at_scale.template_function.apply_template import countdown_template
        reward_function = countdown_reward_fn
        template_function = countdown_template
        collate = countdown_collate_fn
    else:
        from es_at_scale.reward_function.math_grader import boxed_reward_fn
        from es_at_scale.template_function.apply_template import qwen_math_template
        reward_function = boxed_reward_fn
        template_function = qwen_math_template
        collate = collate_fn

    alpha = args.alpha
    if alpha == -1.0:
        alpha = args.sigma / 2

    set_seed(args.seed)

    # Train dataset loader
    for task_name, dataset in load_from_disk(args.train_dataset).items():
        train_dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            collate_fn=collate,
            shuffle=True,
        )

    # Cache eval prompts/targets in memory so we can launch eval cheaply from inside the ES loop.
    eval_dataloader_dict = {}
    if args.eval_dataset and os.path.exists(args.eval_dataset):
        for task_name, dataset in load_from_disk(args.eval_dataset).items():
            # Eval uses --mini-batch-size as its batch size. This is just a sensible
            # default, not a requirement: --mini-batch-size is meant to be the largest
            # batch that fits in GPU memory without triggering an OOM, so it is a safe
            # size for eval generation too. It does not affect the eval result -- the
            # trainer computes pass@1 as a size-weighted mean over batches, so any batch
            # size yields the same number. Override it here if you want to tune eval
            # throughput/memory independently of training.
            eval_dataloader_dict[task_name] = DataLoader(
                dataset,
                batch_size=args.mini_batch_size,
                shuffle=False,
                collate_fn=collate,
            )

    experiment_name = args.experiment_name or (
        f"es-{str(args.train_dataset).split("/")[-1]}"
        f"-sigma{args.sigma}-alpha{alpha}-pop{args.population_size}"
        f"-bs{args.batch_size}-tkn{args.max_tokens}-model{str(args.model_name).split("/")[-1]}"
        f"-seed{args.seed}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )

    print(f"-- Running: {experiment_name} --")

    trainer = EvolutionStrategiesTrainer(
        model_name=args.model_name,
        checkpoint=args.checkpoint,
        sigma=args.sigma,
        alpha=alpha,
        population_size=args.population_size,
        reward_shaping=args.reward_shaping,
        num_iterations=args.n_iterations,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        reward_function=reward_function,
        template_function=template_function,
        train_dataloader=train_dataloader,
        eval_dataloader_dict=eval_dataloader_dict,
        eval_freq=args.eval_freq,
        n_vllm_engines=args.n_vllm_engines,
        n_gpu_per_vllm_engine=args.n_gpu_per_vllm_engine,
        logging=args.logging,
        global_seed=args.seed,
        use_gpus=args.use_gpus,
        experiment_name=experiment_name,
        wandb_project=args.wandb_project,
        save_best_models=args.save_best_models,
        reward_function_timeout=args.reward_function_timeout,
        output_directory=args.output_directory
        

    )

    trainer.fit()


if __name__ == "__main__":
    main()
