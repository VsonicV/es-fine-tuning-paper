import argparse
import copy
import gc
import json
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import numpy as np
import torch
import torch.multiprocessing as mp
import wandb
from accelerate import Accelerator
# === LoRA / PEFT ===
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging

logging.set_verbosity_error()
torch.backends.cuda.matmul.allow_tf32 = True

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-3B-Instruct')
parser.add_argument('--hf_cache_dir', type=str, default='hf_cache')
parser.add_argument('--precision', type=str, default='bf16')
parser.add_argument('--gpu_threads', type=int, default=4, help='Number of parallel threads per GPU')
parser.add_argument('--verbose', action='store_true', help='Print verbose logs')
parser.add_argument('--data_sample', type=int, default=1000, help='Number of data samples to use for training')
parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')
parser.add_argument('--wandb_project', type=str, default='es-fine-tuning-countdown', help='Wandb project name')
parser.add_argument('--wandb_run_name', type=str, default=None, help='Wandb run name (default: auto-generated)')
parser.add_argument('--track_flops', action='store_true', help='Track FLOPs for forward passes')
parser.add_argument('--use_lora', action='store_true', default=True, help='Use LoRA for fine-tuning (default: True)')
args = parser.parse_args()

# Hyperparameters for ES
NUM_ITERATIONS = 1000             # Number of ES iterations (generations)
POPULATION_SIZE = 30              # Population size (number of perturbations per iteration)
SIGMA = 0.001                     # Standard deviation for weight perturbations (noise scale)
ALPHA = 0.0005                    # Learning rate
max_new_tokens = 1024             # Maximum number of tokens allowed to be generated
do_sample = False                 # Greedy decoding for ES
initial_seed = 33                 # Initial random seed

# === LoRA hyperparams ===
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
# Qwen/LLaMA-style module names
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Import countdown reward function
from countdown_task import reward_function

print("Using countdown reward function")

# Dataset will be loaded in main function

def force_memory_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()

def estimate_generation_flops(num_params, input_length, output_length, batch_size=1):
    """
    Estimate FLOPs for autoregressive generation.
    
    For transformer models:
    - Each forward pass through a token position ≈ 2 * num_params FLOPs
    - For generation: we do (input_length + output_length) forward passes
    - Each subsequent token requires attending to all previous tokens
    
    Uses closed-form formula for efficiency.
    """
    # Prefill: input_length tokens in parallel
    prefill_flops = 2 * num_params * input_length * batch_size
    
    # Decode: output_length tokens sequentially, each attending to growing context
    # Sum of (input_length + i) for i in range(output_length)
    # = output_length * input_length + sum(0 to output_length-1)
    # = output_length * input_length + output_length * (output_length - 1) / 2
    decode_flops = 2 * num_params * batch_size * (
        output_length * input_length + output_length * (output_length - 1) // 2
    )
    
    total_flops = prefill_flops + decode_flops
    return total_flops

def save_model_checkpoint(model, tokenizer, iteration, model_name, initial_seed, args, dataset_size):
    """Save LoRA adapter checkpoint at specified iteration"""
    question_num = dataset_size
    save_dir = f"{model_name}_es_random_seed{initial_seed}_pop{POPULATION_SIZE}_iter{iteration}_sigma{SIGMA}_alpha{ALPHA}_{args.precision}_threads{args.gpu_threads}_question_num{question_num}_checkpoint"
    print(f"Saving checkpoint at iteration {iteration} to {save_dir}...")
    # For PeftModel this saves adapters only (desired)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Checkpoint saved successfully.")

def evaluate_model(model, tokenizer, input_text, target_text, accelerator, seed_idx=None, thread_id=None, verbose=False, return_text=False, track_flops=False, num_params=None):
    """
    Generate a response from the model given an input (single or batch) and compute rewards.
    """
    if verbose:
        print(f"Process {accelerator.process_index} Thread {thread_id} evaluating seed {seed_idx}")

    # Handle both single input and batch input
    is_batch = isinstance(input_text, list)
    input_texts = input_text if is_batch else [input_text]
    target_texts = target_text if is_batch else [target_text]

    # Batch tokenization (left pad set globally on tokenizer)
    tokenized_inputs = tokenizer(
        input_texts,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    input_ids = tokenized_inputs["input_ids"].to(accelerator.device)
    attention_mask = tokenized_inputs["attention_mask"].to(accelerator.device)
    
    input_length = input_ids.shape[1]
    batch_size = input_ids.shape[0]
    
    with torch.inference_mode():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize(accelerator.device)

    # Calculate FLOPs if requested
    flops = 0
    if track_flops and num_params is not None:
        output_length = outputs.shape[1] - input_length
        flops = estimate_generation_flops(num_params, input_length, output_length, batch_size)

    # Decode batch outputs
    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    del input_ids, outputs, attention_mask, tokenized_inputs
    torch.cuda.empty_cache()

    # Compute rewards for batch texts
    rewards = []
    for i, (gen_text, tgt_text, inp_text) in enumerate(zip(generated_texts, target_texts, input_texts)):
        numbers = None
        target = None
        if "[" in inp_text and "]" in inp_text:
            start_idx = inp_text.find("[")
            end_idx = inp_text.find("]")
            if start_idx != -1 and end_idx != -1:
                numbers_str = inp_text[start_idx+1:end_idx]
                numbers = [int(n) for n in numbers_str.split() if n.isdigit()]

        if isinstance(tgt_text, str) and tgt_text.isdigit():
            target = int(tgt_text)
        elif isinstance(tgt_text, int):
            target = tgt_text

        model_response = gen_text
        if "assistant:" in gen_text:
            model_response = gen_text.split("assistant:")[-1].strip()

        # Use reward_function from countdown_task.py
        reward_result = reward_function(model_response, numbers, target)
        reward = reward_result["reward"]
        rewards.append(reward)

    if return_text:
        if track_flops:
            return rewards, generated_texts, flops
        else:
            return rewards, generated_texts
    else:
        if track_flops:
            return rewards, flops
        else:
            return rewards

def process_seed(seed_args):
    """Function to process a single seed, used for thread pool"""
    seed_idx, seed, model, tokenizer, accelerator, thread_id, verbose, dataset, track_flops, num_params = seed_args

    if verbose:
        print(f"Process {accelerator.process_index} Thread {thread_id} processing seed {seed_idx} (value: {seed})")

    # Apply noise to trainable params (LoRA params if using LoRA, or all params if not)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        gen = torch.Generator(device=param.device)
        gen.manual_seed(int(seed))
        noise = torch.randn(
            param.shape,
            generator=gen,
            device=param.device,
            dtype=param.dtype
        )
        param.data.add_(SIGMA * noise)

    # Ensure weights are fully loaded before evaluation
    if torch.cuda.is_available():
        torch.cuda.synchronize(accelerator.device)

    # Evaluate all prompts with perturbed weights in batch
    input_texts = [input_text for input_text, _ in dataset]
    target_texts = [target_text for _, target_text in dataset]
    eval_result = evaluate_model(
        model, tokenizer, input_texts, target_texts, accelerator,
        seed_idx=seed_idx, thread_id=thread_id, verbose=verbose, return_text=False, track_flops=track_flops, num_params=num_params
    )
    
    if track_flops:
        rewards, flops = eval_result
    else:
        rewards = eval_result
        flops = 0
    
    total_reward = sum(rewards)

    # Restore original weights (subtract the same noise) for trainable params
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        gen = torch.Generator(device=param.device)
        gen.manual_seed(int(seed))
        noise = torch.randn(
            param.shape,
            generator=gen,
            device=param.device,
            dtype=param.dtype
        )
        param.data.add_(-SIGMA * noise)

    if torch.cuda.is_available():
        torch.cuda.synchronize(accelerator.device)

    average_reward = total_reward / len(dataset)

    force_memory_cleanup()

    if verbose:
        print(f"Process {accelerator.process_index} Thread {thread_id} completed seed {seed_idx} with reward {average_reward:.4f}")

    if track_flops:
        return seed_idx, average_reward, flops
    else:
        return seed_idx, average_reward

# --- Main Evolution Strategies Loop ---
def main():
    accelerator = Accelerator()

    # --- Load Dataset from JSON File ---
    data_path = os.path.join(os.path.dirname(__file__), 'data/countdown.json')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")
    
    with open(data_path, 'r') as f:
        data_json = json.load(f)

    dataset = []
    for item in data_json:
        context = item['context']
        target = item['target']
        dataset.append((context, target))

    # Use a subset of the dataset for training
    dataset = dataset[:args.data_sample]
    if accelerator.is_main_process:
        print(f"Loaded {len(dataset)} countdown samples from {data_path}")

    # Define model parameters early for wandb
    model_name = args.model_name
    hf_cache_dir = args.hf_cache_dir

    if accelerator.is_main_process:
        print(f"Total processes: {accelerator.num_processes}, GPU threads per process: {args.gpu_threads}")
        print(f"Population size: {POPULATION_SIZE}, Iterations: {NUM_ITERATIONS}")
        print(f"Sigma: {SIGMA}, Alpha: {ALPHA}")
        
        # Initialize wandb
        if args.use_wandb:
            lora_suffix = "_lora" if args.use_lora else "_full"
            wandb_run_name = args.wandb_run_name or f"es_{model_name.replace('/', '_')}_pop{POPULATION_SIZE}_sigma{SIGMA}_alpha{ALPHA}_seed{initial_seed}{lora_suffix}"
            config_dict = {
                "model_name": model_name,
                "population_size": POPULATION_SIZE,
                "num_iterations": NUM_ITERATIONS,
                "sigma": SIGMA,
                "alpha": ALPHA,
                "max_new_tokens": max_new_tokens,
                "initial_seed": initial_seed,
                "use_lora": args.use_lora,
                "precision": args.precision,
                "gpu_threads": args.gpu_threads,
                "num_gpus": accelerator.num_processes,
                "data_samples": len(dataset),
            }
            if args.use_lora:
                config_dict.update({
                    "lora_r": LORA_R,
                    "lora_alpha": LORA_ALPHA,
                    "lora_dropout": LORA_DROPOUT,
                })
            wandb.init(
                project=args.wandb_project,
                name=wandb_run_name,
                config=config_dict
            )
            print(f"Wandb initialized: {wandb.run.url}")

    # Load model

    if accelerator.is_main_process:
        print(f"Loading model {model_name}...")

    # Load tokenizer once and set padding
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, cache_dir=hf_cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # === Build LoRA config once (if using LoRA) ===
    lora_cfg = None
    if args.use_lora:
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGET_MODULES,
            bias="none",
            inference_mode=False,
        )

    # Load base model per thread, optionally wrap with LoRA, eval mode for deterministic ES
    model_list = []
    for model_index in range(args.gpu_threads):
        base = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=hf_cache_dir,
            device_map={"": accelerator.process_index},  # Assign devices explicitly
            torch_dtype=torch.float16 if args.precision == 'fp16' else (torch.bfloat16 if args.precision == 'bf16' else torch.float32),
        )
        
        if args.use_lora:
            model = get_peft_model(base, lora_cfg)  # wrap with LoRA
        else:
            model = base  # use full model
            
        model.eval()
        model_list.append(model)

    if accelerator.is_main_process:
        print("Model loaded successfully")
        if args.use_lora:
            print(f"Using LoRA with r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
        else:
            print("Using full model (no LoRA)")

    force_memory_cleanup()

    # Record total training start time
    training_start_time = time.time()
    
    # Initialize FLOPs tracking
    total_flops = 0
    # Cache parameter count for FLOPs calculation (compute once, reuse)
    num_params = sum(p.numel() for p in model_list[0].parameters()) if args.track_flops else None
    if args.track_flops and accelerator.is_main_process:
        print(f"Model has {num_params/1e9:.2f}B parameters")

    np.random.seed(initial_seed)

    # =====================================================================
    # === REPLACED ES LOOP: mini-batch + batched generation per seed ======
    # =====================================================================
    original_model = model_list[0]

    for iteration in range(NUM_ITERATIONS):
        iter_start_time = time.time()

        # Choose a mini-batch of questions (same across processes)
        BATCH_SIZE = min(32, len(dataset))  # mini-batch size (kept local to this loop change)
        if accelerator.is_main_process:
            batch_indices = torch.from_numpy(
                np.random.choice(len(dataset), size=BATCH_SIZE, replace=False)
            ).to(device=accelerator.device, dtype=torch.long)
        else:
            batch_indices = torch.zeros(BATCH_SIZE, device=accelerator.device, dtype=torch.long)

        # Broadcast batch indices so all processes evaluate the same questions
        if accelerator.num_processes > 1:
            torch.distributed.broadcast(batch_indices, src=0)

        batch_indices_list = batch_indices.cpu().tolist()
        batch_texts   = [dataset[i][0] for i in batch_indices_list]
        batch_targets = [dataset[i][1] for i in batch_indices_list]

        # Generate seeds on main process and broadcast (as in original)
        if accelerator.is_main_process:
            seeds = np.random.randint(0, 2**30, size=POPULATION_SIZE, dtype=np.int64).tolist()
            seeds_tensor = torch.tensor(seeds, device=accelerator.device)
        else:
            seeds_tensor = torch.zeros(POPULATION_SIZE, dtype=torch.long, device=accelerator.device)
        if accelerator.num_processes > 1:
            torch.distributed.broadcast(seeds_tensor, src=0)
        seeds = seeds_tensor.cpu().tolist()

        # Assign seeds to each process (same policy as original)
        local_seeds = []
        for seed_idx, seed in enumerate(seeds):
            if seed_idx % accelerator.num_processes == accelerator.process_index:
                local_seeds.append((seed_idx, seed))

        # Prepare reward storage: [BATCH_SIZE, POPULATION_SIZE]
        rewards_matrix_local = torch.zeros(BATCH_SIZE, POPULATION_SIZE, device=accelerator.device, dtype=torch.float32)

        # FLOPs tracking
        iter_flops_local = 0.0

        # Evaluate each local seed on the *batched* mini-batch
        for (seed_idx, seed) in local_seeds:
            # Apply noise to trainable params
            for name, param in original_model.named_parameters():
                if not param.requires_grad:
                    continue
                gen = torch.Generator(device=param.device)
                gen.manual_seed(int(seed))
                noise = torch.randn(param.shape, generator=gen, device=param.device, dtype=param.dtype)
                param.data.add_(SIGMA * noise)

            # Batched evaluation (single generate() for all B prompts)
            eval_result = evaluate_model(
                original_model, tokenizer, batch_texts, batch_targets, accelerator,
                seed_idx=seed_idx, thread_id=0, verbose=args.verbose, return_text=False,
                track_flops=args.track_flops, num_params=num_params
            )
            if args.track_flops:
                rewards_list, flops = eval_result
                iter_flops_local += float(flops)
            else:
                rewards_list = eval_result

            # Fill local rewards column for this seed
            rewards_tensor_seed = torch.tensor(rewards_list, device=accelerator.device, dtype=torch.float32)
            rewards_matrix_local[:, seed_idx] = rewards_tensor_seed

            # Restore original weights (subtract same noise)
            for name, param in original_model.named_parameters():
                if not param.requires_grad:
                    continue
                gen = torch.Generator(device=param.device)
                gen.manual_seed(int(seed))
                noise = torch.randn(param.shape, generator=gen, device=param.device, dtype=param.dtype)
                param.data.add_(-SIGMA * noise)

        # Aggregate rewards across processes into full [B, N]
        if accelerator.num_processes > 1:
            torch.distributed.all_reduce(rewards_matrix_local, op=torch.distributed.ReduceOp.SUM)
        rewards_matrix = rewards_matrix_local  # now holds global matrix since each seed is unique to one proc

        # Z-score per question across seeds
        means = rewards_matrix.mean(dim=1, keepdim=True)
        stds  = rewards_matrix.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-8)
        z = (rewards_matrix - means) / stds  # [B, N]

        # Average z across questions -> per-seed weight
        z_avg = z.mean(dim=0)  # [N]

        # ES parameter update using z_avg weights (regenerate noise once)
        for name, param in original_model.named_parameters():
            if not param.requires_grad:
                continue
            gen = torch.Generator(device=param.device)
            update = torch.zeros_like(param)
            for seed_idx, seed in enumerate(seeds):
                gen.manual_seed(int(seed))
                noise = torch.randn(param.shape, generator=gen, device=param.device, dtype=param.dtype)
                update.add_(noise.mul(float(z_avg[seed_idx].item())))
            update.div_(POPULATION_SIZE)
            param.data.add_(ALPHA * update)
            del update
            torch.cuda.empty_cache()

        # Sync updated trainable params to sibling models (keep original behavior)
        for model_idx in range(1, len(model_list)):
            dst = model_list[model_idx]
            src_named = dict(original_model.named_parameters())
            for n_dst, p_dst in dst.named_parameters():
                if p_dst.requires_grad:
                    p_dst.data.copy_(src_named[n_dst].data)

        # FLOPs aggregation
        iter_flops = 0.0
        if args.track_flops:
            flops_tensor = torch.tensor(iter_flops_local, device=accelerator.device, dtype=torch.float64)
            if accelerator.num_processes > 1:
                torch.distributed.all_reduce(flops_tensor, op=torch.distributed.ReduceOp.SUM)
            iter_flops = float(flops_tensor.item())
            total_flops += iter_flops

        # Metrics over seeds (use per-seed mean reward across questions)
        per_seed_means = rewards_matrix.mean(dim=0)  # [N]
        mean_reward = float(per_seed_means.mean().item())
        min_reward  = float(per_seed_means.min().item())
        max_reward  = float(per_seed_means.max().item())
        std_reward  = float(per_seed_means.std(unbiased=False).item())

        iter_time = time.time() - iter_start_time
        if accelerator.is_main_process:
            flops_str = f", FLOPs: {iter_flops/1e12:.2f}T" if args.track_flops else ""
            print(f"Iteration {iteration + 1}/{NUM_ITERATIONS}, Time: {iter_time:.2f}s, Mean: {mean_reward:.2f}, Min: {min_reward:.2f}, Max: {max_reward:.2f}{flops_str}")

            if args.use_wandb:
                log_dict = {
                    "iteration": iteration + 1,
                    "reward/mean": mean_reward,
                    "reward/min": min_reward,
                    "reward/max": max_reward,
                    "reward/std": std_reward,
                    "time/iteration_seconds": iter_time,
                    "batch/size": BATCH_SIZE,
                }
                if args.track_flops:
                    log_dict["compute/iteration_tflops"] = iter_flops / 1e12
                    log_dict["compute/cumulative_tflops"] = total_flops / 1e12
                    log_dict["compute/tflops_per_second"] = (iter_flops / 1e12) / iter_time
                if torch.cuda.is_available():
                    log_dict["memory/gpu_allocated_mb"] = torch.cuda.memory_allocated() / 1024**2
                    log_dict["memory/gpu_peak_mb"] = torch.cuda.max_memory_allocated() / 1024**2
                wandb.log(log_dict)

            if torch.cuda.is_available():
                print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.2f}MB allocated, {torch.cuda.max_memory_allocated() / 1024**2:.2f}MB peak")

            # Save checkpoint every 100 iterations (adapters only)
            if (iteration + 1) % 100 == 0:
                save_model_checkpoint(original_model, tokenizer, iteration + 1, model_name, initial_seed, args, len(dataset))

        force_memory_cleanup()
    # =====================================================================

    total_time = time.time() - training_start_time

    # Save the final model
    if accelerator.is_main_process:
        print(f"Training completed in {total_time:.2f}s ({total_time/60:.2f} minutes)")
        if args.track_flops:
            print(f"Total FLOPs: {total_flops/1e12:.2f} TFLOPs")
            print(f"Average TFLOPs/s: {(total_flops/1e12)/total_time:.2f}")
        
        if args.use_wandb:
            log_dict = {
                "time/total_training_seconds": total_time, 
                "time/total_training_minutes": total_time/60
            }
            if args.track_flops:
                log_dict["compute/total_tflops"] = total_flops / 1e12
                log_dict["compute/average_tflops_per_second"] = (total_flops / 1e12) / total_time
            wandb.log(log_dict)
        
        question_num = len(dataset)
        save_dir = f"{model_name}_es_random_seed{initial_seed}_pop{POPULATION_SIZE}_iter{NUM_ITERATIONS}_sigma{SIGMA}_alpha{ALPHA}_{args.precision}_threads{args.gpu_threads}_question_num{question_num}_final"
        print(f"Saving final model to {save_dir}...")
        # Save model based on whether LoRA was used
        if args.use_lora:
            # Try to save merged full model; fallback to adapters-only
            try:
                merged = original_model.merge_and_unload()  # returns base model with LoRA merged
                merged.save_pretrained(save_dir)
            except Exception as e:
                print(f"merge_and_unload failed ({e}); saving adapters only.")
                original_model.save_pretrained(save_dir)
        else:
            # Save full model directly
            original_model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"Final model saved successfully.")
        
        if args.use_wandb:
            wandb.finish()

if __name__ == "__main__":
    os.environ["PYTHONWARNINGS"] = "ignore"
    mp.set_start_method('spawn', force=True)
    main()