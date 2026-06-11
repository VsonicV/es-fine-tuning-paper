import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import json
import os
import argparse
import time
from typing import List, Tuple, Dict, Any
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='inference script for ES models (Qwen/Llama/etc)')

    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to the saved model directory')
    parser.add_argument('--base_model_name', type=str, default=None,
                       help='Base model name for tokenizer loading (auto-detect from model_path if not specified)')
    parser.add_argument('--train_data_path', type=str, 
                       default='data/countdown/countdown.json',
                       help='Path to training data JSON file')
    parser.add_argument('--eval_data_path', type=str, 
                       default='data/countdown/countdown_samples.json',
                       help='Path to evaluation data JSON file')
    parser.add_argument('--eval_samples', type=int, default=100,
                       help='Number of evaluation samples to evaluate')
    parser.add_argument('--eval_offset', type=int, default=-100,
                       help='Offset for evaluation data (negative means from end)')
    parser.add_argument('--max_new_tokens', type=int, default=1024,
                       help='Maximum number of new tokens to generate')
    parser.add_argument('--do_sample', action='store_true',
                       help='Whether to use sampling instead of greedy decoding')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Temperature for sampling')
    parser.add_argument('--top_p', type=float, default=0.9,
                       help='Top-p for nucleus sampling')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size for inference (default: min(32, dataset_size))')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--torch_dtype', type=str, default='auto', 
                       choices=['auto', 'float16', 'bfloat16', 'float32'],
                       help='Torch dtype for model loading')
    parser.add_argument('--mixed_precision', type=str, default='fp16', 
                       choices=['no', 'fp16', 'bf16'],
                       help='Mixed precision mode (matching training)')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save inference results (default: based on model name)')
    parser.add_argument('--save_responses', action='store_true',
                       help='Save individual responses to file')
    parser.add_argument('--verbose', action='store_true',
                       help='Print verbose output')
    parser.add_argument('--show_examples', type=int, default=5,
                       help='Number of examples to show in detail')
    
    return parser.parse_args()

def detect_base_model(model_path: str) -> str:
    """Auto-detect base model from model_path"""
    model_path_lower = model_path.lower()
    
    if 'qwen' in model_path_lower:
        return 'Qwen/Qwen2.5-7B-Instruct'
    elif 'llama' in model_path_lower:
        return 'meta-llama/Llama-3.1-8B-Instruct'
    else:
        print(f"Warning: Could not auto-detect base model from path '{model_path}', defaulting to Qwen")
        return 'Qwen/Qwen2.5-7B-Instruct'

def load_data(data_path: str, num_samples: int = None, offset: int = 0) -> List[Tuple[str, str]]:
    """Load dataset from JSON file - exactly matching training script format"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    with open(data_path, 'r') as f:
        data_json = json.load(f)

    dataset = []
    for item in data_json:
        context = item['context']
        target = item['target']
        dataset.append((context, target))
    
    if offset < 0:
        # Take from the end
        start_idx = len(dataset) + offset
        end_idx = len(dataset)
    else:
        start_idx = offset
        end_idx = len(dataset)
    
    dataset = dataset[start_idx:end_idx]
    
    if num_samples is not None and num_samples < len(dataset):
        dataset = dataset[:num_samples]
    
    return dataset

def extract_model_response(generated_text: str) -> str:
    """Extract model response from generated text - exactly matching training script"""
    
    model_response = generated_text
    if "assistant:" in generated_text:
        model_response = generated_text.split("assistant:")[-1].strip()
    return model_response

def extract_numbers_and_target(input_text: str, target_text: str) -> Tuple[List[int], int]:
    """Extract numbers and target from input and target text - exactly matching training script"""
    numbers = None
    target = None
    
    if "[" in input_text and "]" in input_text:
        # Extract numbers from format like "[4 5 1 2]"
        start_idx = input_text.find("[")
        end_idx = input_text.find("]")
        if start_idx != -1 and end_idx != -1:
            numbers_str = input_text[start_idx+1:end_idx]
            numbers = [int(n) for n in numbers_str.split() if n.isdigit()]
    
    if target_text.isdigit():
        target = int(target_text)
    
    return numbers, target

def evaluate_batch(model, tokenizer, input_texts: List[str], target_texts: List[str], 
                  device, args, verbose: bool = False) -> List[Dict[str, Any]]:
    """Evaluate a batch of samples - exactly matching training script evaluation"""
    if verbose:
        print(f"Batch evaluating {len(input_texts)} samples...")
    
    tokenized_inputs = tokenizer(input_texts, return_tensors="pt", padding=True, padding_side="left")
    input_ids = tokenized_inputs["input_ids"].to(device)
    attention_mask = tokenized_inputs["attention_mask"].to(device)
    
    all_results = []
    with torch.inference_mode():
        outputs = model.generate(
            input_ids, 
            attention_mask=attention_mask, 
            max_new_tokens=args.max_new_tokens, 
            do_sample=args.do_sample
        )
        
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
    
    for i, output in enumerate(outputs):
        try:
            generated_text = tokenizer.decode(output, skip_special_tokens=True)
        except TypeError:
            tokens = tokenizer.convert_ids_to_tokens(output, skip_special_tokens=True)
            filtered = [t for t in tokens if t is not None]
            generated_text = tokenizer.convert_tokens_to_string(filtered)
        
        model_response = extract_model_response(generated_text)
        
        input_text = input_texts[i]
        target_text = target_texts[i]
        numbers, target = extract_numbers_and_target(input_text, target_text)
        
        reward_result = reward_function(model_response, numbers, target)
        reward = reward_result["reward"]
        reward_info = reward_result["reward_info"]
        
        all_results.append({
            'input_text': input_text,
            'target_text': target_text,
            'generated_text': generated_text,
            'model_response': model_response,
            'numbers': numbers,
            'target': target,
            'reward': reward,
            'reward_info': reward_info
        })
    
    del input_ids, outputs
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    
    return all_results

def evaluate_single_sample(model, tokenizer, input_text: str, target_text: str, 
                          device, args, verbose: bool = False) -> Dict[str, Any]:
    """Evaluate a single sample (fallback for compatibility)"""
    results = evaluate_batch(model, tokenizer, [input_text], [target_text], device, args, verbose)
    return results[0]

def evaluate_dataset(model, tokenizer, dataset: List[Tuple[str, str]], 
                    device, args, dataset_name: str, batch_size: int = None) -> Dict[str, Any]:
    """Evaluate model on a dataset using batch processing"""
    print(f"\n=== Evaluating on {dataset_name} dataset ({len(dataset)} samples) ===")
    
    # Set default batch size based on dataset size and memory considerations
    if batch_size is None:
        batch_size = min(32, len(dataset))  # Default batch size
    
    print(f"Using batch size: {batch_size}")
    
    all_results = []
    total_reward = 0.0
    total_format_reward = 0.0
    total_answer_reward = 0.0
    
    start_time = time.time()
    
    for batch_start in range(0, len(dataset), batch_size):
        batch_end = min(batch_start + batch_size, len(dataset))
        batch_dataset = dataset[batch_start:batch_end]
        
        if args.verbose:
            print(f"Processing batch {batch_start//batch_size + 1}/{(len(dataset)-1)//batch_size + 1} (samples {batch_start+1}-{batch_end})...")
        
        input_texts = [item[0] for item in batch_dataset]
        target_texts = [item[1] for item in batch_dataset]
        
        batch_results = evaluate_batch(model, tokenizer, input_texts, target_texts, 
                                     device, args, verbose=args.verbose)
        
        all_results.extend(batch_results)
        
        for result in batch_results:
            total_reward += result['reward']
            total_format_reward += result['reward_info']['format_reward']
            total_answer_reward += result['reward_info']['answer_reward']
        
        if batch_start == 0:
            for i, result in enumerate(batch_results[:args.show_examples]):
                print(f"\n--- Example {i+1} ---")
                print(f"Input: {result['input_text']}")
                print(f"Target: {result['target_text']}")
                print(f"Model Response: {result['model_response']}")
                print(f"Reward: {result['reward']:.4f} (Format: {result['reward_info']['format_reward']:.4f}, Answer: {result['reward_info']['answer_reward']:.4f})")
    
    eval_time = time.time() - start_time
    
    # Calculate statistics
    avg_reward = total_reward / len(dataset)
    avg_format_reward = total_format_reward / len(dataset)
    avg_answer_reward = total_answer_reward / len(dataset)
    
    rewards = [r['reward'] for r in all_results]
    std_reward = np.std(rewards)
    min_reward = np.min(rewards)
    max_reward = np.max(rewards)
    
    high_reward_count = sum(1 for r in rewards if r >= 1.0)
    high_reward_percentage = high_reward_count / len(dataset) * 100
    
    answer_rewards = [r['reward_info']['answer_reward'] for r in all_results]
    correct_count = sum(1 for r in answer_rewards if r > 0)
    accuracy = correct_count / len(dataset) * 100
    
    stats = {
        'dataset_name': dataset_name,
        'num_samples': len(dataset),
        'avg_reward': avg_reward,
        'avg_format_reward': avg_format_reward,
        'avg_answer_reward': avg_answer_reward,
        'std_reward': std_reward,
        'min_reward': min_reward,
        'max_reward': max_reward,
        'high_reward_count': high_reward_count,
        'high_reward_percentage': high_reward_percentage,
        'correct_count': correct_count,
        'accuracy': accuracy,
        'eval_time': eval_time,
        'all_results': all_results
    }
    
    # Print summary
    print(f"\n=== {dataset_name} Results Summary ===")
    print(f"Number of samples: {len(dataset)}")
    print(f"Average reward: {avg_reward:.4f} ± {std_reward:.4f}")
    print(f"  - Format reward: {avg_format_reward:.4f}")
    print(f"  - Answer reward: {avg_answer_reward:.4f}")
    print(f"Accuracy (answer_reward > 0): {correct_count}/{len(dataset)} ({accuracy:.1f}%)")
    print(f"High reward samples (≥1.0): {high_reward_count}/{len(dataset)} ({high_reward_percentage:.1f}%)")
    print(f"Reward range: [{min_reward:.4f}, {max_reward:.4f}]")
    print(f"Evaluation time: {eval_time:.2f}s ({eval_time/len(dataset):.3f}s per sample)")
    
    return stats

def save_results(results: Dict[str, Any], output_dir: str, args):
    """Save evaluation results to files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save summary statistics
    summary = {
        'model_path': args.model_path,
        'base_model_name': args.base_model_name,
        'eval_stats': {k: v for k, v in results['eval_stats'].items() if k != 'all_results'},
        'generation_config': {
            'max_new_tokens': args.max_new_tokens,
            'do_sample': args.do_sample,
            'temperature': args.temperature if args.do_sample else None,
            'top_p': args.top_p if args.do_sample else None,
            'batch_size': args.batch_size,
            'mixed_precision': args.mixed_precision,
        }
    }
    
    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")
    
    # Save detailed results if requested
    if args.save_responses:
        eval_details_path = os.path.join(output_dir, 'eval_detailed_results.json')
        
        with open(eval_details_path, 'w') as f:
            json.dump(results['eval_stats']['all_results'], f, indent=2)
        print(f"Eval detailed results saved to: {eval_details_path}")

def main():
    args = parse_args()

    # Auto-detect base model if not specified
    if args.base_model_name is None:
        args.base_model_name = detect_base_model(args.model_path)
        print(f"Auto-detected base model: {args.base_model_name}")

    global reward_function
    from countdown_task import reward_function
    
    # Set default output directory based on model name if not provided
    if args.output_dir is None:
        model_name = os.path.basename(args.model_path.rstrip('/'))
        batch_suffix = f"_batch{args.batch_size}" if args.batch_size else ""
        args.output_dir = f"./inference_results_{model_name}{batch_suffix}"
    
    print("=== Unified ES Model Inference Script ===")
    print(f"Model path: {args.model_path}")
    print(f"Base model: {args.base_model_name}")
    print(f"Eval data: {args.eval_data_path} (samples: {args.eval_samples}, offset: {args.eval_offset})")
    print(f"Output directory: {args.output_dir}")
    print(f"Mixed precision: {args.mixed_precision}")
    
    # Set device exactly as in training script
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Using device: {device}")
    
    if args.torch_dtype == 'auto':
        if args.mixed_precision == 'fp16':
            torch_dtype = torch.float16
        elif args.mixed_precision == 'bf16':
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float16 if device == 'cuda' else torch.float32
    else:
        torch_dtype = getattr(torch, args.torch_dtype)
    print(f"Using torch dtype: {torch_dtype}")
    
    # Load model and tokenizer with robust error handling
    print(f"\nLoading model from {args.model_path}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map="auto" if device == 'cuda' else None,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_path,
                use_fast=False,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Error loading tokenizer from {args.model_path}: {e}")
            print(f"Trying to load tokenizer from base model: {args.base_model_name}")
            tokenizer = AutoTokenizer.from_pretrained(
                args.base_model_name,
                use_fast=False,
                trust_remote_code=True
            )
    except Exception as e:
        print(f"Error loading model from {args.model_path}: {e}")
        print(f"Trying to load model and tokenizer from base model: {args.base_model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_name,
            device_map="auto" if device == 'cuda' else None,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model_name,
            use_fast=False,
            trust_remote_code=True
        )
    
    if not hasattr(tokenizer, 'pad_token'):
        print(f"Warning: Tokenizer does not have pad_token attribute. Type of tokenizer: {type(tokenizer)}")
        print("Attempting to fix by loading tokenizer from base model...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model_name,
            use_fast=False,
            trust_remote_code=True
        )
    
    if hasattr(tokenizer, 'pad_token') and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set tokenizer.pad_token = tokenizer.eos_token ({tokenizer.eos_token})")
    
    if device != 'cuda' and hasattr(model, 'to'):
        model = model.to(device)
    
    model.eval()
    print("Model loaded successfully!")
    
    # Load datasets exactly as in training script
    print(f"\nLoading datasets...")
    eval_dataset = load_data(
        os.path.join(os.path.dirname(__file__), args.eval_data_path),
        num_samples=args.eval_samples,
        offset=args.eval_offset
    )
    
    print(f"Loaded {len(eval_dataset)} evaluation samples")
    
    eval_stats = evaluate_dataset(model, tokenizer, eval_dataset, device, args, "Eval", batch_size=args.batch_size)
    
    results = {
        'eval_stats': eval_stats
    }
    save_results(results, args.output_dir, args)

if __name__ == "__main__":
    main()
