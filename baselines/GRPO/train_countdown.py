import html
import time
import json
import os
import sys
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard.writer import SummaryWriter

from countdown_task import reward_function
from data_types import MiniBatch
from grpo import rollout, update_policy
from optimizer import MemoryEfficientAdamW
from qwen2_model import Transformer
from tokenizer import Tokenizer

from transformers import AutoModelForCausalLM, AutoTokenizer

class JSONCountdownTasksDataset(Dataset):
    """Dataset loading from JSON file similar to main_v3_new.py"""
    
    def __init__(
        self,
        tokenizer: Tokenizer,
        data_path: str,
        split: str = "train",
        test_size: int = 3,
        max_samples: int = 200,  # Limit to first 200 samples by default
    ):
        # Load data from the JSON file
        with open(data_path, 'r') as f:
            data_json = json.load(f)
            
        # Convert the JSON data to the format used in the original dataset
        dataset = []
        for item in data_json:
            context = item['context']
            target = item['target']
            dataset.append((context, target))
        
        # Limit dataset to first max_samples for training
        if split == "train":
            self.data = dataset[:-test_size]
            # If max_samples is -1, use all available samples
            if max_samples > 0 and max_samples < len(self.data):
                self.data = self.data[:max_samples]
            elif max_samples == -1:
                print(f"Using all available training samples: {len(self.data)}")
            print(f"Training on samples 0 to {len(self.data)-1} (total: {len(self.data)} samples)")
        else:  # test
            self.data = dataset[-test_size:]
            print(f"Testing on last {test_size} samples")
            
        self.tokenizer = tokenizer
        print(f"{split} dataset size: {len(self.data)}")
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        # Implement repeat sampling by taking modulo of idx
        idx = idx % len(self.data)  # This ensures we cycle through the data if needed
        
        context, target = self.data[idx]
        
        # Extract numbers from format like "[4 5 1 2]"
        numbers = None
        if "[" in context and "]" in context:
            start_idx = context.find("[")
            end_idx = context.find("]")
            if start_idx != -1 and end_idx != -1:
                numbers_str = context[start_idx+1:end_idx]
                numbers = [int(n) for n in numbers_str.split() if n.isdigit()]
        
        # Use target directly as the target value
        target_value = int(target) if target.isdigit() else 0
        
        # Encode the prefix
        prefix = self.encode_prefix(context)
        
        return {
            "nums": numbers,
            "target": target_value,
            "prefix": prefix["prefix"],
            "prefix_tokens": prefix["prefix_tokens"],
            "prefix_token_ids": prefix["prefix_token_ids"],
        }
    
    def encode_prefix(self, context):
        """Encode the prefix for the model."""
        tokens = self.tokenizer.tokenize(context)
        return {
            "prefix": context,
            "prefix_tokens": tokens.tokens,
            "prefix_token_ids": tokens.ids,
        }
    
    @staticmethod
    def collate_fn(batch):
        """Collate examples into a batch."""
        numbers = [item["nums"] for item in batch]
        target = [item["target"] for item in batch]
        prefix = [item["prefix"] for item in batch]
        prefix_tokens = [item["prefix_tokens"] for item in batch]
        prefix_token_ids = [item["prefix_token_ids"] for item in batch]
        return MiniBatch(
            numbers=numbers,
            target=target,
            prefix=prefix,
            prefix_tokens=prefix_tokens,
            prefix_token_ids=prefix_token_ids,
        )


# HuggingFace model wrapper class to provide the same interface as Qwen2 model
class HFModelWrapper:
    def __init__(self, model, tokenizer):
        self.model = model
        self.hf_tokenizer = tokenizer
        self.pad_token_id = self.hf_tokenizer.pad_token_id
        
    def train(self):
        self.model.train()
        return self
        
    def parameters(self):
        return self.model.parameters()
        
    def forward(self, tokens):
        # Adapt HF model interface to match Qwen2 model interface
        outputs = self.model(input_ids=tokens, return_dict=True)
        return outputs.logits
        
    def inference(self, tokens, start_pos):
        # For HF models, we need to handle KV cache differently
        # Check if model name contains 'llama' to use special handling
        is_llama_model = hasattr(self.model.config, 'model_type') and 'llama' in self.model.config.model_type.lower()
        
        # LLaMA models need special attention to KV cache handling
        if is_llama_model:
            # For LLaMA models, we need to be more careful with the KV cache
            if start_pos == 0:
                # First call, no past_key_values
                outputs = self.model(
                    input_ids=tokens,
                    use_cache=True,
                    past_key_values=None
                )
                self.model.past_key_values = outputs.past_key_values
            else:
                # For subsequent calls, use the stored KV cache
                outputs = self.model(
                    input_ids=tokens,
                    use_cache=True,
                    past_key_values=self.model.past_key_values
                )
                # Update the KV cache
                self.model.past_key_values = outputs.past_key_values
        else:
            # For other models, use the original implementation
            outputs = self.model(
                input_ids=tokens,
                use_cache=True,
                past_key_values=None if start_pos == 0 else self.model.past_key_values
            )
            self.model.past_key_values = outputs.past_key_values
            
        return outputs.logits
        
    def init_kv_cache(self, max_batch_size, max_seq_len, device, dtype):
        # Check if model is LLaMA to handle KV cache initialization differently
        is_llama_model = hasattr(self.model.config, 'model_type') and 'llama' in self.model.config.model_type.lower()
        
        if is_llama_model:
            # For LLaMA models, explicitly initialize past_key_values to None
            self.model.past_key_values = None
        else:
            # Other HF models handle their own KV cache
            pass
        
    def del_kv_cache(self):
        # Ensure KV cache is properly cleared for all model types
        if hasattr(self.model, 'past_key_values'):
            self.model.past_key_values = None
            
        # For LLaMA models, ensure we also clear any other caches
        is_llama_model = hasattr(self.model.config, 'model_type') and 'llama' in self.model.config.model_type.lower()
        if is_llama_model:
            # Force garbage collection to ensure memory is freed
            import gc
            gc.collect()
            torch.cuda.empty_cache()


# HuggingFace tokenizer wrapper to provide the same interface as Qwen2 tokenizer
class HFTokenizerWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        # Add eos_token attribute for compatibility
        self.eos_token = tokenizer.eos_token
        
    def encode_chat(self, messages):
        """Apply chat template and return formatted string"""
        # Check if the model is a LLaMA model to apply special handling
        is_llama_model = False
        if hasattr(self.tokenizer, 'name_or_path'):
            is_llama_model = 'llama' in self.tokenizer.name_or_path.lower()
        
        # Apply chat template with appropriate parameters
        if is_llama_model:
            # LLaMA models need special handling for their chat template
            return self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            # For other models like Qwen, use standard parameters
            return self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        
    def encode_chat_with_response_prompt(self, messages, prompt):
        """Encode chat with additional response prompt"""
        return self.encode_chat(messages) + prompt
        
    def tokenize(self, text):
        # Create a compatible encoding object
        encoding = self.tokenizer(text, return_tensors="pt")
        tokens = self.tokenizer.convert_ids_to_tokens(encoding.input_ids[0])
        ids = encoding.input_ids[0].tolist()
        
        # Create a compatible Encoding object with tokens and ids attributes
        class CompatEncoding:
            def __init__(self, tokens, ids):
                self.tokens = tokens
                self.ids = ids
                
        return CompatEncoding(tokens, ids)
        
    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)


def evaluate(model, tokenizer, device, dtype, config, max_samples=200):
    test_data_path = config["data"].get("test_path", 'data/countdown_samples.json')
    print(f"Loading test data from: {test_data_path}")
    test_dataset = JSONCountdownTasksDataset(
        data_path=test_data_path,
        tokenizer=tokenizer,
        split="test",
        test_size=config["data"]["test_size"],
        max_samples=max_samples,
    )
    generator = torch.Generator(device=device)
    # We reduce the batch size by half as we want to
    # generate twice as long trajectories.
    print("batch_size:", config["training"]["batch_size"])
    dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        collate_fn=JSONCountdownTasksDataset.collate_fn,
        generator=generator,
        batch_size=config["training"]["batch_size"] // 2,
        drop_last=False,
    )
    success = []
    for batch in dataloader:
        episodes = rollout(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            max_gen_len=config["training"]["max_gen_len"] * 2,
            num_answer_per_question=1,
            reward_function=reward_function,
            device=device,
            dtype=dtype,
        )
        success.extend([episode.reward_info["answer_reward"] for episode in episodes])
    return np.mean(success)


# Helper function to create an infinite dataloader
def infinite_dataloader(dataloader):
    while True:
        for batch in dataloader:
            yield batch


def main(config_path: str, model_name: str = None, max_samples: int = 200):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Override model path if specified in command line
    if model_name:
        config["model"]["pretrained_model_path"] = model_name
        print(f"Using model: {model_name}")
    else:
        print(f"Using default model: {config['model']['pretrained_model_path']}")

    pretrained_model_path = Path(config["model"]["pretrained_model_path"])
    device = torch.device(config["model"]["device"])
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(config["model"]["dtype"], torch.bfloat16)
    torch.set_default_device(device)
    torch.random.manual_seed(config["training"]["random_seed"])
    BATCH_SIZE = config["training"]["batch_size"]
    NUM_QUESTIONS_PER_BATCH = config["training"]["num_questions_per_batch"]
    NUM_ANSWERS_PER_QUESTION = BATCH_SIZE // NUM_QUESTIONS_PER_BATCH

    current_time = datetime.now().strftime(r"%Y%m%d-%H%M%S")
    tb_writer = SummaryWriter(log_dir=f"{config['training']['log_dir']}/{current_time}")
    
    # Check if model is from HuggingFace or local Qwen
    is_hf_model = (str(pretrained_model_path).startswith("meta-llama/") or 
                   str(pretrained_model_path).startswith("Qwen/") or
                   str(pretrained_model_path).startswith("Qwen2") or
                   str(pretrained_model_path).startswith("mistralai/"))
    
    if is_hf_model:
        if not 'transformers' in sys.modules:
            raise ImportError("Cannot load HuggingFace model: transformers package is not available")
            
        print(f"Loading HuggingFace model: {pretrained_model_path}")
        try:
            hf_model = AutoModelForCausalLM.from_pretrained(
                str(pretrained_model_path),
                torch_dtype=dtype,
                device_map=device,
                trust_remote_code=True
            )
            hf_tokenizer = AutoTokenizer.from_pretrained(
                str(pretrained_model_path),
                trust_remote_code=True
            )
            
            # Make sure padding and EOS tokens are set
            if hf_tokenizer.pad_token is None:
                if hf_tokenizer.eos_token is not None:
                    hf_tokenizer.pad_token = hf_tokenizer.eos_token
                else:
                    raise ValueError("Model tokenizer has no pad_token or eos_token")
            
            # Create wrapper objects
            model = HFModelWrapper(hf_model, hf_tokenizer)
            tokenizer = HFTokenizerWrapper(hf_tokenizer)
            
        except Exception as e:
            print(f"Error loading HuggingFace model: {e}")
            raise
    else:
        # Use original Qwen model loading
        print(f"Loading local Qwen model from: {pretrained_model_path}")
        tokenizer = Tokenizer(str(pretrained_model_path / "tokenizer.json"))
        model = Transformer.from_pretrained(pretrained_model_path, device=device).train()

    # Use the new dataset class that loads from JSON
    train_data_path = config["data"].get("train_path", 'data/countdown_samples.json')
    print(f"Loading training data from: {train_data_path}")
    train_dataset = JSONCountdownTasksDataset(
        data_path=train_data_path,
        tokenizer=tokenizer,
        split="train",
        test_size=config["data"]["test_size"],
        max_samples=max_samples,  # Use the command-line parameter here
    )
    generator = torch.Generator(device=device)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=JSONCountdownTasksDataset.collate_fn,
        generator=generator,
        batch_size=NUM_QUESTIONS_PER_BATCH,
    )

    # Print dataset contents for inspection
    print("\ndataset contents:")
    for i in range(min(5, len(train_dataset))):
        item = train_dataset[i]
        print(f"\nsample {i+1}:")
        print(f"input text: {item['prefix']}")
        print(f"numbers: {item['nums']}")
        print(f"target: {item['target']}")
        if i >= 4:  # Only print first 5 samples
            print("\n... (remaining samples omitted)")
            break
    print(f"\ntotal samples: {len(train_dataset)}\n")

    optimizer = MemoryEfficientAdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        betas=config["training"]["betas"],
        enabled=config["training"]["memory_efficient_adamw"],
    )

    start_time = time.time()
    ckpt_dir = Path(config["training"]["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Create an infinite iterator for continuous training
    train_iter = infinite_dataloader(train_dataloader)
    
    # Get max_steps from config if available, otherwise train indefinitely
    max_steps = config.get("training", {}).get("max_steps", float('inf'))
    
    step = 1
    while step <= max_steps:  # This will run indefinitely if max_steps is inf
        # Get next batch from the infinite iterator
        batch = next(train_iter)
        
        episodes = rollout(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            max_gen_len=config["training"]["max_gen_len"],
            num_answer_per_question=NUM_ANSWERS_PER_QUESTION,
            reward_function=reward_function,
            device=device,
            dtype=dtype,
        )
        if config["training"]["skip_unfinished_episodes"]:
            episodes = [episode for episode in episodes if episode.is_finished]
        results = update_policy(
            model=model,
            optimizer=optimizer,
            episodes=episodes,
            micro_batch_size=config["training"]["micro_batch_size"],
            pad_token_id=tokenizer.pad_token_id,
            max_grad_norm=config["training"]["max_grad_norm"],
            device=device,
            dtype=dtype,
        )
        torch.cuda.synchronize()
        end_time = time.time()
        duration = end_time - start_time
        start_time = end_time

        # compute and log important metrics
        reward = [episode.reward for episode in episodes]
        formatted_reward = [
            episode.reward_info["format_reward"] for episode in episodes
        ]
        answer_reward = [episode.reward_info["answer_reward"] for episode in episodes]
        num_finished_episodes = sum(episode.is_finished for episode in episodes)
        mean_reward = np.mean(reward)
        std_reward = np.std(reward)
        success_rate = np.mean(answer_reward)
        format_reward = np.mean(formatted_reward)
        grad_norm = results["grad_norm"]
        entropy = results["entropy"]
        lr = optimizer.param_groups[0]["lr"]
        loss = results["loss"]
        mean_response_len = np.mean(
            [len(episode.generated_token_ids) for episode in episodes]
        )
        print(
            f"\rStep {step}, mean_reward: {mean_reward:.2f}, "
            f"train success_rate: {success_rate:.2f}, "
            f"grad_norm: {grad_norm:.2f}, duration: {duration:.2f}, "
            f"num_finished_episodes: {num_finished_episodes}, "
            f"mean_response_len: {mean_response_len:.2f}, "
            f"entropy: {entropy:.2f}"
        )
        
        # Print detailed reward breakdown for a few episodes (similar to main_v2.py)
        print("\n=== Detailed Reward Breakdown ===")
        for i, episode in enumerate(episodes[:3]):  # Print first 3 episodes
            print(f"\nEpisode {i+1}:")
            print(f"Numbers: {batch.numbers[i % len(batch.numbers)]}, Target: {batch.target[i % len(batch.target)]}")
            print(f"Response: {episode.text}")
            print(f"Format reward: {episode.reward_info['format_reward']:.4f}")
            print(f"Answer reward: {episode.reward_info['answer_reward']:.4f}")
            print(f"Total reward: {episode.reward:.4f}")
        print("=== End Reward Breakdown ===\n")
        
        if step % config["training"]["eval_interval"] == 0:
            eval_success_rate = evaluate(model, tokenizer, device, dtype, config, max_samples)
            print(f"\rEval success rate: {eval_success_rate:.2f}" + " " * 100)
            tb_writer.add_scalar("success_rate/eval", eval_success_rate, step)

        tb_writer.add_scalar("loss", loss, step)
        tb_writer.add_scalar("mean_reward", mean_reward, step)
        tb_writer.add_scalar("std_reward", std_reward, step)
        tb_writer.add_scalar("success_rate/train", success_rate, step)
        tb_writer.add_scalar("format_reward", format_reward, step)
        tb_writer.add_scalar("grad_norm", grad_norm, step)
        tb_writer.add_scalar("duration", duration, step)
        tb_writer.add_scalar("num_finished_episodes", num_finished_episodes, step)
        tb_writer.add_scalar("learning_rate", lr, step)
        tb_writer.add_scalar("mean_response_len", mean_response_len, step)
        tb_writer.add_scalar("entropy", entropy, step)
        for i, episode in enumerate(episodes):
            # TensorBoard treats text as markdown.
            text = html.escape(episode.text)
            tb_writer.add_text(f"text_{i}", f"<pre>{text}</pre>", step)

        # save checkpoint
        if step % config["training"]["ckpt_save_interval"] == 0:
            # Extract model name for filename
            model_name_short = str(pretrained_model_path).split("/")[-1] if "/" in str(pretrained_model_path) else str(pretrained_model_path)
            # Get number of data samples
            num_samples = len(train_dataset)
            
            if is_hf_model:
                # Save HuggingFace model differently
                save_dir = ckpt_dir / f"{model_name_short}_samples{num_samples}_step{step:06d}"
                model.model.save_pretrained(save_dir)
                # Also save the tokenizer
                tokenizer.tokenizer.save_pretrained(save_dir)
                print(f"Saved HuggingFace model and tokenizer checkpoint to {save_dir}")
            else:
                # Save Qwen model as before
                output_file = ckpt_dir / f"{model_name_short}_samples{num_samples}_step{step:06d}.pt"
                torch.save(model.state_dict(), output_file)
                print(f"Saved checkpoint to {output_file}")
                
        step += 1


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--model", type=str, help="Model to use for training (default: use model from config)")
    parser.add_argument("--max_samples", type=int, default=200,
                        help="Maximum number of training samples to use (default: 200, use -1 for all samples)")
    args = parser.parse_args()
    main(args.config, args.model, args.max_samples)
