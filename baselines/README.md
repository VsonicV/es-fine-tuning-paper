# Reinforcement Learning Baselines

## GRPO Baseline

We use the [GRPO-Zero](https://github.com/policy-gradient/GRPO-Zero) repository for GRPO experiments.

### Fair Comparison with ES

To ensure a **fair comparison** with ES, we run two configurations with different **group sizes**:

#### Configuration 1: Group Size = 30 (same as ES population size)

```bash
model:
  pretrained_model_path: "meta-llama/Llama-3.2-3B-Instruct"
  device: "cuda"
  dtype: "bfloat16"
data:
  train_path: "data/countdown_samples.json"
  test_path: "data/countdown_samples.json"
  test_size: 128
  train_start_idx: 0
  train_end_idx: 200
training:
  random_seed: 1337
  max_prompt_len: 256
  max_gen_len: 1024
  batch_size: 60
  num_questions_per_batch: 2 # numb_answers_per_question = batch_size / num_questions_per_batch = 30
  # Number of examples per gradient accumulation step
  micro_batch_size: 2
  max_grad_norm: 1.0
  learning_rate: 1.0e-5
  weight_decay: 0.0
  betas: [0.9, 0.999]
  ckpt_dir: "ckpt"
  log_dir: "logs"
  skip_unfinished_episodes: false
  ckpt_save_interval: 100
  eval_interval: 10
  memory_efficient_adamw: false
```

#### Configuration 2: Group Size = 8

```bash
model:
  pretrained_model_path: "meta-llama/Llama-3.2-3B-Instruct"
  device: "cuda"
  dtype: "bfloat16"
data:
  train_path: "data/countdown_samples.json"
  test_path: "data/countdown_samples.json"
  test_size: 128
  train_start_idx: 0
  train_end_idx: 200
training:
  random_seed: 1337
  max_prompt_len: 256
  max_gen_len: 1024
  batch_size: 120
  num_questions_per_batch: 15 # numb_answers_per_question = batch_size / num_questions_per_batch = 8
  # Number of examples per gradient accumulation step
  micro_batch_size: 2
  max_grad_norm: 1.0
  learning_rate: 1.0e-5
  weight_decay: 0.0
  betas: [0.9, 0.999]
  ckpt_dir: "ckpt"
  log_dir: "logs"
  skip_unfinished_episodes: false
  ckpt_save_interval: 100
  eval_interval: 10
  memory_efficient_adamw: false
```

We make sure the total number of sample evaluations is the same as in the ES experiments.

### Usage

```bash
python train_countdown.py --config "your_config_file" --max_samples 200 --model "model_name"
```