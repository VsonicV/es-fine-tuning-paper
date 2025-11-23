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


## PPO Baselines
We use the [TinyZero](https://github.com/Jiayi-Pan/TinyZero)  repository for PPO experiments.


### Usage
```python
export N_GPUS=8
export BASE_MODEL=Qwen/Qwen2.5-3B-Instruct
export DATA_FILE=data/countdown.json
export VAL_DATA_FILE=${VAL_DATA_FILE:-$DATA_FILE}
export ROLLOUT_TP_SIZE=1
export EXPERIMENT_NAME=countdown-qwen2.5-3b-instruct
export VLLM_ATTENTION_BACKEND=XFORMERS
export CHECKPOINT_DIR=/path/to/your/checkpoint/directory

python3 -m verl.trainer.main_ppo \
 data.train_files=$DATA_FILE \
 data.val_files=$VAL_DATA_FILE \
 data.train_batch_size=128 \
 data.max_prompt_length=1024 \
 data.max_response_length=1024 \
 actor_rollout_ref.model.path=$BASE_MODEL \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.model.use_remove_padding=True \
 +actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
 +critic.model.fsdp_config.model_dtype=bfloat16 \
 +actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
 actor_rollout_ref.actor.ppo_mini_batch_size=64 \
 actor_rollout_ref.actor.ppo_micro_batch_size=8 \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
 actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
 critic.optim.lr=1e-5 \
 critic.model.path=Qwen/Qwen2.5-0.5B-Instruct \
 critic.ppo_micro_batch_size=8 \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.logger='["console"]' \
 trainer.n_gpus_per_node=$N_GPUS \
 trainer.nnodes=1 \
 trainer.save_freq=100 \
 trainer.test_freq=100 \
 trainer.total_epochs=100000 \
 trainer.default_local_dir=$CHECKPOINT_DIR \
 trainer.default_hdfs_dir=$CHECKPOINT_DIR \
 trainer.project_name=$EXPERIMENT_NAME \
 trainer.experiment_name=$EXPERIMENT_NAME 2>&1 | tee logs/${EXPERIMENT_NAME}.log
```