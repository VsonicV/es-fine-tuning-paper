# ES-at-Scale: Evolution Strategies at Scale

<div align="center">
  <img src="assets/es_at_scale_final.gif" alt="ES-at-Scale" width="100%"/>
</div>

## Overview

**ES-at-Scale** is an open-source framework for fine-tuning large language models using **Evolution Strategies (ES)** — a fully backpropagation-free, massively parallelizable alternative to RL-based training methods like PPO and GRPO.

Model updates are computed from population-level reward signals, performing **direct optimization in the full parameter space**:
- **No backpropagation**
- **No optimizer states**
- **No activations stored**
- **No dimensionality reduction or low-rank adapters**

Training is built on **Ray** for distributed execution and **vLLM** for high-throughput inference, enabling efficient multi-GPU rollout evaluation at scale.

---

## 🎉 What's New

---

## At a glance

**ES-at-Scale** is free to use, modify, and build on.

- **Massively parallel**: each perturbation (population member) can be evaluated independently
- **Works with non-differentiable rewards**: symbolic graders, program execution, discrete checks
- **Inference-only systems**: leverage vLLM throughput instead of training-time backprop

---

# Training Models of Any Size That Fit Your Hardware

A major design goal of this repository is **model‑size flexibility** — and, in turn, **democratizing LLM fine‑tuning**. Using this exact implementation we have successfully trained **0.5B, 3B, 7B, 14B, 32B, and 72B parameter models** with no architectural changes.

> If the model **fits in your GPUs under vLLM**, it can be fine‑tuned with this ES framework.

Because there are no gradients, optimizer states, or stored activations, the scaling challenge shifts from gradient computation to **inference throughput**. Smaller teams can fine‑tune large models on inference‑optimized clusters, with arbitrary (including non‑differentiable) reward functions.

---

## Requirements

- Python **3.12** is reccommended. Other python versions are untested and may not work
- CUDA-enabled GPUs (**multi-GPU strongly recommended**)
- Linux environment
- CUDA, PyTorch, Ray, and vLLM properly installed

---

## Setup

### 1. Create a virtual environment

> Python 3.12 is supported and tested. Other versions are untested and may not work.

```bash
python3.12 -m venv es
source es/bin/activate
```

### 2. Install the repository

From the repository root:

```bash
pip install -e .
```

### 3. Install additional dependencies

```bash
pip install wandb
```

### 4. Install math evaluation dependencies

```bash
pip install math-verify
pip install pylatexenc
pip install latex2sympy2_extended
```

> **Note**  
> Ensure `nvidia-smi` works correctly and that Ray detects all available GPUs.

---

## Datasets

The trainer **only requires PyTorch DataLoaders**. It does not care how your data is stored — HuggingFace datasets, JSON files, CSV, databases, or anything else all work, as long as you wrap them in a DataLoader that yields the right format.

Each batch must return:

```python
(list_of_prompts, list_of_targets)
```

That's the only requirement. The field names in your underlying data, the file format, and the storage location are entirely up to you.

## Example: plain PyTorch DataLoader

### 1) Build a plain PyTorch dataset

```python
from torch.utils.data import Dataset, DataLoader

class SimpleMathDataset(Dataset):
    def __init__(self):
        self.rows = [
            {"problem": "What is 2+2?", "answer": "4"},
            {"problem": "What is 3+5?", "answer": "8"},
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]
```

### 2) Define a collate function

```python
def collate_fn(batch):
    prompts = [x["problem"] for x in batch]
    answers = [x["answer"] for x in batch]
    return prompts, answers
```

### 3) Create training and evaluation dataloaders

```python
train_dataset = SimpleMathDataset()
eval_dataset = SimpleMathDataset()

train_dataloader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_fn,
)

eval_dataloader_dict = {
    "toy_eval": DataLoader(
        eval_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn,
    )
}
```

### Example directory structure

The structure below shows how the default `train.py` organises datasets on disk. This is the layout expected by the `--train-dataset` and `--eval-dataset` flags, which use HuggingFace `load_from_disk` as a convenience loader. If you load data a different way, this layout is not required — replace the loader in `train.py` with anything that produces a DataLoader yielding `(list[prompt], list[target])` batches.

```
datasets/
├── train/
│   ├── math_lvl3to5_8k/
│   └── countdown/          
└── evaluation_suite/
    ├── math/
    │   ├── amc/
    │   ├── aime/
    │   ├── math500/
    │   ├── minerva/
    │   └── olympiad_bench/
    └── countdown/
        └── countdown_eval/
```

### Specifying datasets with `--train-dataset` / `--eval-dataset`

> **This `DatasetDict` layout is only the convenience loader used by `train.py` — it is not a requirement of the framework.** The trainer itself only consumes PyTorch DataLoaders that yield `(list[prompt], list[target])` batches (see [Datasets](#datasets)). **Any** storage format — JSON, CSV, Parquet, a database, a remote API, an in-memory list — works just as well; you simply plug it in with your own `Dataset`/`DataLoader` code in place of the `load_from_disk` calls in `train.py`. The rest of this subsection applies only if you choose to use the built-in `load_from_disk` loader.

The convenience loader in `train.py` uses HuggingFace `load_from_disk`, which expects a **`DatasetDict` saved to disk** — a folder containing a `dataset_dict.json` file plus one subfolder per split. Always point these flags at the folder that *contains* `dataset_dict.json`, **not** at an individual split subfolder.

- `--train-dataset` — must be a `DatasetDict` folder. `train.py` iterates its splits to build the training DataLoader; conventionally there is a single split named `train`.
- `--eval-dataset` — must be a `DatasetDict` folder. **Each split is evaluated separately** and reported under its split name (e.g. `eval/<split>/pass@1/mean`). This is how the math suite reports `amc`, `aime`, `math500`, `minerva`, and `olympiad_bench` individually.

Create your own from raw data with `save_to_disk`:

```python
from datasets import Dataset, DatasetDict

rows = Dataset.from_list([
    {"problem": "What is 2+2?", "answer": "4"},
    {"problem": "What is 3+5?", "answer": "8"},
])

# Train: a single `train` split is conventional.
DatasetDict({"train": rows}).save_to_disk("datasets/train/my_task")

# Eval: each split becomes a separately-reported benchmark.
DatasetDict({
    "my_eval_a": rows,
    "my_eval_b": rows,
}).save_to_disk("datasets/evaluation_suite/my_task")
```

Then point the flags at the folders that contain `dataset_dict.json`:

```bash
--train-dataset "datasets/train/my_task" \
--eval-dataset "datasets/evaluation_suite/my_task"
```

> The columns each row must contain are dictated by your collate function — e.g. `problem`/`answer` for the math task, `context`/`numbers`/`target` for countdown. See the [Datasets](#datasets) requirements above.

---

## Customization Guide: Reward Function, Prompt Template, and Trainer

This repo is designed so you can plug in **your own task + evaluator** without touching the ES core logic. At a minimum you provide:

1. A **template function** that turns a raw dataset question into the exact prompt string you want to feed the model.
2. A **reward function** that scores each model output against the target answer (can be non-differentiable).
3. (Optional) Your own **trainer** (subclass `EvolutionStrategiesTrainer`) if you want to change batching, logging, or rollout processing.

### 1) Implement a prompt template function

Your template function must accept a single `question: str` and return a **full prompt** string. It is called here:

- training: `input_text = [self.template(i) for i in input_text]`
- eval: `prompts = [self.template(i) for i in input_text]`

Example (Qwen math chat template, from `train.py`):

```python
def apply_qwen_math_template(question: str) -> str:
    return (
        "<|im_start|>system\n"
        "Please reason step by step, and put your final answer within \\boxed{}."
        "<|im_end|>\n"
        "<|im_start|>user\n"
        + question
        + "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
```

Tips:
- Make sure the returned string includes the right “assistant start” marker for your model (if applicable).
- If you want to do few-shot prompting, add examples inside this function.

### 2) Implement a reward function

Your reward function is invoked inside the trainer with:

```python
self.task = functools.partial(reward_function)
...
fmt, r = self.task(response_text, target_text)   # executed in a multiprocessing Pool with a timeout
```

So your function should have this signature:

```python
def my_reward_fn(model_output: str, target: str, fast: bool = False) -> tuple[str, float]:
    ...
```

And it should return:
- `fmt` (a short string label you can use for debugging, e.g. `"ok"`, `"timeout"`, `"bad_format"`)
- `reward` (a scalar float; higher is better)

Minimal example (exact-match):

```python
def exact_match_reward_fn(model_output: str, target: str, fast: bool = False):
    pred = model_output.strip()
    gold = target.strip()
    return ("exact_match", 1.0 if pred == gold else 0.0)
```

More realistic example (extract `\boxed{...}` final answer):

```python
import re

_BOX_RE = re.compile(r"\\boxed\{([^}]*)\}")

def boxed_final_answer_reward_fn(model_output: str, target: str, fast: bool = False):
    m = _BOX_RE.search(model_output)
    if m is None:
        return ("missing_box", 0.0)

    pred = m.group(1).strip()
    gold = target.strip()
    return ("boxed", 1.0 if pred == gold else 0.0)
```

Important notes:
- The reward function runs in a **multiprocessing pool** with a **timeout** in `_postprocess_outputs()`. If your grader might be slow (e.g., symbolic math, code execution), increase the timeout by increase the `reward_function_timeout` parameter. It is set to 60 seconds by default.
- If the reward function throws or times out, training currently assigns **0.0 reward** for that rollout.

### 3) Wire your functions into `train.py`

In `train.py`, import and assign your functions, then pass them into the trainer:

```python
from es_at_scale.trainer.es_trainer import EvolutionStrategiesTrainer
from my_project.reward import my_reward_fn
from my_project.templates import my_template_fn

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
    reward_function=my_reward_fn,
    template_function=my_template_fn,
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
    reward_function_timeout=args.reward_function_timeout,
    save_best_models=args.save_best_models,
)
trainer.fit()
```

### 4) (Optional) Implement your own `EvolutionStrategiesTrainer`

Most users **do not** need to change the trainer. But if you want custom behavior, the intended path is to subclass and override the parts you need:

Common extension points:
- `_postprocess_outputs()` — change reward aggregation, logging payloads, or what gets saved
- `evaluate_population_on_batch()` — change how you schedule seeds across engines
- `launch_engines()` — change vLLM / Ray actor configuration

Example skeleton:

```python
from es_at_scale.trainer.es_trainer import EvolutionStrategiesTrainer

class MyTrainer(EvolutionStrategiesTrainer):
    def eval_step(self, args):

        results = ....
        
        return results
```

Then instantiate `MyTrainer` instead of `EvolutionStrategiesTrainer` in `train.py`.


---

## Usage

### Example: Fine-Tuning with `train.py`

`train.py` is a **task-specific example** showing how to use `EvolutionStrategiesTrainer`. It bundles two ready-to-run tasks — `countdown` (the default) and `math` — selected via `--task`. Each task wires in its own reward function, prompt template, and on-disk HuggingFace datasets. To use ES on your own task, copy `train.py` and replace those three components — the trainer itself is fully task-agnostic.

`--task` selects the bundled configuration:
- `countdown` *(default)* — the Countdown task with the `<think>`/`<answer>` format reward and a pass-through template (the dataset's `context` field already contains the full prompt).
- `math` — math reasoning with the boxed-answer reward and Qwen math chat template.

#### Countdown

The example below fine-tunes **Qwen/Qwen2.5-1.5B-Instruct** on 8 GPUs against the Countdown task:

```bash
python es_at_scale/train.py \
  --task countdown \
  --model-name "Qwen/Qwen2.5-1.5B-Instruct" \
  --sigma 0.001 \
  --population-size 30 \
  --n-iterations 300 \
  --eval-freq 5 \
  --train-dataset "datasets/train/countdown" \
  --eval-dataset "datasets/evaluation_suite/countdown" \
  --batch-size 200 \
  --mini-batch-size 200 \
  --max-tokens 512 \
  --n-vllm-engines 8 \
  --use-gpus "0,1,2,3,4,5,6,7" \
  --output-directory "./experiments/" \
  --experiment-name "my-first-countdown-run" \
  --wandb-project "es-finetuning" \
  --logging wandb
```

#### Math

The example below fine-tunes **Qwen/Qwen2.5-Math-7B** on 8 GPUs against the MATH level 3–5 training set:

```bash
python es_at_scale/train.py \
  --task math \
  --model-name "Qwen/Qwen2.5-Math-7B" \
  --sigma 0.001 \
  --population-size 30 \
  --n-iterations 300 \
  --eval-freq 5 \
  --train-dataset "datasets/train/math_lvl3to5_8k" \
  --eval-dataset "datasets/evaluation_suite/math/" \
  --batch-size 1024 \
  --mini-batch-size 1024 \
  --max-tokens 3000 \
  --n-vllm-engines 8 \
  --use-gpus "0,1,2,3,4,5,6,7" \
  --output-directory "./experiments/" \
  --experiment-name "my-first-math-run" \
  --wandb-project "es-finetuning" \
  --logging wandb
```

---

## Command-Line Arguments

| Argument | Default | Description |
|---|---|---|
| `--task` | `countdown` | Bundled task config: `countdown` or `math`. Selects reward function, prompt template, and dataset collate. |
| `--model-name` | `Qwen/Qwen2.5-1.5B-Instruct` | HuggingFace model ID or local path |
| `--checkpoint` | — | Path to a `.pth` ES checkpoint to resume from |
| `--sigma` | `0.001` | Noise scale for ES perturbations |
| `--alpha` | `sigma/2` | Learning rate (pass `-1` to auto-set to `sigma/2`) |
| `--reward-shaping` | `z-scores` | Reward normalization strategy |
| `--population-size` | `30` | Number of perturbations per iteration |
| `--n-iterations` | `300` | Total number of ES training iterations |
| `--eval-freq` | `5` | Run evaluation every N iterations |
| `--train-dataset` | `datasets/train/countdown` | Path to training DatasetDict on disk |
| `--eval-dataset` | `datasets/evaluation_suite/countdown` | Path to evaluation DatasetDict on disk |
| `--batch-size` | `512` | Number of prompts used to evaluate the current ES population at a given iteration (the reward signal for one weight update is computed over this many prompts) |
| `--mini-batch-size` | `512` | How many of current batch each vLLM engine generates at once. A memory/throughput knob only — it does **not** change the ES update. See note below. |
| `--max-tokens` | `512` | Maximum tokens per generated response |
| `--n-vllm-engines` | `8` | Number of vLLM engines (one per GPU recommended) |
| `--n-gpu-per-vllm-engine` | `1` | GPUs per vLLM engine |
| `--logging` | `wandb` | Logging backend (`wandb` or `none`) |
| `--seed` | `42` | Global random seed |
| `--use-gpus` | `0,1,2,3,4,5,6,7` | Comma-separated GPU indices to use |
| `--reward-function-timeout` | `10` | Timeout (seconds) for reward function calls |
| `--output-directory` | `./experiments/` | Root directory for checkpoints and logs |
| `--save-best-models` | `False` | Save a checkpoint each time eval score improves |
| `--experiment-name` | auto-generated | Name for this run (used in wandb and checkpoint paths) |
| `--wandb-project` | `es-finetuning` | Wandb project name |

### `--batch-size` vs. `--mini-batch-size`

`--batch-size` is the number of prompts used to evaluate the current population at a given timestep — every population member sees the same prompts, and their average rewards are used for an iteration's weight update.

`--mini-batch-size` is a memory lever: it splits that fixed batch into sequential chunks run through each vLLM engine one at a time, so the full batch never has to fit in memory at once. Rewards are accumulated across chunks with size-weighting, so **it does not change the result** — only peak memory and speed. If rollout hits OOM (common with long `--max-tokens` or large models), lower it.

The batch is split into `ceil(batch_size / mini_batch_size)` chunks, processed one after another. There are three cases:

- **`batch_size == mini_batch_size`** (default): one chunk — the whole batch is processed in a single pass. Highest memory.
- **`batch_size > mini_batch_size`**: multiple chunks (e.g. batch 512, mini 128 → 4 passes of 128). Lower peak memory, more sequential passes.
- **`batch_size < mini_batch_size`**: one chunk — the mini-batch is capped at the batch size, so it behaves exactly like the default.

**Evaluation batch size.** In `train.py`, evaluation uses `--mini-batch-size` as its batch size, since `--mini-batch-size` is already tuned to the largest batch that fits in memory without an OOM. Override it in `train.py` if you want to tune it independently.

### Fixed sampling parameters

The following are not configurable via CLI:

| Parameter | Value | Description |
|---|---|---|
| Training temperature | `0.0` | Greedy decoding during training rollouts |
| Training top-p | `1.0` | The entire token distribution is included — no tokens are filtered out |
| Eval temperature | `0.0` | Greedy decoding during evaluation |
| Eval top-p | `1.0` | The entire token distribution is included — no tokens are filtered out |
| Rollouts per prompt | `1` | Single sample per population member |

---

## Evaluation Only

Currently, there is no separate evaluation entry point — evaluation is run through the same `train.py`. Setting `--n-iterations 0` puts the trainer in **eval-only mode**: it runs a single evaluation pass on your unmodified model and exits, performing no training steps and saving no checkpoint.

- Set `--n-iterations 0` to evaluate only.
- Pass `--checkpoint` to evaluate a fine-tuned ES checkpoint, or omit it to evaluate the raw base model.
- Per-sample outputs are written to `experiments/<experiment-name>/eval-output/model_eval_task<name>_iteration0.json`, and pass@1 is printed to stdout.
- Evaluation runs on a **single vLLM engine** (`engines[0]`), so the examples below use `--n-vllm-engines 1`. The other engines are only used to parallelize the population during training.

### Example: evaluate the base model

```bash
python es_at_scale/train.py \
  --task math \
  --model-name "Qwen/Qwen2.5-Math-7B" \
  --eval-dataset "datasets/evaluation_suite/math/" \
  --max-tokens 3000 \
  --n-iterations 0 \
  --n-vllm-engines 1 \
  --use-gpus "0" \
  --output-directory "./experiments/" \
  --experiment-name "eval-base-model" \
  --wandb-project "es-evaluation" \
  --logging "wandb"
```

### Example: evaluate a fine-tuned checkpoint

```bash
python es_at_scale/train.py \
  --task math \
  --model-name "Qwen/Qwen2.5-Math-7B" \
  --checkpoint "experiments/<run>/checkpoint-es_fine_tuned_iteration_300/pytorch_model.pth" \
  --eval-dataset "datasets/evaluation_suite/math/" \
  --max-tokens 3000 \
  --n-iterations 0 \
  --n-vllm-engines 1 \
  --use-gpus "0" \
  --output-directory "./experiments/" \
  --experiment-name "eval-checkpoint" \
  --wandb-project "es-evaluation" \
  --logging "wandb"
```

### Notes

- **The number of GPUs required depends on your hardware and model size.** The examples above use a single GPU (`--use-gpus "0"`, `--n-vllm-engines 1`), which is sufficient for smaller models. Larger models that do not fit on one GPU need tensor parallelism — increase `--n-gpu-per-vllm-engine` and list the corresponding GPUs in `--use-gpus` (e.g. `--n-gpu-per-vllm-engine 4 --use-gpus "0,1,2,3"`).

- **Only greedy decoding is currently supported** for evaluation (temperature `0.0`, top-p `1.0`, one rollout per prompt). These eval sampling parameters are fixed and not configurable via the CLI.
- **In a future release we plan to abstract the evaluation step** so users can customize eval parameters (e.g. sampling temperature, top-p, multiple rollouts per prompt) directly.
- **For now, to run any evaluation beyond greedy decoding**, override the current behavior by subclassing `EvolutionStrategiesTrainer` and reimplementing `eval_step()` (and/or the sampling parameters it builds) in your own trainer class. See the [Customization Guide](#customization-guide-reward-function-prompt-template-and-trainer).

---

## Training Loop Overview

1. Ray launches persistent vLLM engines (typically one per GPU).
2. For each ES iteration:
   - Independent parameter perturbations are sampled.
   - Each population member performs rollout inference in parallel.
   - Rewards are computed using task-specific evaluators.
3. Rewards are normalized and aggregated into a population-level update.
4. The base model is updated **without gradients or backpropagation**.
5. Evaluation runs every `--eval-freq` iterations across all benchmarks.

---

## Outputs

Each run produces a timestamped experiment directory:

```
experiments/
└── es-finetuned-.../
    ├── checkpoints/
    ├── eval-output/
    └── train-output/
```

### Checkpoints

- The final model weights are always saved at the end of training as `checkpoint-es_fine_tuned_iteration_<N>/pytorch_model.pth`.
- When `save_best_models=False` (the default), set to True to save each time a new best mean eval score is achieved during training. These are written to `checkpoints/<experiment_name>-mean<score>/pytorch_model.pth`. Default is `False` to save disk-space.
- When `save_best_models=False`, only the final model is saved.

### Logging

With `--logging wandb`, the following are tracked:

- Training reward statistics
- Evaluation pass@1 metrics
- ES hyperparameters
- Throughput and rollout diagnostics

---

## Performance Notes

- ES scales nearly linearly with the number of GPUs (population parallelism)
- No synchronization barriers from backpropagation
- Ideal for **single-node multi-GPU** and **distributed Ray clusters**
- Tune `--mini-batch-size` to balance memory usage and throughput

---

## Entry Point

`train.py` is a working example for math fine-tuning, not a general CLI. The real entry point for your own task is `EvolutionStrategiesTrainer` — see the [Customization Guide](#customization-guide-reward-function-prompt-template-and-trainer) for how to wire in your own reward function, prompt template, and data loader.

---

## Citation

If you use ES-at-Scale in your research, please cite our paper:

```bibtex
@article{Qiu2025EvolutionStrategies,
  title={Evolution Strategies at Scale: {LLM} Fine-Tuning Beyond Reinforcement Learning},
  author={Qiu, Xin and Gan, Yulu and Hayes, Conor F. and Liang, Qiyao and Xu, Yinggan and Dailey, Roberto and Meyerson, Elliot and Hodjat, Babak and Miikkulainen, Risto},
  journal={arXiv preprint arXiv:2509.24372},
  year={2025},
  eprint={2509.24372},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  doi={10.48550/arXiv.2509.24372}
}
```
