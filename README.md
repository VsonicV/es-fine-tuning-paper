
# es-fine-tuning-paper

This repo contains the source code for the paper "Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement
Learning" (https://arxiv.org/abs/2509.24372). Evolution strategies (ES) is used to directly optimize billions of
parameters of large language models (LLMs).

Feel free to join the ES fine-tuning forum in [Discussions](https://github.com/VsonicV/es-fine-tuning-paper/discussions).

## News

10/27/2025: :fire::fire::fire: An accelerated version with **10X+ speed-up** in running time is added to the repo!
See [here](https://github.com/VsonicV/es-fine-tuning-paper?tab=readme-ov-file#accelerated-version-10x-speed-up). :rocket::rocket::rocket:

Note: we are still actively adding more experimental codes into this repo. We expect breaking change to the
accelerated implementations.

## Setup

Create a virtual environment with python version >= 3.10 and activate it

```bash
python -m venv es
source es/bin/activate
```

From the root of the repository run following command to install all the relevant python packages

```bash
pip install -r requirement.txt
```

## Usage

For running the main ES code on conciseness fine-tuning

```bash
accelerate launch \
    --num_processes 2 \
    --num_machines 1 \
    --machine_rank 0 \
    es_fine-tuning_conciseness.py \
    --gpu_threads=1 \
    --model_name=Qwen/Qwen2.5-7B-Instruct
```

`--num_processes` specifies the number of GPUs to use and `--gpu_threads` specifies the number of threads inside
each GPU. The total number of parallel evaluations is thereby equal to `num_processes`*`gpu_threads`.

For running the main ES code on the Countdown task

```bash
accelerate launch \
    --num_processes 4 \
    --num_machines 1\
    --machine_rank 0 \
    countdown/es_fine-tuning_countdown.py \
    --data_sample 200 \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --gpu_threads 1
```

### Other Parameters

- `--gpu_ids`: Specify which GPUs to use (CUDA device id), argument for `accelerate launch`
- `--model_name`: HuggingFace model to fine-tune
- `--hf_cache_dir`: Directory for HuggingFace cache
- `--precision`: Model precision, default to be `bf16`
- `--verbose`: Enable detailed logging if this argument is present in the command line

Note: The original implementation uses a partially correlated noise. To use complete i.i.d. noise, please use
`es_fine-tuning_conciseness_iid.py` and `countdown/es_fine-tuning_countdown_iid.py` instead.
See [here](https://github.com/VsonicV/es-fine-tuning-paper/discussions/7) for more details.

## Accelerated Version (10X+ Speed-up)

If you are using the latest accelerated version `es-fine-tuning_countdown_accl.py`, please also install
the `vllm` and `tensorboard` by:

```bash
pip install vllm==0.11.0
pip install tensorboard
```

For running the accelerated version on the Countdown task:

```bash
# Single-GPU quickstart
python es_fine-tuning_countdown_accl.py \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --cuda_devices 0 \
  --num_engines 1 \
  --population_size 30 \
  --num_iterations 1000

# Multi-GPU run (one vLLM engine per GPU)
python es_fine-tuning_countdown_accl.py \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --cuda_devices 0,1,2,3 \
  --num_engines 4 \
  --population_size 30 \
  --num_iterations 1000 \
  --sigma 0.001 \
  --alpha 0.0005 \
  --experiment_dir es-ft-experiment
```

On preliminary 4xH100 setting, accelerated version achieves ~10 times speed-up with similar convergence rate.

## Contributing to the repository

If you would like to contribute to this repository, please follow these [steps](./docs/README_contributing.md)
to make sure your code is in the proper format and is free of lint errors.

## Citation

If you find this work helpful in your research, please cite:

```bibtex
@misc{qiu2025evolutionstrategiesscalellm,
      title={Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning}, 
      author={Xin Qiu and Yulu Gan and Conor F. Hayes and Qiyao Liang and Elliot Meyerson and Babak Hodjat and Risto Miikkulainen},
      year={2025},
      eprint={2509.24372},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2509.24372}, 
}
```
