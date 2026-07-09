# Inference backends

ES-at-scale runs its rollouts and its weight perturbations on a pluggable
**inference backend**. The trainer (`EvolutionStrategiesTrainer`) is backend-agnostic:
it holds an `ESBackend` and never imports vLLM or SGLang. Select one at runtime:

```bash
python es_at_scale/train.py --backend vllm   ...   # default
python es_at_scale/train.py --backend sglang ...
```

## The abstraction

`es_at_scale/backends/`:

| file | role |
|---|---|
| `base.py` | `ESBackend` ABC, `SamplingConfig`, `GenerationOutput`/`Completion` |
| `es_ops.py` | the shared ES weight math (perturb / restore / apply_update / snapshot / checksum). **Pure torch — imports no backend.** Installed into every backend env. |
| `vllm_backend.py` | vLLM engines; ES ops via `collective_rpc` → `WorkerExtension`; update on engine-0 + NCCL broadcast |
| `sglang_backend.py` | trainer-side driver: spawns out-of-process SGLang workers, ZeroMQ |
| `sglang_worker.py` | runs in the SGLang env; hosts the engine + ZMQ REP loop |
| `sglang_shim.py` | subclasses `sglang.Engine` + monkeypatches `es_*` onto the spawned `Scheduler` |

The ES algorithm is written once in `es_ops.py`; each backend only supplies a
`named_parameters()` accessor, generation, and the cross-engine sync policy.

## vLLM (default)

Unchanged mechanism from the original code, now behind the interface. Runs in the
**`es-prefix-cache`** conda env (python 3.12 + `vllm==0.11.0` + `transformers==4.57.6`).

```bash
conda activate es-prefix-cache
pip install -e ".[vllm]"
```

## SGLang

SGLang has no vLLM-style in-worker RPC, and its model lives in a spawned child
process. We inject ES weight ops with a **no-fork in-process shim**: subclass
`sglang.Engine`, override `run_scheduler_process_func` to monkeypatch `es_*`
methods (reaching `tp_worker.model_runner.model.named_parameters()`) onto the
`Scheduler` inside the spawned child, then drive them via `engine.collective_rpc`.
Only seeds/sampling/prompts cross the wire; noise is regenerated on the GPU.

Because SGLang's dependency tree conflicts with vLLM's, it runs **out-of-process in
its own conda env**, driven over ZeroMQ. Weight logic mirrors the vLLM backend
exactly: perturb in place, subtract-restore, apply the committed ES update on
engine 0, then **NCCL-broadcast** engine-0's weights to every engine each iteration
(via SGLang's `StatelessProcessGroup`/`PyNccl` inter-engine group — the same
primitives vLLM uses). Each engine holds only **1× the model weights**; the
per-iteration broadcast is the tradeoff that scales to large models, versus keeping
a full-size snapshot per engine (2× VRAM) or a slow CPU-resident one.

### Setup (one-time)

```bash
conda create -n sglang-es python=3.12 -y
/home/yinggan/miniconda3/envs/sglang-es/bin/pip install "sglang[all]==0.5.6.post2"
/home/yinggan/miniconda3/envs/sglang-es/bin/pip install -e "/path/to/es-at-scale[sglang]"   # no vllm
```

Tell the trainer where that env's python is (either flag or env var):

```bash
export ES_SGLANG_PYTHON=/home/yinggan/miniconda3/envs/sglang-es/bin/python
python es_at_scale/train.py --backend sglang --use-gpus 4,5,6,7 --n-vllm-engines 4 ...
#   or: --sglang-python /home/yinggan/miniconda3/envs/sglang-es/bin/python
```

### Guardrails (enforced in `sglang_worker.py`)

- `disable_radix_cache=True` — else KV computed under a perturbed model could be
  reused across weight changes and silently corrupt rewards. **Load-bearing.**
- `disable_overlap_schedule=True` — deterministic timing vs weight edits.
- CUDA graphs stay **on** (in-place `copy_`/`add_` preserves param storage).
- Startup `es_selftest` asserts the shim reached the child and the weight path
  resolves — a hard failure if the pinned SGLang internals ever move.
- After each ES step, engine-0's weights are NCCL-broadcast to every engine, so the
  transient per-engine bf16 drift from subtract-restore is re-synced (identical to
  vLLM). `checksums()` can verify engines match.

### Constraints / notes

- **Version-pinned to SGLang 0.5.6.post2** (the version whose internals the shim
  targets). Bumping SGLang requires re-verifying `run_scheduler_process_func`,
  `collective_rpc`, and `tp_worker.model_runner.model`; the self-test guards it.
- **N independent TP=1 engines** (one per GPU); `dp_size>1` is not supported. The
  inter-engine broadcast group is one rank per engine (TP>1 would need per-TP-rank
  groups, as vLLM does).
- Each engine holds **1× the model weights** (no snapshot); the committed update is
  computed on engine 0 and broadcast, so VRAM does not grow with engine count.
- Generation is not bit-identical to vLLM (different kernels); parity is measured
  by pass@1 / argmax agreement, not equality.

## Testing

```bash
# Tier A — deterministic, CPU, no engine:
python tests/test_es_ops.py

# Tier B — real engines on GPU 4-7:
python tests/e2e/test_backend.py --backend vllm   --gpus 4,5
ES_SGLANG_PYTHON=.../sglang-es/bin/python \
python tests/e2e/test_backend.py --backend sglang --gpus 6,7
```
