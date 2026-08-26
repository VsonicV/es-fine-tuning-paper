# Setup & debugging notes (for future reference)

This documents *why* the code in this repo differs from a fresh clone of
https://github.com/VsonicV/es-at-scale.git, and the debugging trail that led
to each change. See `/workspace/PROGRESS.md` for current status and the
recommended launch command — this file is the "why," that one is the "what/now."

## Environments

Two separate Python environments exist on this machine — don't mix them up:

1. **`/venv/main`** — the base Vast.ai image's own environment, running the
   instance's built-in "vLLM API" supervisor service (unrelated to this
   training job). Has `vllm==0.26.0` installed. Has `es_at_scale` also
   installed here (`pip install --no-deps -e .`) from early experimentation —
   harmless leftover, not used for the final approach.
2. **`/workspace/es-at-scale/.venv`** — dedicated venv for this repo, built
   with the repo's actual pinned dependencies from `setup.py`
   (`vllm==0.11.0`, `transformers==4.57.6`). **Use this one for training.**

### Why two environments
The repo's `setup.py` pins `vllm==0.11.0`, which conflicts with the base
image's `vllm==0.26.0`. Initially tried running against the newer 0.26.0
(already installed, faster to get started) rather than the pinned version.
This led to three separate real bugs (below), all specific to 0.26.0's
handling of this repo's unusual usage pattern: N independent, co-located,
single-GPU `LLM()` instances constructed via Ray in one process (not vLLM's
typical single-engine-per-process or native multi-node data-parallel setup).
After the third bug (see "CUDA driver error" below) turned out to still be
unresolved even after eliminating concurrency and isolating three separate
cache directories, the decision was made to stop reverse-engineering 0.26.0
internals and just build a venv with the version the repo actually targets.
**That fully resolved everything on the first try** — an 8-engine smoke test
passed cleanly with vllm==0.11.0, no patches needed for that version's bugs.

## Code changes in `es_at_scale/trainer/es_trainer.py`

### 1. Version-agnostic `get_ip`/`get_open_port` import
```python
try:
    from vllm.utils.network_utils import get_ip, get_open_port  # vLLM >= 0.20ish
except ImportError:
    from vllm.utils import get_ip, get_open_port  # vLLM 0.11.0 (pinned by this repo)
```
vLLM moved these between versions. Harmless either way; keeps the file working
if the venv ever changes vLLM version again.

### 2. `ESNcclLLM.__init__` — per-engine port/cache isolation
Added an `engine_idx` parameter that sets, before `super().__init__()`:
- `VLLM_DP_MASTER_PORT` — unique per engine (`29500 + engine_idx*100`)
- `VLLM_CACHE_ROOT`, `TRITON_CACHE_DIR`, `CUDA_CACHE_PATH` — each pointed at a
  per-engine subdirectory

**Root cause this was fixing (vLLM 0.26.0 only, confirmed NOT needed under
0.11.0, but left in place since it's harmless):**
- With `n_vllm_engines=8` co-located on one node, every engine independently
  resolves `data_parallel_rank_local` and `data_parallel_master_port` to `0`
  (traced through `vllm/config/parallel.py`: since `data_parallel_size==1`,
  it falls back to reading `VLLM_DP_RANK_LOCAL`/`VLLM_DP_MASTER_PORT` from
  `os.environ`, which are unset → `vllm.envs` defaults both to `0`). This
  makes every engine compute the *identical* TCPStore bind port
  (`0 + 100 + 0 = port 100`) — only the first to bind succeeds, the rest die
  with `EADDRINUSE` / `DistNetworkError`.
- After fixing the port collision, hit `RuntimeError: CUDA driver error:
  invalid argument` during `profile_run`/CUDA-graph-capture on every engine
  except the first. Chased this through three separate shared-cache-directory
  theories in sequence — `VLLM_CACHE_ROOT` (vLLM's torch.compile artifact
  cache), `TRITON_CACHE_DIR` (Triton's own kernel cache, separate from the
  above), `CUDA_CACHE_PATH`/`~/.nv/ComputeCache` (the NVIDIA driver's own
  PTX→SASS JIT cache) — isolating all three per-engine. **The error persisted
  regardless.**

### 3. `launch_engines()` — serialized engine construction
```python
# was: engines = [ray.remote(...)(ESNcclLLM).remote(...) for idx, strategy in enumerate(strategies)]
# now: a for-loop that does ray.get(engine.collective_rpc.remote("_set_seed", args=(0,)))
#      after each engine's construction, before starting the next
```
This was the fix that actually resolved the CUDA driver error under 0.26.0.
Diagnostic trail: even with all three caches isolated, concurrent engine
launch still failed (always engine 0 succeeds, all others fail identically).
Serializing engine startup (so only one engine ever compiles/captures CUDA
graphs at a time) did NOT immediately fix it either on the first retry — the
*second* engine still failed, in complete isolation, no concurrency at all.
A follow-up isolation test proved GPU 1 (the second engine's GPU) works fine
completely on its own — ruling out a hardware fault on that GPU. This pointed
to some kind of process-sequence-dependent state corruption specific to
vLLM 0.26.0's compile/capture pipeline when multiple engines are constructed
in sequence within one process — root cause never fully pinned down, because
switching to vllm==0.11.0 (below) made the whole problem disappear.

**Cost/benefit of leaving this in for 0.11.0**: only tested 0.11.0 *with*
this serialization active (never re-tested without it once 0.11.0 was
installed) — so it's unknown whether 0.11.0 strictly needs it. It adds
~4-5 minutes to engine startup (8 engines × ~30-40s each, sequential instead
of parallel) versus a run that lasts hours to days — negligible cost either
way, so it wasn't worth the risk of re-testing without it. Fine to leave as-is.

### 4. `gpu_memory_utilization` — default `0.7`, now overridable
Tested `0.85` (more KV cache headroom) — identical wall-clock time to `0.7`
on the same workload (384.5s vs 384.7s for 1024 prompts @ max_tokens=4096).
Confirms the workload is compute-bound, not memory-bound, in this regime.
Reverted to `0.7` since `0.85` gave no benefit and left less headroom for the
ES-specific weight-perturbation/broadcast tensors (which allocate outside
vLLM's own memory budget — see `worker_extension.py`'s `perturb_self_weights`
etc., each does raw `torch.randn` per parameter tensor).

**Update — this is now `ES_VLLM_GPU_MEM_UTIL`, default unchanged at `0.7`.** The
paragraph above holds on 48GB cards. It does not hold on small-VRAM GPUs, where
`0.7` is not survivable at all: the headroom it leaves is smaller than what the
ES weight update needs.

`update_weights_from_seeds` holds three tensors live at once for each parameter —
an fp32 accumulator, the bf16 noise, and `noise.to(torch.float32) * coeffs[i]`.
On Qwen2.5-1.5B's tied `151936x1536` embedding that peaks at **~2.2GB** for that
one parameter, and it OOMs at `worker_extension.py:121` *after generation has
already succeeded*, which makes it look like a training bug rather than a
budgeting one.

Two env knobs, both defaulting to the previous behaviour so nothing changes on
the boxes this file was written for:

| var | default | when to set it |
|---|---|---|
| `ES_VLLM_GPU_MEM_UTIL` | `0.7` | lower it until `total - vLLM budget` exceeds ~2.5GB |
| `ES_VLLM_MAX_MODEL_LEN` | unset (model's own) | set to `max_prompt + max_tokens` when the KV cache cannot hold the model's full context |

The second is not optional once the first is lowered: Qwen2.5's own
`max_model_len` is **131072**, and below roughly `0.55` on a 12GB card the KV
cache can no longer hold it, so vLLM refuses to start rather than degrading.

Verified on 8x RTX 3060 12GB with `0.6` / `4096`: 2.63GiB and 98,352 tokens of
KV cache per engine, two training iterations, checkpoint written.

**Peak is reducible if this ever becomes binding on a larger model.**
`update_accumulator.add_(noise, alpha=coeffs[i])` would drop the separate fp32
`term` and cut ~892MiB off the peak. Left alone deliberately — it changes the
accumulation numerics, which is a training-math decision, not a memory fix.

### 5. `start_iteration` param (feature, not a bug fix)
Added to `__init__` (default `0`) and used in `fit()`:
```python
iteration, epoch = self.start_iteration, 0  # was: iteration, epoch = 0, 0
```
Without this, `--checkpoint` correctly restores model *weights* but `fit()`
always restarts its internal iteration counter (and therefore the ES seed
schedule, since seeds are `global_seed + iteration`) at 0 — so resuming a
run without this would silently replay the exact same population-perturbation
seeds already used, and mislabel iteration numbers in logs/W&B. Verified with
a smoke test (`start_iteration=2, n_iterations=3` correctly ran only
iterations 2 and 3, not 0-3).

## Code changes in `es_at_scale/train.py`
Added `--start-iteration` CLI flag (default `0`), wired through to the
trainer constructor. Use with `--checkpoint` to continue a shorter run into a
longer one — see the example in `/workspace/PROGRESS.md`.

## Other environment gotchas hit along the way
- **pip install failures that looked like network timeouts were actually disk
  space.** `TMPDIR`/pip's cache default to the container's root overlay, which
  is only 8GB (vs `/workspace`'s 256GB persistent volume). Downloading
  `vllm==0.11.0`'s dependencies (torch + full CUDA library stack, several GB)
  filled the root disk mid-download, which pip reported as connection/timeout
  errors before finally surfacing "No space left on device." Fixed by setting
  `TMPDIR=/workspace/tmp` and `PIP_CACHE_DIR=/workspace/.pip_cache` before any
  pip install into `.venv`. If installing more packages later, keep using
  those env vars (or just always `export` them at the top of the shell).
- `HF_HOME=/workspace/.hf_home` is now persisted in `/workspace/.env`, so it
  auto-loads in new shells/sessions — no need to `export` it manually anymore.
