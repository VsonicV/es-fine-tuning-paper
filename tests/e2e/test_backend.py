"""End-to-end backend integration test (real engines, real GPUs).

Exercises the full ESBackend contract against a live engine and validates the
correctness guardrails that matter for ES:
  * generate works and is deterministic (greedy);
  * perturb changes weights, restore returns EXACTLY to baseline output
    (this is also the radix-cache-off regression: stale KV would break it);
  * all engines are bit-identical at start and STAY identical after a committed
    ES update (validates the engine-0-update + NCCL broadcast on real GPUs);
  * save -> perturb -> load round-trips back to a consistent state.

Run (from repo root, in the matching env):
  vLLM  :  python tests/e2e/test_backend.py --backend vllm   --gpus 4,5
  SGLang:  ES_SGLANG_PYTHON=/path/to/sglang-es/python \
           python tests/e2e/test_backend.py --backend sglang --gpus 6,7
"""

import argparse
import os
import sys
import tempfile

from es_at_scale.backends import SamplingConfig, get_backend

PROMPT = ["The capital of France is"]


def build(backend, model, gpus):
    if backend == "vllm":
        return get_backend(
            "vllm", model_name=model, n_engines=len(gpus),
            n_gpu_per_engine=1, use_gpus=",".join(str(g) for g in gpus),
        )
    return get_backend(
        "sglang", model_name=model, n_engines=len(gpus), gpus=gpus,
        n_gpu_per_engine=1,
    )


def gen_text(b, engine_idx, sampling):
    out = b.wait([b.generate_async(engine_idx, PROMPT, sampling)])[0][0]
    return out.outputs[0].text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["vllm", "sglang"], required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--gpus", default="4,5")
    args = ap.parse_args()
    gpus = [int(x) for x in args.gpus.split(",") if x != ""]

    sampling = SamplingConfig(n=1, temperature=0.0, top_p=1.0, max_tokens=32, seed=0)
    b = build(args.backend, args.model, gpus)
    b.start()
    try:
        # 1) engines bit-identical right after load, BEFORE any perturbation.
        #    (After a bare perturb/restore, vLLM's subtract-restore leaves tiny
        #    bf16 drift on the touched engine until the next update re-broadcasts;
        #    that transient is expected, so we check identity here and post-update.)
        cks0 = b.checksums()
        if all(c is not None for c in cks0):
            assert len(set(cks0)) == 1, f"engines differ after load: {cks0}"
            print(f"[ok] {len(cks0)} engines bit-identical after load")

        # 2) baseline greedy generation on engine 0
        base_text = gen_text(b, 0, sampling)
        print(f"[info] baseline: {base_text!r}")

        # 3) perturb -> (informational) generation may change
        b.wait([b.perturb_async(0, 12345, 0.01)])
        pert_text = gen_text(b, 0, sampling)
        print(f"[info] perturbed: {pert_text!r} (changed={pert_text != base_text})")

        # 4) restore -> MUST return exactly to baseline output
        b.wait([b.restore_async(0, 12345, 0.01)])
        rest_text = gen_text(b, 0, sampling)
        assert rest_text == base_text, (
            f"restore did not reproduce baseline output:\n base={base_text!r}\n rest={rest_text!r}"
        )
        print("[ok] perturb/restore round-trip reproduces baseline (radix-cache-off holds)")

        # 5) committed ES update keeps every engine identical
        seeds = [1, 2, 3, 4]
        coeffs = [0.5, -0.5, 1.0, -1.0]
        b.sync_after_update(seeds, coeffs, alpha=0.01, population_size=8)
        cks2 = b.checksums()
        if all(c is not None for c in cks2):
            assert len(set(cks2)) == 1, f"engines drifted after update: {cks2}"
            print("[ok] engines stay bit-identical after committed update")

        # 6) save -> perturb -> load restores a consistent state across engines
        tmp = tempfile.mktemp(suffix=".pth")
        try:
            b.save(tmp)
            b.wait([b.perturb_async(0, 999, 0.05)])
            b.load(tmp)
            cks3 = b.checksums()
            if all(c is not None for c in cks3):
                assert len(set(cks3)) == 1, f"engines inconsistent after load: {cks3}"
            print("[ok] save/perturb/load round-trip consistent")
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

        print(f"\nBACKEND E2E PASSED ({args.backend})")
    finally:
        b.shutdown()


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"FAILED: {e}")
        sys.exit(1)
