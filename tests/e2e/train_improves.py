"""Tier-B behavioral e2e: a short REAL training run must not diverge and should
trend upward, on either backend. ES is noisy, so this asserts non-divergence
(second-half mean >= first-half mean - tol) rather than a strict monotone gate;
the deterministic Tier-A tests carry the correctness weight.

Runs ``es_at_scale/train.py`` as a subprocess and parses the per-iteration
"Mean reward:" lines.

  python tests/e2e/train_improves.py --backend vllm   --gpus 4,5
  ES_SGLANG_PYTHON=.../sglang-es/bin/python \
  python tests/e2e/train_improves.py --backend sglang --gpus 6,7
"""

import argparse
import os
import re
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["vllm", "sglang"], required=True)
    ap.add_argument("--gpus", default="4,5")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--pop", type=int, default=8)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--tol", type=float, default=0.02)
    args = ap.parse_args()

    n_engines = len([g for g in args.gpus.split(",") if g != ""])
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cmd = [
        args.python, "es_at_scale/train.py",
        "--backend", args.backend,
        "--task", "countdown",
        "--model-name", args.model,
        "--use-gpus", args.gpus,
        "--n-vllm-engines", str(n_engines),
        "--n-iterations", str(args.iters),
        "--population-size", str(args.pop),
        "--batch-size", "32", "--mini-batch-size", "32",
        "--max-tokens", "256", "--sigma", "0.001",
        "--eval-freq", "100000", "--logging", "none",
        "--train-dataset", "datasets/train/countdown",
        "--eval-dataset", "",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = repo + os.pathsep + env.get("PYTHONPATH", "")
    print("[info] running:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=repo, env=env, capture_output=True, text=True)
    sys.stdout.write(proc.stdout[-4000:])
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-4000:])
        print(f"FAILED: train.py exited {proc.returncode}")
        sys.exit(1)

    rewards = [float(x) for x in re.findall(r"Mean reward: ([-\d.eE+]+)", proc.stdout)]
    print(f"[info] reward trajectory ({len(rewards)}): {rewards}")
    if len(rewards) < 4:
        print("FAILED: too few reward points parsed")
        sys.exit(1)
    if not all(r == r for r in rewards):  # NaN check
        print("FAILED: non-finite reward encountered")
        sys.exit(1)
    half = len(rewards) // 2
    first = sum(rewards[:half]) / half
    second = sum(rewards[half:]) / (len(rewards) - half)
    print(f"[info] first-half mean={first:.4f}  second-half mean={second:.4f}")
    if second < first - args.tol:
        print(f"FAILED: training diverged (second-half {second:.4f} << first-half {first:.4f})")
        sys.exit(1)
    print(f"\nTRAIN-IMPROVES PASSED ({args.backend}): non-divergent, trend {second - first:+.4f}")


if __name__ == "__main__":
    main()
