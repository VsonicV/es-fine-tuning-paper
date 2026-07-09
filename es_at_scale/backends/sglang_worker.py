"""SGLang ES worker: runs in the dedicated `sglang-es` conda env.

Hosts one ``ESEngine`` (an ``sglang.Engine`` whose scheduler children carry the
``es_*`` weight-op methods, see :mod:`es_at_scale.backends.sglang_shim`) and serves
a ZeroMQ REP loop. The trainer-side :class:`~es_at_scale.backends.sglang_backend.SGLangBackend`
drives it with tiny messages (seeds, sampling, prompts); weight noise never leaves
the GPU. One worker == one engine == one GPU (TP=1 by default).

Launched via ``python -m es_at_scale.backends.sglang_worker ...`` with
``CUDA_VISIBLE_DEVICES`` pinned by the parent so the engine sees its GPU as cuda:0.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import pickle
import signal
import sys
import tempfile
import traceback

import zmq

from es_at_scale.backends.sglang_shim import make_es_engine


def _die_with_parent():
    """Linux: receive SIGKILL if the parent (trainer) dies, so we never orphan a
    GPU-holding worker. (SGLang's own kill-on-parent-death covers only the
    scheduler child, not this REP-loop process.)"""
    try:
        PR_SET_PDEATHSIG = 1
        ctypes.CDLL("libc.so.6").prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
    except Exception:
        pass


def _reshape_generate(prompts, results):
    if isinstance(results, dict):
        results = [results]
    outputs = []
    for prompt, d in zip(prompts, results):
        token_ids = d.get("output_ids")
        comp = {"text": d.get("text", ""), "token_ids": list(token_ids or [])}
        outputs.append({"prompt": prompt, "completions": [comp]})
    return outputs


def _handle(engine, msg):
    op = msg["op"]
    if op == "ping":
        return {"ok": True}
    if op == "perturb":
        engine.collective_rpc(
            "es_perturb", seed=int(msg["seed"]), sigma=float(msg["sigma"]),
            negate=bool(msg.get("negate", False)),
        )
        return {"ok": True}
    if op == "restore":
        engine.collective_rpc("es_restore", seed=int(msg["seed"]), sigma=float(msg["sigma"]))
        return {"ok": True}
    if op == "apply_update":
        engine.collective_rpc(
            "es_apply_update", seeds=[int(s) for s in msg["seeds"]],
            coeffs=[float(c) for c in msg["coeffs"]], alpha=float(msg["alpha"]),
            population_size=int(msg["population_size"]),
        )
        return {"ok": True}
    if op == "init_broadcast_group":
        engine.collective_rpc(
            "es_init_broadcast_group", master_address=msg["master_address"],
            master_port=int(msg["master_port"]), rank=int(msg["rank"]),
            world_size=int(msg["world_size"]),
        )
        return {"ok": True}
    if op == "broadcast_weights":
        engine.collective_rpc("es_broadcast_weights", src_rank=int(msg["src"]))
        return {"ok": True}
    if op == "generate":
        s = msg["sampling"]
        if int(s.get("n", 1)) != 1:
            raise NotImplementedError("SGLang backend supports n=1 sampling only")
        sp = {
            "temperature": float(s["temperature"]),
            "top_p": float(s["top_p"]),
            "max_new_tokens": int(s["max_tokens"]),
            "n": 1,
        }
        results = engine.generate(prompt=list(msg["prompts"]), sampling_params=sp)
        return {"ok": True, "outputs": _reshape_generate(list(msg["prompts"]), results)}
    if op == "checksum":
        with tempfile.NamedTemporaryFile("r", suffix=".json", delete=False) as f:
            path = f.name
        try:
            engine.collective_rpc("es_checksum", out_path=path)
            import json
            with open(path) as fh:
                sig = json.load(fh)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        return {"ok": True, "checksum": sig}
    if op == "save":
        engine.collective_rpc("es_save", path=msg["path"])
        return {"ok": True}
    if op == "load":
        engine.collective_rpc("es_load", path=msg["path"])
        return {"ok": True}
    raise ValueError(f"unknown op: {op}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--tp-size", type=int, default=1)
    ap.add_argument("--mem-fraction-static", type=float, default=0.7)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--random-seed", type=int, default=42)
    ap.add_argument("--disable-cuda-graph", action="store_true")
    args = ap.parse_args()

    _die_with_parent()

    engine = make_es_engine(
        model_path=args.model,
        tp_size=args.tp_size,
        dp_size=1,
        mem_fraction_static=args.mem_fraction_static,
        dtype=args.dtype,
        random_seed=args.random_seed,
        skip_tokenizer_init=False,
        disable_radix_cache=True,       # guardrail: no cross-request KV reuse under changing weights
        disable_overlap_schedule=True,  # deterministic timing vs weight edits
        disable_cuda_graph=args.disable_cuda_graph,
        trust_remote_code=True,
    )

    def _shutdown(*_):
        try:
            engine.shutdown()
        finally:
            os._exit(0)

    signal.signal(signal.SIGTERM, _shutdown)

    # Startup self-test: verify the shim reached the child and the weight path resolves.
    with tempfile.NamedTemporaryFile("r", suffix=".json", delete=False) as f:
        st_path = f.name
    try:
        engine.collective_rpc("es_selftest", out_path=st_path)
        import json
        with open(st_path) as fh:
            info = json.load(fh)
        print(f"[sglang_worker] self-test OK: param_count={info['param_count']}", flush=True)
    finally:
        try:
            os.unlink(st_path)
        except OSError:
            pass

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(args.endpoint)
    print(f"[sglang_worker] ready on {args.endpoint}", flush=True)

    while True:
        raw = sock.recv()
        msg = pickle.loads(raw)
        if msg.get("op") == "shutdown":
            sock.send(pickle.dumps({"ok": True}))
            break
        try:
            reply = _handle(engine, msg)
        except Exception as e:  # structured error -> trainer raises, never hangs
            reply = {"ok": False, "error": f"{type(e).__name__}: {e}",
                     "traceback": traceback.format_exc()}
        reply["id"] = msg.get("id")
        sock.send(pickle.dumps(reply))

    try:
        engine.shutdown()
    except Exception:
        pass
    sys.exit(0)


if __name__ == "__main__":
    main()
