"""Tier-A deterministic correctness test for the shared ES math (es_ops).

Backend-free, CPU-only, fast. Run: ``python tests/test_es_ops.py``.
Validates the perturb / restore / apply_update / snapshot / checksum primitives
against hand-computed references so the extraction from the original
worker_extension is provably behavior-preserving.
"""

import sys

import torch

from es_at_scale.backends import es_ops


def make_model(seed=0):
    torch.manual_seed(seed)
    m = torch.nn.Sequential(
        torch.nn.Linear(16, 8),
        torch.nn.Linear(8, 4),
    ).to(torch.float32)
    return m


def params_fn_of(m):
    return lambda: m.named_parameters()


def clone_state(m):
    return {k: v.detach().clone() for k, v in m.named_parameters()}


def test_perturb_reproducible_and_antithetic():
    m = make_model()
    pf = params_fn_of(m)
    base = clone_state(m)

    # same seed + scale -> identical perturbation (bit-exact, from the same base)
    es_ops.perturb(pf, seed=1234, noise_scale=0.01)
    after_a = clone_state(m)
    es_ops.restore_from_snapshot(pf, base)
    es_ops.perturb(pf, seed=1234, noise_scale=0.01)
    after_b = clone_state(m)
    es_ops.restore_from_snapshot(pf, base)
    for k in after_a:
        assert torch.equal(after_a[k], after_b[k]), f"perturb not reproducible for {k}"

    # antithetic: around a ZERO base, perturb(+/-) is exactly +/- the signed noise
    # (nonzero base would round asymmetrically -- that is the documented restore drift).
    for p in m.parameters():
        p.data.zero_()
    zero = clone_state(m)
    es_ops.perturb(pf, seed=1234, noise_scale=0.01, negate=False)
    pos = clone_state(m)
    es_ops.restore_from_snapshot(pf, zero)
    es_ops.perturb(pf, seed=1234, noise_scale=0.01, negate=True)
    for k, v in m.named_parameters():
        assert torch.equal(v.detach(), -pos[k]), f"negate mismatch {k}"
    es_ops.restore_from_snapshot(pf, zero)
    print("[ok] perturb reproducible + antithetic")


def test_snapshot_restore_bit_exact():
    m = make_model()
    pf = params_fn_of(m)
    base = es_ops.snapshot(pf)
    es_ops.perturb(pf, seed=7, noise_scale=0.05)
    es_ops.restore_from_snapshot(pf, base)
    for k, v in m.named_parameters():
        assert torch.equal(v.detach(), base[k]), f"restore not bit-exact for {k}"
    print("[ok] snapshot restore is bit-exact")


def test_apply_update_matches_reference():
    m = make_model()
    pf = params_fn_of(m)
    base = es_ops.snapshot(pf)

    seeds = [11, 22, 33, 44]
    coeffs = [0.5, -1.5, 2.0, 0.0]
    alpha = 0.02
    N = 8  # population_size (deliberately != len(seeds), as in the real algorithm)

    # reference: theta + (alpha/N) * sum_i coeff_i * randn(seed_i), fp32 accumulate
    ref = {}
    for name, p in m.named_parameters():
        acc = torch.zeros_like(p.data, dtype=torch.float32)
        for i, s in enumerate(seeds):
            g = torch.Generator(device=p.device).manual_seed(int(s))
            noise = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=g)
            acc += noise.to(torch.float32) * coeffs[i]
        acc *= alpha / N
        ref[name] = (base[name] + acc.to(p.dtype)).clone()

    es_ops.apply_update(pf, seeds, coeffs, alpha, N)
    for k, v in m.named_parameters():
        assert torch.equal(v.detach(), ref[k]), f"apply_update mismatch for {k}"
    print("[ok] apply_update matches hand-computed reference")


def test_checksum_detects_change():
    m = make_model()
    pf = params_fn_of(m)
    c0 = es_ops.checksum(pf)
    c1 = es_ops.checksum(pf)
    assert c0 == c1, "checksum not stable"
    es_ops.perturb(pf, seed=99, noise_scale=1e-3)
    c2 = es_ops.checksum(pf)
    assert c2 != c0, "checksum failed to detect a change"
    print("[ok] checksum stable + change-detecting")


def test_apply_update_is_deterministic():
    # apply_update is a pure deterministic function of (base, seeds, coeffs): two
    # independent models with the same base + same update end bit-identical.
    a, b = make_model(), make_model()
    pa, pb = params_fn_of(a), params_fn_of(b)
    # sync b to a's weights first
    base = es_ops.snapshot(pa)
    es_ops.restore_from_snapshot(pb, base)
    assert es_ops.checksum(pa) == es_ops.checksum(pb)

    seeds, coeffs = [1, 2, 3], [0.3, -0.7, 1.1]
    es_ops.apply_update(pa, seeds, coeffs, 0.01, 5)
    es_ops.apply_update(pb, seeds, coeffs, 0.01, 5)
    assert es_ops.checksum(pa) == es_ops.checksum(pb), "engines drifted after identical update"
    for (ka, va), (kb, vb) in zip(a.named_parameters(), b.named_parameters()):
        assert torch.equal(va.detach(), vb.detach()), f"lockstep broken for {ka}"
    print("[ok] independent engines stay bit-identical under the same update")


def main():
    tests = [
        test_perturb_reproducible_and_antithetic,
        test_snapshot_restore_bit_exact,
        test_apply_update_matches_reference,
        test_checksum_detects_change,
        test_apply_update_is_deterministic,
    ]
    for t in tests:
        t()
    print("\nALL es_ops TESTS PASSED")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"FAILED: {e}")
        sys.exit(1)
