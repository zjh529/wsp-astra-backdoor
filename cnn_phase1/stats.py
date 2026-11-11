"""Wilson intervals, bootstrap means, exact paired tests, and Holm.

No scipy. n=10 sign-flips are enumerated. Binary paired tests use the
one-sided exact McNemar binomial on the discordant pairs.
"""
from __future__ import annotations

import math

import numpy as np

Z_95 = 1.959963984540054
N_BOOT = 10000
BOOT_SEED = 20261110
HOLM_ALPHA = 0.05
HOLM_FAMILY = (
    "tpr_real_gt_no_weights",
    "tpr_real_gt_random_init",
    "asr_astra_lt_random",
    "functional_backdoor_gt_clean",
)


def wilson_interval(successes: int, n: int, z: float = Z_95) -> dict:
    if n <= 0:
        return {"k": successes, "n": n, "estimate": None, "low": None, "high": None}
    estimate = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (estimate + z2 / (2.0 * n)) / denom
    margin = z * math.sqrt(estimate * (1.0 - estimate) / n + z2 / (4.0 * n * n)) / denom
    return {
        "k": successes,
        "n": n,
        "estimate": estimate,
        "low": max(0.0, center - margin),
        "high": min(1.0, center + margin),
    }


def bootstrap_mean_ci(values, *, seed: int = BOOT_SEED, n_boot: int = N_BOOT) -> dict:
    array = np.asarray(list(values), dtype=np.float64)
    n = int(array.size)
    if n == 0:
        return {"n": 0, "mean": None, "low": None, "high": None}
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, n, size=(n_boot, n))
    means = array[draw].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {"n": n, "mean": float(array.mean()), "low": float(low), "high": float(high)}


def mcnemar_greater(left, right) -> dict:
    """One-sided p that left is 1 more often than right. Ties are ignored."""
    a = np.asarray(list(left), dtype=int)
    b = np.asarray(list(right), dtype=int)
    if a.shape != b.shape:
        raise ValueError("paired arrays differ in length")
    only_left = int(np.sum((a == 1) & (b == 0)))
    only_right = int(np.sum((a == 0) & (b == 1)))
    discordant = only_left + only_right
    if discordant == 0:
        p_value = 1.0
    else:
        total = 0
        for k in range(only_left, discordant + 1):
            total += math.comb(discordant, k)
        p_value = total / (2**discordant)
    return {
        "n_pairs": int(a.size),
        "only_left": only_left,
        "only_right": only_right,
        "p": p_value,
    }


def signflip_p(values, *, alternative: str) -> dict:
    array = np.asarray(list(values), dtype=np.float64)
    n = int(array.size)
    if n == 0:
        return {"n": 0, "mean": None, "p": None}
    if n > 16:
        raise ValueError("sign-flip enumeration is for the phase-1 pair counts")
    observed = float(array.mean())
    count = 0
    total = 1 << n
    for mask in range(total):
        signed = 0.0
        for index in range(n):
            sign = -1.0 if (mask >> index) & 1 else 1.0
            signed += sign * array[index]
        mean = signed / n
        if alternative == "greater" and mean >= observed - 1e-12:
            count += 1
        elif alternative == "less" and mean <= observed + 1e-12:
            count += 1
    if alternative not in ("greater", "less"):
        raise ValueError(alternative)
    return {"n": n, "mean": observed, "p": count / total}


def holm(
    p_values: dict[str, float | None],
    *,
    alpha: float = HOLM_ALPHA,
    family: tuple[str, ...] | None = None,
) -> dict:
    order = HOLM_FAMILY if family is None else family
    named = [(name, p_values[name]) for name in order if name in p_values]
    testable = [(name, float(p)) for name, p in named if p is not None]
    ordered = sorted(testable, key=lambda item: (item[1], item[0]))
    m = len(ordered)
    decisions = {}
    still = True
    for rank, (name, p_value) in enumerate(ordered):
        threshold = alpha / (m - rank)
        reject = still and p_value <= threshold
        if not reject:
            still = False
        decisions[name] = {"p": p_value, "threshold": threshold, "reject": reject}
    for name, p_value in named:
        if p_value is None:
            decisions[name] = {"p": None, "threshold": None, "reject": False, "reason": "not tested"}
    return {"alpha": alpha, "family": list(order), "m": m, "tests": decisions}
