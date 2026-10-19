"""Semantics that the 2.8.0 performance rewrites must keep."""
import numpy as np

import geatpy as ea


def _donors(n, reps, pairs=1, rows=None):
    """Random donors of mutde, recovered from Xr0 = self, F = 1 and row k = 2**k.

    With these inputs every trial value is x_i + sum(2**b - 2**c), so each difference gives the
    pair of donors (b, c) exactly (powers of two up to 2**30 are exact in float64).
    """
    chrom = (2.0 ** np.arange(n)).reshape(-1, 1)
    field = np.array([[-2.0 ** 40], [2.0 ** 40], [0]])
    out = []
    for _ in range(reps):
        xr = [np.arange(n)] + [None] * (2 * pairs)
        if rows is None:
            trial = ea.mutde('RI', chrom, field, xr, 1.0)
        else:
            trial = ea.mutde('RI', chrom, field, xr, 1.0, 1, rows)
        out.append(trial[:, 0] - chrom[:len(trial), 0])
    return np.concatenate(out)


def _pair(diff):
    """(b, c) with 2**b - 2**c == diff, b != c."""
    v = int(diff)
    lo = (abs(v) & -abs(v)).bit_length() - 1  # trailing zeros
    hi = (abs(v) + (1 << lo)).bit_length() - 1
    return (hi, lo) if v > 0 else (lo, hi)


def test_mutde_donors_are_distinct_uniform_and_exclude_the_target():
    np.random.seed(1)
    n, reps = 8, 3000
    diffs = _donors(n, reps)
    target = np.tile(np.arange(n), reps)
    assert np.all(diffs != 0)  # the two difference donors are never the same row
    counts = np.zeros((n, n))
    for i, d in zip(target, diffs):
        b, c = _pair(d)
        assert b != i and c != i
        counts[i, b] += 1
    # every other row is equally likely: reps / (n - 1) ~ 428.6 draws each, sd ~ 19
    expected = reps / (n - 1)
    off = counts[~np.eye(n, dtype=bool)]
    assert np.all(np.abs(off - expected) < 6 * np.sqrt(expected))


def test_mutde_falls_back_to_any_row_when_donors_run_out():
    np.random.seed(2)
    # 3 rows, 4 random donors (two pairs): only 2 candidates besides the target, so the last
    # draws may be any row (including the target), which must not hang or fail
    out = _donors(3, 200, pairs=2)
    assert out.shape == (600,)
    assert np.all(np.isfinite(out))


def test_mutde_with_more_offspring_than_rows_draws_from_every_row():
    np.random.seed(3)
    n = 4
    # Mask_N as an integer > N: the target is not excluded and every row can be a donor
    chrom = (2.0 ** np.arange(n)).reshape(-1, 1)
    field = np.array([[-2.0 ** 40], [2.0 ** 40], [0]])
    seen = set()
    for _ in range(300):
        trial = ea.mutde('RI', chrom, field, [np.zeros(2 * n, dtype=int), None, None], 1.0, 1, 2 * n)
        assert trial.shape == (2 * n, 1)
        for d in trial[:, 0] - 1.0:
            b, c = _pair(d)
            seen.update((b, c))
    assert seen == set(range(n))
