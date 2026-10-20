"""Semantics that the 2.8.0 performance rewrites must keep."""
import numpy as np
import pytest

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


def test_mutde_refills_the_donor_pool_when_donors_run_out():
    np.random.seed(2)
    # 3 rows, 4 random donors (two pairs): only 2 candidates besides the target, so the pool is
    # refilled with every row (the target included, as in geatpy 2.7.0), which must not hang or fail
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


def _front(rs, n, m):
    x = rs.rand(n, m)
    return x / np.linalg.norm(x, axis=1, keepdims=True) + 0.05 * rs.rand(n, m)


def _parallel_cases():
    """Deterministic operators on inputs large enough to take their parallel paths."""
    rs = np.random.RandomState(0)
    o3, o5 = _front(rs, 6000, 3), _front(rs, 1500, 5)
    dup = np.round(rs.rand(5000, 4) * 5) / 5
    cv = rs.rand(6000, 2) - 0.8
    lv, cri = ea.ndsortESS(o3[:3000], 1500)
    ref = ea.crtup(3, 1500)[0]
    yield 'ndsortESS', lambda: ea.ndsortESS(o3)
    yield 'ndsortESS_need', lambda: ea.ndsortESS(o5, 700)
    yield 'ndsortTNS_cv', lambda: ea.ndsortTNS(o3, 6000, None, cv, [1, -1, 1])
    yield 'ndsortDED_dup', lambda: ea.ndsortDED(dup)
    yield 'refselect', lambda: ea.refselect(o3[:3000], lv, cri, 1500, ref)
    yield 'refgselect', lambda: ea.refgselect(o3[:3000], ref, 2.0, cv[:3000])
    for metric in ('euclidean', 'cityblock', 'cosine', 'cosine_similarity'):
        yield 'cdist_' + metric, lambda m=metric: ea.cdist(o3[:2000], o5[:, :3], m)
    yield 'GD', lambda: ea.indicator.GD(o3[:2000], ref)
    yield 'IGD', lambda: ea.indicator.IGD(o3[:2000], ref)
    yield 'Spacing', lambda: ea.indicator.Spacing(o3[:2000])


def _same(a, b):
    if isinstance(a, tuple):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray):
        return a.dtype == b.dtype and np.array_equal(a, b, equal_nan=True)
    return type(a) is type(b) and (a == b or (a != a and b != b))


def test_parallel_paths_match_the_serial_ones_bit_for_bit():
    try:
        results = []
        for threads in (1, 4, 3):
            ea.set_num_threads(threads)
            assert ea.get_num_threads() == threads
            results.append({name: f() for name, f in _parallel_cases()})
    finally:
        ea.set_num_threads(None)
    for name in results[0]:
        assert _same(results[0][name], results[1][name]), name
        assert _same(results[0][name], results[2][name]), name


def test_set_num_threads_none_restores_the_default():
    default = ea.get_num_threads()
    assert default >= 1
    ea.set_num_threads(2)
    assert ea.get_num_threads() == 2
    ea.set_num_threads(None)
    assert ea.get_num_threads() == default
    ea.set_num_threads(0)
    assert ea.get_num_threads() == default


def _sort_in_child(seed):
    rs = np.random.RandomState(seed)
    return ea.ndsortESS(_front(rs, 6000, 3))[1]


@pytest.mark.filterwarnings('ignore:This process .* is multi-threaded:DeprecationWarning')
def test_parallel_operators_work_in_forked_children():
    import multiprocessing as mp
    if 'fork' not in mp.get_all_start_methods():
        return
    ea.set_num_threads(4)
    try:
        expected = [_sort_in_child(s) for s in range(3)]  # builds the pool in the parent
        with mp.get_context('fork').Pool(2) as pool:
            got = pool.map_async(_sort_in_child, range(3)).get(timeout=120)
    finally:
        ea.set_num_threads(None)
    assert got == expected


def test_fortran_ordered_inputs_give_the_same_results():
    rs = np.random.RandomState(0)
    a, b = np.asfortranarray(rs.rand(500, 3)), rs.rand(3, 300).T  # both Fortran ordered
    for metric in ('euclidean', 'cityblock', 'cosine', 'cosine_similarity'):
        expected = ea.cdist(np.ascontiguousarray(a), np.ascontiguousarray(b), metric)
        assert np.array_equal(ea.cdist(a, b, metric), expected)
    for chrom in (rs.randint(0, 2, (50, 40)), rs.randint(0, 2, (50, 40)).astype(float)):
        np.random.seed(1)
        got = ea.mutbin('BG', np.asfortranarray(chrom))
        np.random.seed(1)
        assert np.array_equal(got, ea.mutbin('BG', np.ascontiguousarray(chrom)))
