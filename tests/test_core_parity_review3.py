# -*- coding: utf-8 -*-
"""Regression tests for NaN (and Inf) inputs to the sorting operators and refselect.

Rust's standard sort panics (PanicException) when a comparator is not a total order, which the old
`partial_cmp(..).unwrap_or(Equal)` comparators were not once NaN appeared. The expected values below
come from the original geatpy 2.7.0 binaries (macOS arm64 wheel, see scripts/parity/fuzz/README.md),
so the file can be run against both cores.
"""
import importlib.util

import numpy as np
import pytest

import geatpy as ea

nan, inf = np.nan, np.inf
NDSORT = [ea.ndsortESS, ea.ndsortTNS, ea.ndsortDED]
PERM = np.array([np.random.RandomState(k).permutation(6) for k in range(4)], float)


def _with(a, value):
    a = a.copy()
    a[1, 2] = value
    return a


@pytest.mark.parametrize('ndsort', NDSORT)
def test_ndsort_never_panics_on_nan(ndsort):
    rs = np.random.RandomState(1)
    for _ in range(200):
        o = rs.rand(40, 3)
        o[rs.rand(40, 3) < 0.1] = nan
        levels, max_level = ndsort(o)
        assert levels.shape == (40,)
        if ndsort is not ea.ndsortDED:  # the original ndsortDED mislabels fronts (see AGENTS.md)
            assert np.isfinite(levels).all() and levels.max() == max_level


# Every row keeps its own unique entry when it holds NaN; numpy sorts NaN after every number.
@pytest.mark.parametrize('ndsort', [ea.ndsortESS, ea.ndsortTNS])
@pytest.mark.parametrize('objv, cv, expected', [
    ([[nan], [2.], [1.]], None, ([3., 2., 1.], 3)),
    ([[nan], [0.3], [nan], [0.1], [0.3]], None, ([3., 2., 4., 1., 2.], 4)),
    ([[nan, 0.], [nan, 0.], [nan, 1.]], None, ([1., 2., 3.], 3)),
    ([[nan, nan], [nan, nan], [0., 1.], [1., 0.]], None, ([1., 1., 1., 1.], 1)),
    ([[nan, 1, 2], [nan, 1, 2], [0., 3, 3], [nan, 1, 2]], None, ([1., 2., 1., 3.], 3)),
    ([[nan, 2., 2.], [2., 0., 0.], [0., nan, 2.]], None, ([2., 1., 1.], 2)),
    ([[inf, 0.], [0., inf], [1., 1.], [-inf, 2.]], None, ([1., 2., 1., 1.], 2)),
    # constraints: the column maximum starts from the first row, so a NaN there is the maximum
    ([[nan], [nan], [2.]], [[0.], [1.], [0.]], ([2., 3., 1.], 3)),
    ([[nan, nan], [1., 0.], [0., 0.]], [[0.], [0.], [0.]], ([1., 2., 1.], 2)),
    ([[1., 0., nan], [nan, 2., 0.], [nan, 1., 2.]], [[0.], [1.], [0.]], ([1., 1., 1.], 1)),
    ([[nan, 1.], [0.5, 0.5], [1., 0.], [0.2, 0.9]], [[0.], [0.], [1.], [0.]], ([2., 1., 3., 1.], 3)),
    ([[0.2, 1.], [nan, 0.5], [1., 0.], [0.5, 0.9]], [[0.], [0.], [1.], [0.]], ([1., 1., 2., 1.], 2)),
])
def test_ndsort_nan_semantics(ndsort, objv, cv, expected):
    cv = None if cv is None else np.array(cv)
    levels, max_level = ndsort(np.array(objv), None, None, cv)
    assert (levels.tolist(), max_level) == expected


def test_ndsort_two_objectives_with_nan_ens_and_tens_differ():
    # ENS compares a row with the nearest earlier member of the front only; T-ENS walks its tree
    objv = np.array([[0.95, 0.63], [0.7, 0.11], [0.84, nan]])
    levels, max_level = ea.ndsortESS(objv)
    assert (levels.tolist(), max_level) == ([1., 1., 1.], 1)
    levels, max_level = ea.ndsortTNS(objv)
    assert (levels.tolist(), max_level) == ([2., 1., 1.], 2)


@pytest.mark.parametrize('objv, expected', [
    ([[0.1, 0.5], [0.2, nan], [0.3, 0.2], [0.4, 0.1], [nan, 0.3], [0.6, 0.05]], [nan] * 6),
    ([[0.1], [0.2], [nan], [0.4], [0.5]], [inf, nan, inf, nan, nan]),
    ([[0.1, 0.5], [0.2, inf], [0.3, 0.2], [0.4, 0.1], [inf, 0.3], [0.6, 0.05]],
     [nan, inf, 0., 0., inf, nan]),
])
def test_crowdis_with_nan_and_inf(objv, expected):
    objv = np.array(objv)
    np.testing.assert_array_equal(ea.crowdis(objv, np.ones(len(objv))), expected)


def test_selection_puts_nan_fitness_last_and_ecs_takes_the_first_nan():
    assert sorted(ea.dup(np.array([[0.3], [nan], [0.9], [0.1]]), 3).tolist()) == [0, 2, 3]
    assert ea.ecs(np.array([[0.3], [nan], [0.9], [nan]]), 2).tolist() == [1, 1]  # np.argmax
    assert ea.ecs(np.array([[0.3], [0.9], [0.9]]), 2).tolist() == [1, 1]


def test_migrate_and_refselect_accept_nan():
    fitn = [np.array([[0.1], [nan], [0.5], [0.3], [nan], [0.2]]),
            np.array([[nan], [0.4], [0.6], [0.1], [0.7], [0.2]])]
    for _ in range(50):
        ea.migrate([6, 6], 0.5, 1, 1, 0, fitn, False)
    rs = np.random.RandomState(0)
    for _ in range(50):
        o = rs.rand(40, 3)
        o[rs.rand(40, 3) < 0.1] = nan
        flag = ea.refselect(o, np.r_[np.ones(25), 2 * np.ones(15)], 2, 30, ea.crtup(3, 20)[0])
        assert flag.dtype == bool and flag.sum() == 30


@pytest.mark.parametrize('xov', [ea.xovpmx, ea.xovox])
@pytest.mark.parametrize('value, error, message', [
    (nan, ValueError, 'cannot convert float NaN to integer'),
    (inf, OverflowError, 'cannot convert float infinity to integer'),
    (-inf, OverflowError, 'cannot convert float infinity to integer'),
])
def test_permutation_crossover_rejects_nan_and_inf(xov, value, error, message):
    with pytest.raises(error, match=message):
        xov(_with(PERM, value))


def test_mutpp_rejects_nan():
    with pytest.raises(RuntimeError, match='Chromosome error'):
        ea.mutpp('P', _with(PERM, nan), np.array([[0] * 6, [5] * 6, [1] * 6]))


@pytest.mark.parametrize('xov', [ea.xovdp, ea.xovsp, ea.xovud])
@pytest.mark.parametrize('gene_id, as_int', [
    ([0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5], [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]),
    ([-.5, -.5, .5, .5, 1.5, 1.5], [0, 0, 0, 0, 1, 1]),
    # numpy casts NaN to int32 as 0 on arm64 (as INT_MIN on x86_64, where the original differs)
    ([0, 0, 1, 1, 2, 2, nan, nan, 4, 4, 5, 5], [0, 0, 1, 1, 2, 2, 0, 0, 4, 4, 5, 5]),
])
def test_gene_id_is_cast_to_int32(xov, gene_id, as_int):
    chrom = np.random.RandomState(0).rand(6, len(gene_id))
    for seed in range(20):
        np.random.seed(seed)
        a = xov(chrom, 1.0, False, np.array(gene_id, float))
        np.random.seed(seed)
        b = xov(chrom, 1.0, False, np.array(as_int, float))
        np.testing.assert_array_equal(a, b)


# refselect normalises by the hyperplane through the extreme points, solved like geatpy 2.7.0 with
# numpy (`det == 0` falls back to the column maxima). With NaN objectives the outcome depends on how
# numpy's OpenBLAS picks LU pivots: a NaN never stays the running maximum of `idamax`, so an exact
# zero can become the pivot (det == 0); past 8 rows a block holding NaN wins outright.
def _refselect_bench(seed, with_nan):
    rs = np.random.RandomState(seed)
    o = rs.rand(40, 3)
    if with_nan:
        o[rs.rand(40, 3) < 0.1] = nan
    return np.where(ea.refselect(o, np.r_[np.ones(25), 2 * np.ones(15)], 2, 30, ea.crtup(3, 20)[0]))[0].tolist()


_FIRST_25 = list(range(25))


@pytest.mark.parametrize('seed, chosen', [
    (15, [25, 26, 29, 33, 36]), (61, [26, 29, 31, 33, 35]), (64, [26, 27, 28, 31, 35]),
    (68, [25, 26, 27, 28, 29]),  # det is NaN: numpy solves to NaN intercepts
    (77, [25, 29, 30, 31, 34]), (143, [27, 32, 33, 34, 39]), (175, [26, 30, 35, 37, 39]),
    (178, [25, 27, 35, 36, 37]), (194, [25, 26, 27, 34, 35]),
])
def test_refselect_nan_objectives_follow_numpy_lu(seed, chosen):
    assert _refselect_bench(seed, True) == _FIRST_25 + chosen


def _refselect_fuzz_case(with_nan, t):
    # the generator of the differential run these cases come from (scratch fuzz, seeds 10000 + t)
    rs = np.random.RandomState(10000 + t)
    m = int(rs.choice([2, 3, 3, 3, 4, 5, 8, 10, 12]))
    n = int(rs.randint(6, 80))
    style = t % 5
    if style == 0:
        o = rs.rand(n, m)
    elif style == 1:
        o = rs.randint(0, 6, (n, m)).astype(float)
    elif style == 2:
        o = rs.rand(n, m)
        k = rs.randint(1, n)
        o[rs.randint(0, n, k)] = o[rs.randint(0, n, k)]
    elif style == 3:
        o = np.round(rs.rand(n, m) * 4) / 4 * rs.choice([1, 100, 1e-3])
    else:
        o = rs.rand(n, m)
        o[rs.rand(n, m) < 0.15] = 0.0
    if with_nan:
        o[rs.rand(n, m) < rs.choice([0.02, 0.1, 0.3])] = nan
    if rs.rand() < 0.2:
        o[rs.rand(n, m) < 0.05] = rs.choice([inf, -inf]) if with_nan else o.max()
    n1 = int(rs.randint(0, n - 1))
    n2 = int(rs.randint(2, n - n1 + 1))
    levels = np.r_[np.ones(n1), 2 * np.ones(n2), 3 * np.ones(n - n1 - n2)]
    need = int(rs.randint(n1 + 1, n1 + n2))
    z = ea.crtup(m, {2: 12, 3: 20, 4: 30, 5: 40, 8: 60, 10: 80, 12: 100}[m])[0]
    return o, levels, need, z


@pytest.mark.parametrize('with_nan, t, chosen', [
    # integer objectives: ties in 1 - cos^2 break like the original's normalise-then-matmul rounding
    (False, 14588, [0]), (False, 3558, [0, 2]), (False, 1665, [0, 2, 4]),
    (False, 13510, [0, 1, 2, 4]),  # 10 objectives, continuous
    # NaN with 10-12 objectives: OpenBLAS idamax compares blocks of 8 by a NaN-propagating fmax
    (True, 7214, [0, 1, 2, 3, 4, 5, 6]), (True, 18916, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 17]),
    # det == 0 with a NaN in the first row: the column "maximum" starts from it and stays NaN
    (True, 6717, [0, 1, 2, 3]), (True, 16331, [0, 4, 5, 7, 12]),
])
def test_refselect_fuzz_cases_match_original(with_nan, t, chosen):
    o, levels, need, z = _refselect_fuzz_case(with_nan, t)
    assert np.where(ea.refselect(o, levels, 2, need, z))[0].tolist() == chosen


RUST_CORE = importlib.util.find_spec('_geatpy_core') is not None


@pytest.mark.skipif(not RUST_CORE, reason='intentional difference from geatpy 2.7.0 (see AGENTS.md)')
@pytest.mark.parametrize('seed, chosen, original', [
    (131, [29, 31, 32, 34, 35], [25, 26, 27, 28, 31]),
    (145, [26, 31, 33, 37, 38], [25, 26, 31, 33, 36]),
    (194, [26, 31, 33, 36, 37], [25, 26, 27, 28, 37]),
])
def test_refselect_repeated_extreme_point_falls_back_to_column_maxima(seed, chosen, original):
    # One individual is the extreme point of two objectives, so the extreme-point matrix is exactly
    # singular. numpy's LU leaves a ~1e-17 pivot (det ~ 1e-19 != 0) and the original solves it into
    # meaningless (even negative) intercepts; the Rust core detects the zero pivot and uses the
    # column maxima, as the reference algorithm intends.
    del original
    assert _refselect_bench(seed, False) == _FIRST_25 + chosen
