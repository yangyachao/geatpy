# -*- coding: utf-8 -*-
"""Regression tests for the third parity review (agy cross-review + randomised differential testing).

Every assertion here also holds for the original geatpy 2.7.0 binaries (checked with its macOS
arm64 wheel, see scripts/parity/README.md), so the file can be run against both cores.
"""
import numpy as np
import pytest

import geatpy as ea

F6 = np.arange(6.).reshape(-1, 1)
PERM = np.array([np.random.RandomState(k).permutation(6) for k in range(4)])
FP = np.array([[0] * 6, [5] * 6, [1] * 6])
FDR = np.array([[0., 0, 0], [5, 5, 5], [0, 0, 0]])
CH = np.array([[1., 2, 3], [4, 0, 1]])
ROWS = np.repeat(np.arange(8).reshape(-1, 1), 6, axis=1)  # row i is all i: shows the gene sources


def _field_d_with_len(n):
    fd = ea.crtfld('BG', np.array([1]), np.array([[0], [7]])).copy()
    fd[0, 0] = n
    return fd


# --- integer outputs are int32 (except the numpy-based crtbp / urs / ecs / otos with Nsel = N) --
@pytest.mark.parametrize('call', [
    lambda: ea.tour(F6, 6), lambda: ea.etour(F6, 6), lambda: ea.rws(F6, 6), lambda: ea.sus(F6, 6, None),
    lambda: ea.dup(F6, 6), lambda: ea.rcs(F6), lambda: ea.rps(F6, 4), lambda: ea.otos(F6, 3),
    lambda: ea.selecting('tour', F6, 6), lambda: ea.xovsp(ROWS), lambda: ea.xovud(ROWS, 1, 4),
    lambda: ea.xovpmx(PERM), lambda: ea.xovox(PERM), lambda: ea.mutinv('P', PERM, FP),
    lambda: ea.mutpp('P', PERM, FP), lambda: ea.mutbin('BG', (PERM > 2).astype(int)),
    lambda: ea.crtip(3, np.array([[0, 0], [5, 5], [1, 1]])), lambda: ea.crtpp(3, FP),
    lambda: ea.bs2int(np.array([[1, 0, 1]]), ea.crtfld('BG', np.array([1]), np.array([[0], [7]]))),
    lambda: ea.ri2bs(np.array([[3.]]), ea.crtfld('BG', np.array([1]), np.array([[0], [7]]))),
])
def test_integer_outputs_are_int32(call):
    assert call().dtype == np.int32


@pytest.mark.parametrize('call', [
    lambda: ea.crtbp(3, 4), lambda: ea.urs(F6, 6), lambda: ea.ecs(F6, 6), lambda: ea.otos(F6, 6),
    lambda: ea.selecting('ecs', F6, 6),
])
def test_numpy_based_integer_outputs_are_int64(call):
    assert call().dtype == np.int64


def test_mutde_mask_output_is_int32_for_integer_variables():
    fdr = np.array([[0., 0], [9, 9], [1, 1]])
    out = ea.mutde('RI', np.array([[1., 2], [3, 4], [5, 6]]), fdr, [np.arange(3)] * 3, 0.5, 1,
                   np.array([[True, False], [True, True], [False, True]]))
    assert out.dtype == np.int32 and out.shape == (4,)


# --- behaviour --------------------------------------------------------------------------------
def test_otos_with_nsel_equal_to_n_returns_arange():
    for _ in range(20):
        assert ea.otos(np.array([[3.], [1], [2], [5]]), 4).tolist() == [0, 1, 2, 3]


def test_refgselect_keeps_the_individual_at_the_ideal_point():
    # [2.3, 0.8] is best on both objectives: it becomes a zero vector after translation
    flag, _ = ea.refgselect(np.array([[2.5, 1.5], [2.3, 0.8]]), ea.crtup(2, 10)[0], 2.0)
    assert flag.tolist() == [True, True]


HV_O = np.array([[0., 1.], [0.2, 0.1], [0.4, -0.5], [0.8, -0.7]])


@pytest.mark.parametrize('objv,pf,expected', [
    # negative objectives (e.g. ZDT3): the original shifts by min(0, column min) before scaling
    (HV_O, np.array([[0., 1.], [0.85, -0.77]]), 0.6553773283958435),
    (HV_O, None, 0.633838415145874),
    (HV_O - 1, None, 0.5656108856201172),
    (np.array([[0.5, 0.5, -0.2], [0.1, 0.9, 0.3]]), None, 0.04873780533671379),
])
def test_hv_normalises_negative_objectives_like_the_original(objv, pf, expected):
    assert abs(ea.indicator.HV(objv, pf) - expected) < 1e-6  # the original computes in float32


@pytest.mark.parametrize('name', ['recdis', 'xovud'])
def test_global_uniform_recombination_draws_genes_from_the_whole_population(name):
    out = getattr(ea, name)(ROWS, 1, 8)
    assert out.shape == (8, 6)
    # with whole-row copies every row would hold one value; per-gene sources mix several rows
    assert np.mean([len(set(r)) for r in out]) > 2


@pytest.mark.parametrize('dtype', [np.int32, np.int64, np.float64])
def test_mutbin_accepts_fortran_ordered_chromosomes(dtype):
    chrom = np.random.RandomState(0).randint(0, 2, (50, 40)).astype(dtype)
    np.random.seed(1)
    got = ea.mutbin('BG', np.asfortranarray(chrom))
    np.random.seed(1)
    assert np.array_equal(got, ea.mutbin('BG', np.ascontiguousarray(chrom)))


def test_xovud_default_probability_is_one_half():
    a, b = np.zeros((1, 400), dtype=int), np.ones((1, 400), dtype=int)
    changed = np.mean([ea.xovud(np.vstack([a, b]))[0].mean() for _ in range(50)])
    assert abs(changed - 0.5) < 0.03


def test_selecting_passes_the_tournament_size():
    picks = np.concatenate([ea.selecting('tour', F6, 6, 6) for _ in range(300)])
    assert picks.mean() > 4.2  # tournaments of 6 favour the best individual (index 5)


# --- validation ---------------------------------------------------------------------------------
@pytest.mark.parametrize('call', [
    lambda: ea.xovsp(ROWS, -0.1), lambda: ea.xovdp(ROWS, 1.5), lambda: ea.recint(CH, 2),
    lambda: ea.recsbx(CH, 0.7, False, -1), lambda: ea.recndx(CH, 0.7, False, -1),
    lambda: ea.recint(CH, 0.7, False, -1), lambda: ea.xovsp(ROWS, 0.7, 3.0),
    lambda: ea.xovpmx(PERM, 0.7, False, 3), lambda: ea.xovox(np.array([[0, 0, 1], [1, 2, 0]])),
    lambda: ea.xovpmx(np.array([[0, 0, 1], [1, 2, 0]])),
    lambda: ea.mutpp('P', np.array([[0, 0, 1, 2, 3, 4]]), FP),
    lambda: ea.crtbp(0, 5), lambda: ea.crtbp(5, 0), lambda: ea.crtbp(2.5, 5),
    lambda: ea.bs2ri(np.ones((2, 32), dtype=int), _field_d_with_len(32)),
    lambda: ea.ri2bs(np.array([[3.]]), _field_d_with_len(70)),
    lambda: ea.mutgau('RI', CH, FDR, 2), lambda: ea.mutgau('RI', CH, FDR, np.array([0.1, 1.2, 0.3])),
    lambda: ea.mutbga('RI', CH, FDR, 0.5, -0.1), lambda: ea.mutbga('RI', CH, FDR, 0.5, 0.5, 2.5),
    lambda: ea.mutbga('RI', CH, FDR, 0.5, 0.5, np.array([0, 3, 3])),
    lambda: ea.mutinv('P', PERM, FP, 1, 1), lambda: ea.mutinv('P', PERM, FP, 1, [1]),
    lambda: ea.mutinv('P', PERM, FP, 2), lambda: ea.mutmove('P', PERM, FP, 1, 0),
    lambda: ea.mutmove('P', PERM, FP, 1, 2, 2), lambda: ea.mutpp('P', PERM, FP, 1, 0),
    lambda: ea.mutde('RI', CH, FDR, [np.arange(2)] * 3, -0.5),
    lambda: ea.mutde('RI', CH, FDR, [np.arange(2)] * 3, np.array([0.5, 0.6])),
    lambda: ea.ranking(np.ones((3, 2))), lambda: ea.ranking(np.arange(4.)),
    lambda: ea.ranking(F6, None, None, None, 1.5), lambda: ea.ranking(F6, None, None, 0, 2.5),
    lambda: ea.ranking(F6, None, None, 1, 5), lambda: ea.scaling(F6, None, None, 1.5),
    lambda: ea.powing(F6, None, None, 0), lambda: ea.indexing(F6, None, None, -1),
    lambda: ea.otos(F6, 0), lambda: ea.migrate([10], 0.2), lambda: ea.migrate([10, 0], 0.2),
    lambda: ea.migrate([10, 10], 0), lambda: ea.migrate([10, 10], 0.2, 3),
    lambda: ea.migrate([10, 10], 0.2, 0, 2), lambda: ea.migrate([10, 10], 0.2, 0, 0, 3),
])
def test_invalid_arguments_raise_runtime_error(call):
    with pytest.raises(RuntimeError):
        call()


@pytest.mark.parametrize('call', [
    lambda: ea.mutswap('P', PERM, FP, 2),  # no Pm check in mutswap
    lambda: ea.mutinv('P', PERM, FP, 1, 7), lambda: ea.mutmove('P', PERM, FP, 1, 1),
    lambda: ea.mutde('RI', CH, FDR, [np.arange(2)] * 3, 0), lambda: ea.mutde('RI', CH, FDR, [np.arange(2)] * 3, [None, 0.5]),
    lambda: ea.mutbga('RI', CH, FDR, 0.5, 0.5, 0), lambda: ea.ranking(F6, None, None, 1, 4),
    lambda: ea.migrate([10, 10], 1), lambda: ea.xovox(np.array([[0, 1, 5], [1, 2, 0]])),
])
def test_valid_edge_arguments_are_accepted(call):
    call()


# --- reproducibility: the random operators draw from numpy's global RandomState, like the original --
FD_BG = ea.crtfld('BG', np.array([0, 0]), np.array([[0., 0], [1, 1]]), np.array([[1, 1], [1, 1]]))
BITS = np.random.RandomState(0).randint(0, 2, (6, int(FD_BG[0].sum())))
REAL = np.random.RandomState(1).rand(6, 3)
FDR3 = np.array([[0., 0, 0], [1, 1, 1], [0, 0, 0]])


@pytest.mark.parametrize('call', [
    lambda: ea.crtbp(4, 6), lambda: ea.crtrp(4, FDR3), lambda: ea.crtpp(4, FP),
    lambda: ea.mutbin('BG', BITS), lambda: ea.xovdp(BITS, 1), lambda: ea.xovud(BITS, 0.5),
    lambda: ea.xovpmx(PERM), lambda: ea.mutinv('P', PERM, FP, 1),
    lambda: ea.recsbx(REAL, 1), lambda: ea.mutpolyn('RI', REAL, FDR3, 1),
    lambda: ea.mutgau('RI', REAL, FDR3, 1), lambda: ea.selecting('tour', F6, 6),
    lambda: ea.selecting('rws', F6, 6),
])
def test_np_random_seed_makes_operators_reproducible(call):
    np.random.seed(123)
    a = call()
    np.random.seed(123)
    b = call()
    np.testing.assert_array_equal(a, b)


def test_optimize_seed_makes_runs_reproducible():
    def run():
        problem = ea.benchmarks.Rastrigrin(5)
        alg = ea.soea_SEGA_templet(problem, ea.Population(Encoding='BG', NIND=20), MAXGEN=30, logTras=0)
        res = ea.optimize(alg, seed=7, verbose=False, drawing=0, outputMsg=False, drawLog=False, saveFlag=False)
        return res['ObjV'][0][0]
    assert run() == run()
