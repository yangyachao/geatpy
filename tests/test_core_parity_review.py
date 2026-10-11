# -*- coding: utf-8 -*-
"""Regression tests for the second parity review (decompiled geatpy 2.7.0 core vs the Rust core).

Every expected value below was produced by the original 2.7.0 binaries (docker, see scripts/parity).
"""
import numpy as np
import pytest

import geatpy as ea

FI = np.array([[0, 0], [9, 9], [1, 1]])  # all variables integer
FM = np.array([[0, 0], [9, 9], [1, 0]])  # mixed
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=float)


# --- crtup / crtgp ------------------------------------------------------------------------------
def test_crtup_single_objective_is_linspace():
    w, n = ea.crtup(1, 5)
    assert n == 5
    np.testing.assert_allclose(w.ravel(), np.linspace(0, 1, 5))
    w, n = ea.crtup(1, None, 3)
    assert n == 4
    np.testing.assert_allclose(w.ravel(), np.linspace(0, 1, 4))


@pytest.mark.parametrize('m,num', [(2, 1), (3, 2), (5, 4), (10, 5)])
def test_crtup_rejects_num_below_dim(m, num):
    with pytest.raises(RuntimeError):
        ea.crtup(m, num)


@pytest.mark.parametrize('m,num', [(1, 1), (2, 3), (3, 7), (4, 10)])
def test_crtgp_rejects_degenerate_grid(m, num):
    with pytest.raises(RuntimeError):
        ea.crtgp(m, num)


# --- crtidp ------------------------------------------------------------------------------------
O = np.array([[1., 5.], [2., 3.], [4., 1.], [0.5, 9.]])
CV_ALL = np.array([[1.], [2.], [0.5], [3.]])
CV_SOME = np.array([[0.], [0.], [2.], [0.]])


@pytest.mark.parametrize('args,kw,expected', [
    ((O, CV_ALL), {}, [4.5, 9.5]),
    ((O, CV_SOME), {}, [0.5, 3.0]),
    ((O, CV_SOME), dict(maxormins=np.array([1, 1]), reverse=True), [6.0, 11.0]),
    ((O, CV_ALL), dict(maxormins=np.array([1, 1]), reverse=True), [7.0, 12.0]),
    ((O, CV_SOME, np.array([-1, 1])), {}, [-2.0, 3.0]),
])
def test_crtidp_penalises_infeasible_rows(args, kw, expected):
    np.testing.assert_allclose(ea.crtidp(*args, **kw), expected)


# --- boundfix / FixType --------------------------------------------------------------------------
@pytest.mark.parametrize('t', [1, 2, 3, 4])
def test_boundfix_degenerate_range_collapses_to_lb(t):
    out = ea.boundfix('RI', np.array([[2.0]]), np.array([[1.], [1.], [0.]]), t)
    assert out.tolist() == [[1.0]]


def test_fixtype_is_validated():
    with pytest.raises(RuntimeError):
        ea.boundfix('RI', np.array([[2.0]]), np.array([[0.], [1.], [0.]]), 5)
    with pytest.raises(RuntimeError):
        ea.mutpolyn('RI', X, FM, None, None, 0)


# --- int32 output when every variable is an integer ---------------------------------------------
RI_OPS = {
    'crtri': lambda F: ea.crtri(4, F),
    'boundfix': lambda F: ea.boundfix('RI', X, F),
    'mutpolyn': lambda F: ea.mutpolyn('RI', X, F),
    'mutgau': lambda F: ea.mutgau('RI', X, F, None, 1.0),
    'mutuni': lambda F: ea.mutuni('RI', X, F),
    'mutbga': lambda F: ea.mutbga('RI', X, F),
    'mutde': lambda F: ea.mutde('RI', X, F),
    'mutswap': lambda F: ea.mutswap('RI', X, F),
    'mutinv': lambda F: ea.mutinv('RI', X, F),
    'mutmove': lambda F: ea.mutmove('RI', X, F),
    'mutinv_int_input': lambda F: ea.mutinv('RI', X.astype(int), F),
}


@pytest.mark.parametrize('name', sorted(RI_OPS))
def test_ri_output_dtype_follows_vartypes(name):
    assert np.asarray(RI_OPS[name](FI)).dtype == np.int32
    assert np.asarray(RI_OPS[name](FM)).dtype == np.float64


def test_bs2ri_all_integer_returns_int32():
    fd = ea.crtfld('BG', np.array([1, 1]), np.array([[0, 0], [7, 3]]))
    out = ea.bs2ri(np.array([[1, 0, 1, 1, 1]]), fd)
    assert out.dtype == np.int32 and out.tolist() == [[5, 3]]


def test_mutpolyn_accepts_one_dimensional_chromosome():
    out = ea.mutpolyn('RI', np.array([12.0, 15.0]), np.array([[10., 10.], [20., 20.], [0, 0]]), 1.0)
    assert out.shape == (2,)
    assert np.all((out >= 10) & (out <= 20))


def test_mutde_mask_returns_one_value_per_selected_element():
    mask = np.array([[True, False], [False, False], [True, True], [False, True]])
    out = ea.mutde('RI', X, FM, None, 0.5, 1, mask)
    assert out.shape == (int(mask.sum()),)


# --- initialisation ------------------------------------------------------------------------------
def test_crtpp_rejects_columns_with_different_bounds():
    with pytest.raises(RuntimeError):
        ea.crtpp(3, np.array([[0, 1, 0], [5, 5, 5], [1, 1, 1]]))


# --- selection -----------------------------------------------------------------------------------
def test_sus_returns_pointers_in_ascending_order():
    for _ in range(20):
        idx = np.asarray(ea.sus(np.arange(10.).reshape(-1, 1), 10, None))
        assert np.all(np.diff(idx) >= 0)


def test_etour_keeps_the_best_individual():
    fit = np.random.rand(30, 1)
    best = int(np.argmax(fit))
    for _ in range(50):
        assert best in np.asarray(ea.etour(fit, 5, 2))
    assert np.asarray(ea.etour(fit, 1, 2)).tolist() == [best]


# --- multi-population ----------------------------------------------------------------------------
def test_migrate_and_mselecting_return_python_lists():
    ab, fo, fp = ea.migrate([10, 10, 10], 0.2, 0, 0, 0)
    assert isinstance(ab[0], list) and isinstance(fo[0], list)
    assert isinstance(fp, np.ndarray) and fp.dtype == np.int32
    assert [len(a) for a in ab] == [8, 8, 8] and [len(f) for f in fo] == [2, 2, 2]
    sel = ea.mselecting('dup', [np.arange(10.).reshape(-1, 1)] * 2, 6, 0.1)
    assert all(isinstance(s, list) for s in sel)


# --- crtfld validation ---------------------------------------------------------------------------
@pytest.mark.parametrize('enc,vt,border', [
    ('RI', 0, 0), ('RI', 1, 0), ('P', 0, 0), ('BG', 0, 1), ('BG', 1, 1), ('BG', 0, 0),
])
def test_crtfld_rejects_empty_range_after_borders(enc, vt, border):
    with pytest.raises(RuntimeError):
        ea.crtfld(enc, np.array([vt]), np.array([[2.0], [2.0]]), np.array([[border], [border]]))


def test_crtfld_accepts_degenerate_closed_range_for_ri():
    F = ea.crtfld('RI', np.array([0]), np.array([[2.0], [2.0]]), np.array([[1], [1]]))
    assert F[:, 0].tolist() == [2.0, 2.0, 0.0]


@pytest.mark.parametrize('prec', [9, 10])
def test_crtfld_rejects_codes_longer_than_31_bits(prec):
    with pytest.raises(RuntimeError):
        ea.crtfld('BG', np.array([0]), np.array([[0.], [1000.]]), None, [prec])
