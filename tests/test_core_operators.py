import pytest
import numpy as np
import geatpy as ea

def test_encoding_operators():
    # crtfld
    var_types = np.array([0, 1])
    ranges = np.array([[0, 10], [0, 5]])
    borders = np.array([[1, 1], [1, 1]])
    fld = ea.crtfld('RI', var_types, ranges, borders)
    assert fld.shape == (3, 2)

    # crtbp
    chrom = ea.crtbp(10, 8)
    assert chrom.shape == (10, 8)
    assert np.all((chrom == 0) | (chrom == 1))

    # crtrp
    field_dr = np.array([[0, 0], [10, 10], [0, 0]])
    chrom_real = ea.crtrp(10, field_dr)
    assert chrom_real.shape == (10, 2)
    assert np.all(chrom_real >= 0) and np.all(chrom_real <= 10)

    # crtpp
    chrom_perm = ea.crtpp(5, field_dr)
    assert chrom_perm.shape == (5, 2)

    # boundfix
    fixed = ea.boundfix('RI', np.array([[-1.0, 15.0]]), field_dr)
    assert fixed[0, 0] >= 0.0
    assert fixed[0, 1] <= 10.0

def test_selection_operators():
    fitn_v = np.array([1.0, 5.0, 2.0, 8.0, 3.0])
    
    # tour
    sel_tour = ea.tour(fitn_v, 4)
    assert len(sel_tour) == 4

    # dup
    sel_dup = ea.dup(fitn_v, 3)
    assert len(sel_dup) == 3

    # selecting
    sel = ea.selecting('tour', fitn_v, 4)
    assert len(sel) == 4

    # ranking
    obj_v = np.array([10.0, 2.0, 5.0])
    ranks = ea.ranking(obj_v)
    assert ranks.shape == (3, 1)
    # smaller obj_v (index 1) should have highest rank
    assert ranks[1, 0] > ranks[0, 0]

    # otos & ecs
    fit_2n = np.array([1.0, 2.0, 5.0, 1.0])
    otos_res = ea.otos(fit_2n, 2)
    assert len(otos_res) == 2
    assert otos_res[0] == 2  # fit_2n[2] = 5 > fit_2n[0] = 1
    assert otos_res[1] == 1  # fit_2n[1] = 2 > fit_2n[3] = 1

    ecs_res = ea.ecs(fitn_v, 4)
    assert len(ecs_res) == 4
    assert np.all(ecs_res == 3)  # index 3 has max fit 8.0

def test_recombination_operators():
    chrom = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    
    # recsbx
    new_chrom = ea.recsbx(chrom, 1.0, False, 20.0)
    assert new_chrom.shape == chrom.shape

    # recdis
    new_dis = ea.recdis(chrom)
    assert new_dis.shape == chrom.shape

    # xovsp
    new_sp = ea.xovsp(chrom)
    assert new_sp.shape == chrom.shape

    # xovdp
    new_dp = ea.xovdp(chrom)
    assert new_dp.shape == chrom.shape

    # xovpmx
    perm_chrom = np.array([[0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0]])
    new_pmx = ea.xovpmx(perm_chrom)
    assert new_pmx.shape == perm_chrom.shape

    # xovox
    new_ox = ea.xovox(perm_chrom)
    assert new_ox.shape == perm_chrom.shape

def test_mutation_operators():
    chrom = np.array([[1.0, 2.0], [3.0, 4.0]])
    field_dr = np.array([[0, 0], [10, 10], [0, 0]])

    # mutpolyn
    mutated = ea.mutpolyn('RI', chrom, field_dr)
    assert mutated.shape == chrom.shape
    assert np.all(mutated >= 0) and np.all(mutated <= 10)

    # mutbin
    bin_chrom = np.array([[0.0, 1.0], [1.0, 0.0]])
    mut_bin = ea.mutbin('BG', bin_chrom, pm=1.0)
    assert np.all(mut_bin == (1.0 - bin_chrom))

    # mutswap
    perm_chrom = np.array([[0.0, 1.0, 2.0, 3.0]])
    mut_swap = ea.mutswap('P', perm_chrom, pm=1.0)
    assert mut_swap.shape == perm_chrom.shape
    assert set(mut_swap[0]) == {0, 1, 2, 3}

def test_multiobjective_and_indicators():
    # ndsortESS, ndsortDED
    obj_v = np.array([[1.0, 5.0], [2.0, 3.0], [5.0, 1.0], [4.0, 4.0]])
    levels, front_cnt = ea.ndsortESS(obj_v)
    assert len(levels) == 4
    # individual [4, 4] is dominated by [2, 3]
    assert levels[3] > levels[1]

    # crowdis
    cd = ea.crowdis(obj_v, levels)
    assert len(cd) == 4

    # cdist
    dists = ea.cdist(obj_v, obj_v)
    assert dists.shape == (4, 4)
    assert np.allclose(np.diag(dists), 0.0)

    # Indicators
    p_front = np.array([[1.0, 5.0], [2.0, 3.0], [5.0, 1.0]])
    ref_front = np.array([[1.0, 5.0], [2.0, 3.0], [5.0, 1.0]])
    gd = ea.indicator.GD(p_front, ref_front)
    assert gd >= 0.0
    assert np.isclose(gd, 0.0)

    igd = ea.indicator.IGD(p_front, ref_front)
    assert np.isclose(igd, 0.0)

    spacing = ea.indicator.Spacing(p_front)
    assert spacing >= 0.0

    hv = ea.indicator.HV(p_front, np.array([[10.0, 10.0]]))
    assert hv > 0.0
