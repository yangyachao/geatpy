# -*- coding: utf-8 -*-
"""Second batch: global-mode recombination, defaults and validation (checks the agy findings)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fz  # noqa: E402

ea = fz.ea


def cases():
    C = []

    def add(name, f, reps=1):
        C.append((name, f, reps))

    R = fz.R
    real = np.round(R(1).uniform(-3, 3, (6, 7)), 3)
    ints = np.repeat(np.arange(8).reshape(-1, 1), 6, axis=1)  # row i is all i: shows where genes come from
    perm = np.array([R(2 + k).permutation(7) for k in range(6)])
    for fn, X in [('recsbx', real), ('recint', real), ('reclin', real), ('recndx', real), ('recdis', ints),
                  ('xovsp', ints), ('xovdp', ints), ('xovud', ints), ('xovsh', ints), ('xovsec', ints),
                  ('xovbd', ints), ('xovexp', ints), ('xovpmx', perm), ('xovox', perm)]:
        f = getattr(ea, fn)
        for k in [3, 8, 20]:
            for p in [0.7, 1]:
                add('%s_g%d_p%s' % (fn, k, p), lambda f=f, X=X, k=k, p=p: f(X, p, k), 1500)
        add('%s_default' % fn, lambda f=f, X=X: f(X), 1500)
        add('%s_xovr_neg' % fn, lambda f=f, X=X: f(X, -0.1))
        add('%s_xovr_big' % fn, lambda f=f, X=X: f(X, 1.5))
        add('%s_1d' % fn, lambda f=f, X=X: f(np.ascontiguousarray(X[:, 0])))
        add('%s_halfn_neg' % fn, lambda f=f, X=X: f(X, 0.7, -1))
    # distinct-source statistics for global discrete operators: number of distinct values per row
    for fn in ['recdis', 'xovud', 'xovsp', 'xovdp']:
        add('%s_g_distinct' % fn, lambda f=getattr(ea, fn): np.array([len(set(r)) for r in f(ints, 1, 6)]), 1500)
    add('reclin_g_ratio', lambda: (lambda Y: np.std((Y - real[:1]) / (real[1:2] - real[:1] + 1e-12), axis=1))(ea.reclin(real, 1, 6)), 800)
    bad = np.array([[0, 0, 1, 2], [1, 2, 3, 0], [3, 2, 1, 0], [0, 1, 2, 3]])
    add('xovox_badperm', lambda: ea.xovox(bad))
    add('xovpmx_badperm', lambda: ea.xovpmx(bad))
    add('mutpp_badperm', lambda: ea.mutpp('P', bad, np.array([[0] * 4, [3] * 4, [1] * 4])))
    add('mutinv_badperm', lambda: ea.mutinv('P', bad, np.array([[0] * 4, [3] * 4, [1] * 4])))

    # selection: params2 / tour size
    fit = np.array([[1.], [10.]])
    fit6 = np.arange(6.).reshape(-1, 1)
    add('tour_k10_n2', lambda: ea.tour(fit, 50, 10), 1500)
    add('etour_k10_n2', lambda: ea.etour(fit, 50, 10), 1500)
    add('tour_k5_n6', lambda: ea.tour(fit6, 6, 5), 1500)
    add('selecting_tour_p4', lambda: ea.selecting('tour', fit6, 6, 4), 1500)
    add('selecting_tour_p2', lambda: ea.selecting('tour', fit6, 6, 2), 1500)
    add('selecting_etour_p4', lambda: ea.selecting('etour', fit6, 6, 4), 1500)
    add('selecting_dup', lambda: ea.selecting('dup', fit6, 6), 1500)
    add('mselecting_tour', lambda: ea.mselecting('tour', [fit6, fit6[:4]], 5), 1500)
    add('tour_k0', lambda: ea.tour(fit6, 6, 0))
    # crtbp / field validation
    add('crtbp_0', lambda: ea.crtbp(0, 5))
    add('crtbp_l0', lambda: ea.crtbp(5, 0))
    FD = ea.crtfld('BG', np.array([1]), np.array([[0], [7]]))
    FD32 = FD.copy()
    FD32[0, 0] = 32
    add('bs2ri_len32', lambda: ea.bs2ri(np.ones((2, 32), dtype=int), FD32))
    add('ri2bs_len32', lambda: ea.ri2bs(np.array([[3.], [5.]]), FD32))
    FD70 = FD.copy()
    FD70[0, 0] = 70
    add('ri2bs_len70', lambda: ea.ri2bs(np.array([[3.], [5.]]), FD70))
    # mutation validation
    FDR = np.array([[0., 0, 0], [5, 5, 5], [0, 0, 0]])
    ch = np.array([[1., 2, 3], [4, 0, 1], [2, 2, 2]])
    add('mutbga_shrink_neg', lambda: ea.mutbga('RI', ch, FDR, 0.5, -0.2))
    add('mutbga_shrink_big', lambda: ea.mutbga('RI', ch, FDR, 0.5, 1.5))
    add('mutbga_grad0', lambda: ea.mutbga('RI', ch, FDR, 0.5, 0.5, 0))
    P = np.array([R(30 + k).permutation(6) for k in range(5)])
    FP = np.array([[0] * 6, [5] * 6, [1] * 6])
    add('mutinv_len1', lambda: ea.mutinv('P', P, FP, 1, 1))
    add('mutinv_len0', lambda: ea.mutinv('P', P, FP, 1, 0))
    add('mutinv_len9', lambda: ea.mutinv('P', P, FP, 1, 9))
    add('mutmove_len1', lambda: ea.mutmove('P', P, FP, 1, 1))
    add('mutmove_len9', lambda: ea.mutmove('P', P, FP, 1, 9))
    add('mutinv_P_fdr', lambda: ea.mutinv('P', P, FP, 1), 1500)
    add('mutmove_P_fdr', lambda: ea.mutmove('P', P, FP, 1), 1500)
    add('mutswap_P_fdr', lambda: ea.mutswap('P', P, FP, 1), 1500)
    add('mutpp_pm', lambda: ea.mutpp('P', P, FP, 0.5), 1500)
    add('mutbin_bg', lambda: ea.mutbin('BG', (P > 2).astype(int), None, 0.5), 1500)
    add('mutpolyn_pm_big', lambda: ea.mutpolyn('RI', ch, FDR, 1.5))
    add('mutgau_pm_neg', lambda: ea.mutgau('RI', ch, FDR, -0.5))
    add('ranking_nl_sp_big', lambda: ea.ranking(fit6, None, None, 1, 7))
    add('ranking_lin_sp_big', lambda: ea.ranking(fit6, None, None, 0, 2.5))
    add('scaling_smul1', lambda: ea.scaling(fit6, None, None, 1))
    return C


if __name__ == '__main__':
    fz.cases = cases
    fz.main()
