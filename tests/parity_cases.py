# -*- coding: utf-8 -*-
"""
Behavioural parity cases between the Rust core and the official geatpy 2.7.0 core.

The same module is executed against both cores:
  * scripts/parity/run_original_core.sh runs it inside a linux/amd64 Python 3.6 container
    with the original 2.7.0 binaries and stores the results in
    tests/fixtures/core_parity_orig_2_7_0.json;
  * tests/test_core_parity.py runs it against the Rust core and compares.

Only black-box inputs/outputs are recorded; nothing here is derived from the original
implementation's code. Keep this file Python 3.6 compatible.
"""
import numpy as np

# ---------------------------------------------------------------------------
# Shared inputs (fixed seed, plain numpy so both environments build identical data)
# ---------------------------------------------------------------------------
_rs = np.random.RandomState(12345)
X2 = _rs.rand(60, 2)
X3 = _rs.rand(60, 3)
X4 = _rs.rand(40, 4)
DUP2 = np.vstack([X2[:20], X2[:10]])
CV2 = np.c_[_rs.rand(60) - 0.7, _rs.rand(60) - 0.8]
CV1 = CV2[:, :1]
MOM = np.array([1, -1])
F1 = X2[:10, :1].copy()
F1_TIES = np.array([[3.], [1.], [2.], [1.], [5.], [3.], [0.], [2.], [4.], [0.]])
CV10 = CV2[:10]
W2 = np.array([[0.2, 0.8]] * 6)
FIELD_BG = None  # built lazily because it needs ea.crtfld


def _bin_chrom(field, n, seed):
    return (np.random.RandomState(seed).rand(n, int(np.sum(field[0]))) > 0.5).astype(int)


def _nd(objv):
    import geatpy as ea
    return objv[ea.ndsortESS(objv)[0] == 1]


def _cases(ea):
    """name -> zero-arg callable. Every callable must be deterministic."""
    FBG = ea.crtfld('BG', np.array([0, 1]), np.array([[-1, 0], [2, 9]]), np.array([[0, 1], [1, 0]]),
                    [3, 0], [1, 0], [0, 0])
    FBG_IN = ea.crtfld('BG', np.array([0, 1]), np.array([[-1, 0], [2, 9]]), np.array([[1, 1], [1, 1]]),
                       [3, 0], [1, 0], [0, 0])
    FBG_OPEN = ea.crtfld('BG', np.array([0]), np.array([[0.], [1.]]), np.array([[0], [0]]), [2])
    FDR = np.array([[-1., 0, 0], [2, 1, 10], [0, 1, 0]])
    PF2 = _nd(X2)
    PF3 = _nd(X3)
    c = {}
    # --- non-dominated sorting -------------------------------------------------
    for fn in ['ndsortESS', 'ndsortTNS', 'ndsortDED']:
        f = getattr(ea, fn)
        c[fn + '_all'] = lambda f=f: f(X2)
        c[fn + '_need30'] = lambda f=f: f(X2, 30)
        c[fn + '_need1'] = lambda f=f: f(X2, 1)
        c[fn + '_level2'] = lambda f=f: f(X2, None, 2)
        c[fn + '_need30_level2'] = lambda f=f: f(X2, 30, 2)
        c[fn + '_cv'] = lambda f=f: f(X2, None, None, CV2)
        c[fn + '_cv_need20'] = lambda f=f: f(X2, 20, None, CV2)
        c[fn + '_maxmin'] = lambda f=f: f(X2, None, None, None, MOM)
        c[fn + '_3obj_need30'] = lambda f=f: f(X3, 30)
        c[fn + '_4obj'] = lambda f=f: f(X4)
        c[fn + '_dup'] = lambda f=f: f(DUP2)
        c[fn + '_dup_need15'] = lambda f=f: f(DUP2, 15)
    # --- crowding / distances / aggregation ------------------------------------
    c['crowdis_levels'] = lambda: ea.crowdis(X2, ea.ndsortESS(X2)[0])
    c['crowdis_ones'] = lambda: ea.crowdis(X3, np.ones(len(X3)))
    c['crowdis_partial'] = lambda: ea.crowdis(X2, ea.ndsortESS(X2, 20)[0])
    for m in ['euclidean', 'cityblock', 'cosine', 'cosine_similarity']:
        c['cdist_' + m] = lambda m=m: ea.cdist(X3[:5], X3[5:9], m)
    c['cdist_default'] = lambda: ea.cdist(X3[:5], X3[5:9])
    c['mergecv'] = lambda: ea.mergecv(CV2)
    c['mergecv_threshold'] = lambda: ea.mergecv(CV2, 0.1)
    c['mergecv_count'] = lambda: ea.mergecv(CV2, None, True)
    c['tcheby'] = lambda: ea.tcheby(X2[:6], W2)
    c['tcheby_ideal'] = lambda: ea.tcheby(X2[:6], W2, np.zeros(2))
    c['tcheby_cv'] = lambda: ea.tcheby(X2[:6], W2, np.zeros(2), CV2[:6])
    c['tcheby_maxmin'] = lambda: ea.tcheby(X2[:6], W2, None, None, MOM)
    c['tcheby_one_row'] = lambda: ea.tcheby(X2[:1], W2)
    c['pbi'] = lambda: ea.pbi(X2[:6], W2)
    c['pbi_ideal_theta'] = lambda: ea.pbi(X2[:6], W2, np.zeros(2), None, None, 2.0)
    c['pbi_cv'] = lambda: ea.pbi(X2[:6], W2, np.zeros(2), CV2[:6])
    c['pbi_maxmin'] = lambda: ea.pbi(X2[:6], W2, None, None, MOM)
    c['awGA'] = lambda: ea.awGA(X2[:10])
    c['awGA_cv'] = lambda: ea.awGA(X2[:10], CV10)
    c['awGA_maxmin'] = lambda: ea.awGA(X2[:10], None, MOM)
    c['crtidp'] = lambda: ea.crtidp(X2)
    c['crtidp_cv'] = lambda: ea.crtidp(X2, CV2)
    c['crtidp_maxmin'] = lambda: ea.crtidp(X2, None, MOM)
    c['crtidp_old'] = lambda: ea.crtidp(X2, None, None, np.array([0.05, 0.5]))
    c['crtidp_reverse'] = lambda: ea.crtidp(X2, maxormins=np.array([1, 1]), reverse=True)
    c['crtidp_kw_old'] = lambda: ea.crtidp(X2, maxormins=np.array([1, 1]), old_idealPoint=np.array([0.05, 0.5]))
    # --- fitness assignment -----------------------------------------------------
    for fn in ['ranking', 'scaling', 'powing', 'indexing']:
        f = getattr(ea, fn)
        c[fn] = lambda f=f: f(F1)
        c[fn + '_ties'] = lambda f=f: f(F1_TIES)
        c[fn + '_cv'] = lambda f=f: f(F1, CV10)
        c[fn + '_max'] = lambda f=f: f(F1, None, np.array([-1]))
        c[fn + '_cv_max'] = lambda f=f: f(F1, CV10, np.array([-1]))
    c['ranking_sp'] = lambda: ea.ranking(F1, None, None, None, 1.5)
    c['scaling_smul'] = lambda: ea.scaling(F1, None, None, 3)
    c['powing_k'] = lambda: ea.powing(F1, None, None, 3)
    c['indexing_beta'] = lambda: ea.indexing(F1, None, None, 5)
    # --- deterministic selection -----------------------------------------------
    c['dup'] = lambda: ea.dup(F1, 6)
    c['dup_more'] = lambda: ea.dup(F1, 25)
    c['ecs'] = lambda: ea.ecs(F1, 6)
    c['otos'] = lambda: ea.otos(F1, 5)
    c['otos_2'] = lambda: ea.otos(F1, 2)
    # --- reference points ------------------------------------------------------
    for m, n in [(2, 100), (3, 91), (3, 100), (5, 100), (8, 100), (10, 200)]:
        c['crtup_%d_%d' % (m, n)] = lambda m=m, n=n: ea.crtup(m, n)
    c['crtup_div'] = lambda: ea.crtup(3, None, 12)
    c['crtgp_2_50'] = lambda: ea.crtgp(2, 50)
    c['crtgp_3_100'] = lambda: ea.crtgp(3, 100)
    # --- encoding --------------------------------------------------------------
    c['crtfld_RI'] = lambda: ea.crtfld('RI', np.array([0, 1, 1]), np.array([[-1, 0.5, -3], [2, 9.5, 4]]),
                                       np.array([[1, 0, 1], [0, 1, 0]]))
    c['crtfld_RI_prec'] = lambda: ea.crtfld('RI', np.array([0, 0]), np.array([[1, 2], [3, 4]]),
                                            np.array([[0, 1], [1, 0]]), [1, 2])
    c['crtfld_RI_doc'] = lambda: ea.crtfld('RI', np.array([0, 1, 1]), np.array([[1.1, 2, 3.1], [3, 4, 5]]),
                                           np.array([[1, 0, 1], [1, 1, 1]]))
    c['crtfld_P'] = lambda: ea.crtfld('P', np.array([0, 0, 0]), np.array([[0, 0, 0], [9, 9, 9]]),
                                      np.array([[1, 1, 1], [1, 1, 1]]))
    c['crtfld_BG'] = lambda: FBG
    c['crtfld_BG_default'] = lambda: ea.crtfld('BG', np.array([0, 0, 1]), np.array([[-5.12, 0, 0], [5.12, 1, 7]]),
                                               np.array([[1, 1, 1], [1, 1, 1]]))
    c['crtfld_BG_open'] = lambda: FBG_OPEN
    c['crtfld_BG_gray'] = lambda: ea.crtfld('BG', np.array([0, 1]), np.array([[0, 0], [1, 100]]),
                                            np.array([[1, 1], [1, 1]]), [2, 0], [1, 1])
    c['bs2ri'] = lambda: ea.bs2ri(_bin_chrom(FBG, 12, 1), FBG)
    c['bs2ri_inclusive'] = lambda: ea.bs2ri(_bin_chrom(FBG_IN, 12, 2), FBG_IN)
    c['bs2ri_open_zero'] = lambda: ea.bs2ri(np.zeros((1, int(FBG_OPEN[0].sum())), int), FBG_OPEN)
    c['bs2ri_open_one'] = lambda: ea.bs2ri(np.ones((1, int(FBG_OPEN[0].sum())), int), FBG_OPEN)
    c['bs2ri_float_chrom'] = lambda: ea.bs2ri(_bin_chrom(FBG, 6, 3).astype(float), FBG)
    c['bs2int'] = lambda: ea.bs2int(_bin_chrom(FBG_IN, 8, 4), FBG_IN)
    c['bs2real'] = lambda: ea.bs2real(_bin_chrom(FBG_IN, 8, 5), FBG_IN)
    c['ri2bs'] = lambda: ea.ri2bs(np.array([[-1.0, 0], [0.5, 4], [2, 9], [1.3, 7]]), FBG_IN)
    c['ri2bs_roundtrip'] = lambda: ea.bs2ri(ea.ri2bs(np.array([[-1.0, 0], [0.5, 4], [2, 9]]), FBG_IN), FBG_IN)
    for t in [1, 2, 3]:
        c['boundfix_%d' % t] = lambda t=t: ea.boundfix('RI', np.array([[-2.5, 0.4, 12.0], [3.7, -0.6, -3.2]]), FDR, t)
    # --- reference-point based environmental selection ----------------------------
    c['refselect_3obj'] = lambda: (lambda l: ea.refselect(X3, l[0], l[1], 30, ea.crtup(3, 28)[0]))(ea.ndsortESS(X3, 30))
    c['refselect_2obj'] = lambda: (lambda l: ea.refselect(X2, l[0], l[1], 20, ea.crtup(2, 20)[0]))(ea.ndsortESS(X2, 20))
    c['refselect_maxmin'] = lambda: (lambda l: ea.refselect(X2, l[0], l[1], 25, ea.crtup(2, 12)[0], MOM))(
        ea.ndsortESS(X2, 25, None, None, MOM))
    c['refselect_4obj'] = lambda: (lambda l: ea.refselect(X4, l[0], l[1], 20, ea.crtup(4, 35)[0]))(ea.ndsortESS(X4, 20))
    c['refselect_all'] = lambda: ea.refselect(X2[:5], np.ones(5), 1, 10, ea.crtup(2, 5)[0])
    c['refgselect'] = lambda: ea.refgselect(X3, ea.crtup(3, 28)[0], 0.5)
    c['refgselect_cv'] = lambda: ea.refgselect(X2, ea.crtup(2, 15)[0], 1.0, CV2)
    c['refgselect_gamma'] = lambda: ea.refgselect(X3, ea.crtup(3, 28)[0], 2.0, None, np.full(28, 0.3))
    c['refgselect_maxmin'] = lambda: ea.refgselect(X2, ea.crtup(2, 15)[0], 0.5, None, None, MOM)
    # --- extra fitness assignment options ------------------------------------------
    c['ranking_rm1'] = lambda: ea.ranking(F1, None, None, 1, 3.0)
    c['ranking_mask'] = lambda: ea.ranking(F1, None, None, None, None, np.arange(10.) * 2)
    for fn in ['ranking', 'scaling', 'powing', 'indexing']:
        c[fn + '_equal'] = lambda f=getattr(ea, fn): f(np.ones((6, 1)))
    # --- indicators ------------------------------------------------------------
    c['GD'] = lambda: ea.indicator.GD(X2[:20], PF2)
    c['IGD'] = lambda: ea.indicator.IGD(X2[:20], PF2)
    c['Spacing'] = lambda: ea.indicator.Spacing(PF2)
    c['HV_2d'] = lambda: ea.indicator.HV(PF2)
    c['HV_2d_pf'] = lambda: ea.indicator.HV(PF2, np.array([[1.2, 1.2]]))
    c['HV_2d_pf_front'] = lambda: ea.indicator.HV(PF2[:5], PF2)
    c['HV_3d'] = lambda: ea.indicator.HV(PF3)
    c['HV_3d_pf'] = lambda: ea.indicator.HV(PF3, np.array([[1.1, 1.1, 1.1]]))
    return c


# Cases where the original 2.7.0 binary is demonstrably wrong; the Rust core follows the
# documented behaviour instead and tests/test_core_parity.py checks that separately.
KNOWN_ORIGINAL_DEFECTS = {
    'ndsortDED': 'original ndsortDED compares the first rows of ObjV instead of the remaining '
                 'individuals from the second front on, producing invalid fronts',
    'mergecv': 'original mergecv never resets its accumulator, returning a running sum over rows',
    'refselect': 'original builds the candidate matrix in index order but treats its first N1 rows as the '
                 'earlier fronts, mixing up niche counts and critical-front candidates',
}
_IDEAL_RETURNED = 'original returns the computed ideal point instead of the aggregated values when idealPoint is None'
KNOWN_ORIGINAL_DEFECT_CASES = {name: _IDEAL_RETURNED for name in
                               ['tcheby', 'tcheby_maxmin', 'tcheby_one_row', 'pbi', 'pbi_maxmin']}
KNOWN_ORIGINAL_DEFECT_CASES['refselect_all'] = ('original returns np.arange(N) instead of the documented all-True '
                                                'flags when no selection is needed')


# Operators that return their selection in random order (as the original does): compare as multisets.
ORDER_FREE = {'dup', 'dup_more', 'otos', 'otos_2'}


def known_defect(name):
    if name in KNOWN_ORIGINAL_DEFECT_CASES:
        return KNOWN_ORIGINAL_DEFECT_CASES[name]
    for prefix, reason in KNOWN_ORIGINAL_DEFECTS.items():
        if name.startswith(prefix):
            return reason
    return None


def deterministic_cases(ea):
    return _cases(ea)


# ---------------------------------------------------------------------------
# Random operators: compared statistically (value, absolute tolerance)
# ---------------------------------------------------------------------------
def random_stats(ea):
    rs = np.random.RandomState(7)
    P = rs.rand(2000, 10)
    F = np.array([[0.] * 10, [1.] * 10, [0] * 10])
    H = np.vstack([P[:1000], rs.rand(1000, 10)])
    OUT = P.copy()
    OUT[:, 0] = 1.7
    OUT[:, 1] = -0.4
    B = (P > .5).astype(int)
    PERM = np.array([np.random.RandomState(i).permutation(10) for i in range(2000)])
    FP = np.array([[0] * 10, [9] * 10, [1] * 10])

    def ch(Y, X):
        return float((Y != X[:len(Y)]).mean())

    def dl(Y, X):
        return float(np.abs(Y - X[:len(Y)]).mean())

    def oob(Y):
        return float(((Y < 0) | (Y > 1)).mean())

    def perm_ok(Y):
        return float(np.mean([len(set(r)) == 10 for r in Y]))

    s = {}
    s['xovexp_cr05_half'] = (lambda: ch(ea.Xovexp(XOVR=0.5, Half_N=True).do(H), H), 0.03)
    s['xovbd_cr05_half'] = (lambda: ch(ea.Xovbd(XOVR=0.5, Half_N=True).do(H), H), 0.03)
    s['xovsp_changed'] = (lambda: ch(ea.Xovsp(XOVR=0.7).do(P), P), 0.05)
    s['xovdp_changed'] = (lambda: ch(ea.Xovdp(XOVR=0.7).do(P), P), 0.05)
    s['xovud_changed'] = (lambda: ch(ea.Xovud(XOVR=0.7).do(P), P), 0.03)
    s['xovsh_changed'] = (lambda: ch(ea.Xovsh(XOVR=0.7).do(P), P), 0.05)
    s['recsbx_delta'] = (lambda: dl(ea.Recsbx(XOVR=1, n=20).do(P), P), 0.01)
    s['recsbx_changed'] = (lambda: ch(ea.Recsbx(XOVR=1, n=20).do(P), P), 0.05)
    s['recint_default_changed'] = (lambda: ch(ea.Recint().do(P), P), 0.05)
    s['recint_default_delta'] = (lambda: dl(ea.Recint().do(P), P), 0.02)
    s['reclin_default_changed'] = (lambda: ch(ea.Reclin().do(P), P), 0.05)
    s['recdis_default_changed'] = (lambda: ch(ea.Recdis().do(P), P), 0.03)
    s['recndx_delta'] = (lambda: dl(ea.Recndx().do(P), P), 0.02)
    s['xovpmx_valid'] = (lambda: perm_ok(ea.Xovpmx(XOVR=1).do(PERM)), 0.0)
    s['xovpmx_changed'] = (lambda: ch(ea.Xovpmx(XOVR=1).do(PERM), PERM), 0.05)
    s['xovox_valid'] = (lambda: perm_ok(ea.Xovox(XOVR=1).do(PERM)), 0.0)
    s['xovox_changed'] = (lambda: ch(ea.Xovox(XOVR=1).do(PERM), PERM), 0.05)
    s['mutpolyn_pm01_changed'] = (lambda: ch(ea.Mutpolyn(Pm=0.1).do('RI', P, F), P), 0.02)
    s['mutpolyn_pm01_delta'] = (lambda: dl(ea.Mutpolyn(Pm=0.1).do('RI', P, F), P), 0.002)
    s['mutpolyn_repairs_all'] = (lambda: oob(ea.Mutpolyn(Pm=0.2).do('RI', OUT, F)), 0.0)
    s['mutgau_default_changed'] = (lambda: ch(ea.Mutgau().do('RI', P, F), P), 0.02)
    s['mutgau_default_delta'] = (lambda: dl(ea.Mutgau().do('RI', P, F), P), 0.005)
    s['mutgau_repairs_all'] = (lambda: oob(ea.Mutgau(Pm=0.2).do('RI', OUT, F)), 0.0)
    s['mutuni_default_changed'] = (lambda: ch(ea.Mutuni().do('RI', P, F), P), 0.02)
    s['mutuni_repairs_all'] = (lambda: oob(ea.Mutuni(Pm=0.2).do('RI', OUT, F)), 0.0)
    s['mutbga_default_delta'] = (lambda: dl(ea.Mutbga().do('RI', P, F), P), 0.0015)
    s['mutbga_repairs_all'] = (lambda: oob(ea.Mutbga(Pm=0.2).do('RI', OUT, F)), 0.0)
    s['mutbin_default_changed'] = (lambda: ch(ea.Mutbin().do('BG', B, None), B), 0.02)
    s['mutde_delta'] = (lambda: dl(ea.Mutde(F=0.5).do('RI', P, F, [np.arange(2000)]), P), 0.02)
    s['mutde_F_list_delta'] = (lambda: dl(ea.Mutde(F=[0.5, None]).do('RI', P, F, [np.arange(2000)]), P), 0.03)
    s['mutde_repairs_all'] = (lambda: oob(ea.Mutde(F=0.5).do('RI', OUT, F, [np.arange(2000)])), 0.0)
    s['mutinv_changed'] = (lambda: ch(ea.Mutinv(Pm=1).do('P', PERM, FP), PERM), 0.04)
    s['mutinv_valid'] = (lambda: perm_ok(ea.Mutinv(Pm=1).do('P', PERM, FP)), 0.0)
    s['mutswap_changed'] = (lambda: ch(ea.Mutswap(Pm=1).do('P', PERM, FP), PERM), 0.04)
    s['mutmove_changed'] = (lambda: ch(ea.Mutmove(Pm=1).do('P', PERM, FP), PERM), 0.04)
    s['mutpp_changed'] = (lambda: ch(ea.Mutpp(Pm=1).do('P', PERM, FP), PERM), 0.04)
    fit = np.arange(100, dtype=float).reshape(-1, 1)
    for sel in ['tour', 'etour', 'rws', 'sus', 'dup', 'ecs', 'urs', 'rcs']:
        s['sel_%s_mean' % sel] = (lambda sel=sel: float(np.mean(
            [fit[ea.selecting(sel, fit, 100)].mean() for _ in range(40)])), 2.0)
    s['rps_unique'] = (lambda: float(len(np.unique(ea.rps(100, 100)))), 0.0)
    s['crtpp_valid'] = (lambda: perm_ok(ea.crtpp(500, FP)), 0.0)
    s['crtrp_mean'] = (lambda: float(ea.crtrp(5000, F).mean()), 0.02)
    s['crtbp_mean'] = (lambda: float(ea.crtbp(5000, 10).mean()), 0.02)
    s['rwGA_mean'] = (lambda: float(np.mean([ea.rwGA(X2).mean() for _ in range(50)])), 0.02)
    s['HV_4d'] = (lambda: float(ea.indicator.HV(_nd(X4))), 0.03)
    return s


# ---------------------------------------------------------------------------
# End-to-end template runs: (key, template, problem factory, encoding, NIND, MAXGEN)
# ---------------------------------------------------------------------------
def template_runs(ea):
    problems = {
        'Rastrigin10': lambda: ea.benchmarks.Rastrigrin(Dim=10),
        'ZDT1': lambda: ea.benchmarks.ZDT1(),
        'DTLZ2': lambda: ea.benchmarks.DTLZ2(M=3),
        'C1-DTLZ1': lambda: ea.benchmarks.C1_DTLZ1(M=3),
    }
    plan = []
    for t in ['DE_rand_1_bin', 'DE_rand_1_L', 'DE_best_1_bin', 'DE_best_1_L', 'DE_currentToBest_1_bin',
              'DE_currentToBest_1_L', 'DE_currentToRand_1', 'DE_targetToBest_1_bin', 'DE_targetToBest_1_L',
              'ES_1_plus_1', 'ES_miu_plus_lambda']:
        plan.append(('soea_%s_templet' % t, 'Rastrigin10', 'RI', 50, 200))
    for t in ['SGA', 'EGA', 'SEGA', 'GGAP_SGA', 'steadyGA', 'studGA']:
        plan.append(('soea_%s_templet' % t, 'Rastrigin10', 'RI', 50, 200))
        plan.append(('soea_%s_templet' % t, 'Rastrigin10', 'BG', 50, 200))
    for t in ['NSGA2', 'NSGA2_DE', 'NSGA2_archive', 'awGA', 'MOEAD', 'MOEAD_DE', 'MOEAD_archive',
              'PPS_MOEAD_DE_archive']:
        plan.append(('moea_%s_templet' % t, 'ZDT1', 'RI', 100, 150))
    for t in ['NSGA3', 'NSGA3_DE', 'RVEA', 'RVEA_RES', 'NSGA2', 'MOEAD']:
        plan.append(('moea_%s_templet' % t, 'DTLZ2', 'RI', 91, 150))
    for t in ['NSGA2', 'NSGA3']:
        plan.append(('moea_%s_templet' % t, 'C1-DTLZ1', 'RI', 91, 150))
    return [('%s[%s]@%s' % (name, enc, prob), name, problems[prob], enc, nind, gen)
            for name, prob, enc, nind, gen in plan]


def igd(objv, pf):
    if len(objv) == 0:
        return float('inf')
    d = np.sqrt(((pf[:, None, :] - objv[None, :, :]) ** 2).sum(-1))
    return float(d.min(1).mean())


def run_template(ea, name, make_problem, enc, nind, gen):
    """Returns the scalar quality of one run: best objective (soea) or IGD (moea)."""
    prob = make_problem()
    alg = getattr(ea, name)(prob, ea.Population(enc, NIND=nind), MAXGEN=gen, logTras=0)
    res = ea.optimize(alg, verbose=False, drawing=0, outputMsg=False, drawLog=False, saveFlag=False)
    objv = np.asarray(res['optPop'].ObjV, float)
    if name.startswith('soea'):
        return float(objv.min())
    pf = prob.ReferObjV if prob.ReferObjV is not None else prob.calReferObjV()
    return igd(objv, np.asarray(pf, float))


# ---------------------------------------------------------------------------
# JSON-safe encoding shared by generator and tests
# ---------------------------------------------------------------------------
def encode(v):
    if isinstance(v, (list, tuple)):
        return {'__seq__': [encode(x) for x in v]}
    if isinstance(v, np.ndarray) or isinstance(v, (float, int, np.floating, np.integer, bool, np.bool_)):
        a = np.asarray(v, dtype=float)
        return {'shape': list(a.shape), 'data': [repr(float(x)) for x in a.ravel()]}
    return {'repr': repr(v)}


def decode(e):
    if '__seq__' in e:
        return [decode(x) for x in e['__seq__']]
    if 'data' in e:
        return np.array([float(x) for x in e['data']], dtype=float).reshape(e['shape'])
    return e['repr']


def run_case(f):
    try:
        return {'status': 'ok', 'value': encode(f())}
    except BaseException as exc:  # noqa: B902 - PanicException derives from BaseException
        return {'status': 'error', 'error': '%s: %s' % (type(exc).__name__, str(exc)[:200])}
