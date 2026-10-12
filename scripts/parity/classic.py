# -*- coding: utf-8 -*-
"""Classic benchmark comparison: run the same template/problem pairs on two geatpy installs.

usage:
  python -I classic.py list                     # case names
  python -I classic.py run CASE SEED            # one run, prints a JSON line
  python classic.py cmp A.jsonl B.jsonl         # compare two result files

Single-objective cases report the best objective value; multi-objective cases report IGD and HV
against the problem's reference front. Run it from outside the repository (or with -I) so the
installed geatpy is used. Python 3.6 compatible.
"""
import json
import sys
import time
import warnings

warnings.filterwarnings('ignore')


def _problems(ea):
    """Problems not in ea.benchmarks: maximisation, mixed integers, a constrained single objective."""
    import numpy as np

    class MaxRastrigin(ea.Problem):  # maximise -Rastrigin (optimum 0)
        def __init__(self, Dim=10):
            ea.Problem.__init__(self, 'MaxRastrigin', 1, [-1], Dim, [0] * Dim, [-5.12] * Dim, [5.12] * Dim,
                                [1] * Dim, [1] * Dim)

        def evalVars(self, x):
            return -(10 * x.shape[1] + (x ** 2 - 10 * np.cos(2 * np.pi * x)).sum(1, keepdims=True))

    class MixedRastrigin(ea.Problem):  # half the variables are integers in [-5, 5] (optimum 0)
        def __init__(self, Dim=10):
            ea.Problem.__init__(self, 'MixedRastrigin', 1, [1], Dim, [0] * (Dim // 2) + [1] * (Dim - Dim // 2),
                                [-5] * Dim, [5] * Dim, [1] * Dim, [1] * Dim)

        def evalVars(self, x):
            return 10 * x.shape[1] + (x ** 2 - 10 * np.cos(2 * np.pi * x)).sum(1, keepdims=True)

    class ConSphere(ea.Problem):  # min sum((x-1)^2) s.t. sum(x) <= 5 (optimum 2.5 at x = 0.5)
        def __init__(self, Dim=10):
            ea.Problem.__init__(self, 'ConSphere', 1, [1], Dim, [0] * Dim, [-10] * Dim, [10] * Dim,
                                [1] * Dim, [1] * Dim)

        def evalVars(self, x):
            return ((x - 1) ** 2).sum(1, keepdims=True), x.sum(1, keepdims=True) - 5

    class MaxZDT1(ea.Problem):  # ZDT1 with both objectives negated and maximised
        def __init__(self, Dim=30):
            self.base = ea.benchmarks.ZDT1(Dim=Dim)  # before Problem.__init__, which may ask for the front
            ea.Problem.__init__(self, 'MaxZDT1', 2, [-1, -1], Dim, [0] * Dim, [0] * Dim, [1] * Dim,
                                [1] * Dim, [1] * Dim)

        def evalVars(self, x):
            return -self.base.evalVars(x)

        def calReferObjV(self):
            return -self.base.calReferObjV()

    return {'MaxRastrigin': MaxRastrigin, 'MixedRastrigin': MixedRastrigin, 'ConSphere': ConSphere,
            'MaxZDT1': MaxZDT1}


def _p(name, *args):
    return lambda ea: _problems(ea)[name](*args)


# name: (template, problem factory, encoding, NIND, MAXGEN). The encoding may also be
# ('psy', [encodings], [variable indices per encoding]) or ('multi', encoding, number of populations).
CASES = [
    ('ackley_DE_rand_1_bin', 'soea_DE_rand_1_bin_templet', lambda ea: ea.benchmarks.Ackley(30), 'RI', 50, 1000),
    ('rosenbrock_DE_best_1_L', 'soea_DE_best_1_L_templet', lambda ea: ea.benchmarks.Rosenbrock(30), 'RI', 50, 1000),
    ('rastrigin_SEGA_BG', 'soea_SEGA_templet', lambda ea: ea.benchmarks.Rastrigrin(30), 'BG', 50, 1000),
    ('griewangk_studGA_RI', 'soea_studGA_templet', lambda ea: ea.benchmarks.Griewangk(30), 'RI', 50, 1000),
    ('schwefel_EGA_RI', 'soea_EGA_templet', lambda ea: ea.benchmarks.Schwefel(30), 'RI', 50, 1000),
    ('sphere_ES_1_plus_1', 'soea_ES_1_plus_1_templet', lambda ea: ea.benchmarks.Sphere(30), 'RI', 1, 3000),
    ('att48_SEGA_P', 'soea_SEGA_templet', lambda ea: ea.benchmarks.TSP('att48'), 'P', 100, 1000),
    ('att48_studGA_P', 'soea_studGA_templet', lambda ea: ea.benchmarks.TSP('att48'), 'P', 100, 1000),
    ('zdt1_NSGA2', 'moea_NSGA2_templet', lambda ea: ea.benchmarks.ZDT1(), 'RI', 100, 200),
    ('zdt3_NSGA2', 'moea_NSGA2_templet', lambda ea: ea.benchmarks.ZDT3(), 'RI', 100, 200),
    ('zdt4_NSGA2', 'moea_NSGA2_templet', lambda ea: ea.benchmarks.ZDT4(), 'RI', 100, 500),
    ('dtlz1_NSGA3', 'moea_NSGA3_templet', lambda ea: ea.benchmarks.DTLZ1(3), 'RI', 92, 500),
    ('dtlz2_MOEAD', 'moea_MOEAD_templet', lambda ea: ea.benchmarks.DTLZ2(3), 'RI', 100, 300),
    ('dtlz2_RVEA', 'moea_RVEA_templet', lambda ea: ea.benchmarks.DTLZ2(3), 'RI', 105, 300),
    ('c1dtlz1_NSGA3', 'moea_NSGA3_templet', lambda ea: ea.benchmarks.C1_DTLZ1(3), 'RI', 92, 500),
    ('bnh_NSGA2', 'moea_NSGA2_templet', lambda ea: ea.benchmarks.BNH(), 'RI', 100, 200),
    # paths the cases above do not reach
    ('maxrastrigin_SEGA_RI', 'soea_SEGA_templet', _p('MaxRastrigin'), 'RI', 50, 500),
    ('maxrastrigin_DE_rand_1_bin', 'soea_DE_rand_1_bin_templet', _p('MaxRastrigin'), 'RI', 50, 500),
    ('maxzdt1_NSGA2', 'moea_NSGA2_templet', _p('MaxZDT1'), 'RI', 100, 200),
    ('maxzdt1_MOEAD', 'moea_MOEAD_templet', _p('MaxZDT1'), 'RI', 100, 200),
    ('mixedint_SEGA_RI', 'soea_SEGA_templet', _p('MixedRastrigin'), 'RI', 50, 500),
    ('mixedint_DE_rand_1_bin', 'soea_DE_rand_1_bin_templet', _p('MixedRastrigin'), 'RI', 50, 500),
    ('mixedint_SEGA_BG', 'soea_SEGA_templet', _p('MixedRastrigin'), 'BG', 50, 500),
    ('consphere_SEGA_RI', 'soea_SEGA_templet', _p('ConSphere'), 'RI', 50, 500),
    ('consphere_DE_best_1_L', 'soea_DE_best_1_L_templet', _p('ConSphere'), 'RI', 50, 500),
    ('rastrigin_SEGA_RI_odd', 'soea_SEGA_templet', lambda ea: ea.benchmarks.Rastrigrin(10), 'RI', 51, 500),
    ('zdt1_NSGA2_odd', 'moea_NSGA2_templet', lambda ea: ea.benchmarks.ZDT1(), 'RI', 101, 200),
    ('att48_SGA_P', 'soea_SGA_templet', lambda ea: ea.benchmarks.TSP('att48'), 'P', 100, 1000),
    ('att48_EGA_P', 'soea_EGA_templet', lambda ea: ea.benchmarks.TSP('att48'), 'P', 100, 1000),
    ('att48_steadyGA_P', 'soea_steadyGA_templet', lambda ea: ea.benchmarks.TSP('att48'), 'P', 100, 1000),
    ('att48_GGAP_SGA_P', 'soea_GGAP_SGA_templet', lambda ea: ea.benchmarks.TSP('att48'), 'P', 100, 1000),
    ('rastrigin_multi_SEGA', 'soea_multi_SEGA_templet', lambda ea: ea.benchmarks.Rastrigrin(30), ('multi', 'RI', 3),
     30, 500),
    ('rastrigin_psy_SEGA', 'soea_psy_SEGA_templet', lambda ea: ea.benchmarks.Rastrigrin(10),
     ('psy', ['BG', 'RI'], [list(range(5)), list(range(5, 10))]), 50, 500),
    ('rastrigin_psy_studGA', 'soea_psy_studGA_templet', lambda ea: ea.benchmarks.Rastrigrin(10),
     ('psy', ['RI', 'BG'], [list(range(5)), list(range(5, 10))]), 50, 500),
    ('zdt1_psy_NSGA2', 'moea_psy_NSGA2_templet', lambda ea: ea.benchmarks.ZDT1(),
     ('psy', ['RI', 'BG'], [list(range(15)), list(range(15, 30))]), 100, 200),
    ('dtlz2_psy_NSGA3', 'moea_psy_NSGA3_templet', lambda ea: ea.benchmarks.DTLZ2(3),
     ('psy', ['RI', 'BG'], [list(range(6)), list(range(6, 12))]), 91, 200),
]


def _population(ea, problem, enc, nind):
    if isinstance(enc, str):
        return ea.Population(Encoding=enc, NIND=nind)
    if enc[0] == 'multi':  # the algorithm does not create the fields of a population list
        field = ea.crtfld(enc[1], problem.varTypes, problem.ranges, problem.borders)
        return [ea.Population(enc[1], field, nind) for _ in range(enc[2])]
    fields = (problem.varTypes, problem.ranges, problem.borders)
    return ea.PsyPopulation(enc[1], fields, NIND=nind, EncoIdxs=enc[2])


def run(name, seed):
    import numpy as np
    import geatpy as ea
    tpl, mk, enc, nind, gen = [c[1:] for c in CASES if c[0] == name][0]
    problem = mk(ea)
    alg = getattr(ea, tpl)(problem, _population(ea, problem, enc, nind), MAXGEN=gen, logTras=0)
    t = time.time()
    res = ea.optimize(alg, seed=seed, verbose=False, drawing=0, outputMsg=False, drawLog=False, saveFlag=False)
    out = {'case': name, 'seed': seed, 'version': ea.__version__, 'time': time.time() - t}
    objv = res['ObjV']
    out['feasible'] = bool(res['success'])
    if tpl.startswith('soea'):
        out['best'] = float(objv[0][0]) if objv is not None and len(objv) else None
    else:
        pf = problem.getReferObjV()
        ok = objv is not None and len(objv) > 0
        out['IGD'] = float(ea.indicator.IGD(objv, pf)) if ok else None
        out['HV'] = float(ea.indicator.HV(objv, pf)) if ok else None
    print(json.dumps(out))


def _rank_z(a, b):
    """Mann-Whitney U as a z score (positive: b tends to be larger)."""
    allv = sorted([(x, 0) for x in a] + [(x, 1) for x in b])
    rank, i = [0.0] * len(allv), 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        for k in range(i, j):
            rank[k] = (i + j + 1) / 2.0
        i = j
    n1, n2 = len(a), len(b)
    r2 = sum(rank[k] for k, (_, g) in enumerate(allv) if g == 1)
    u = r2 - n2 * (n2 + 1) / 2.0
    return (u - n1 * n2 / 2.0) / ((n1 * n2 * (n1 + n2 + 1) / 12.0) ** 0.5)


def cmp(fa, fb):
    import numpy as np

    def load(f):
        rows = {}
        for line in open(f):
            if line.startswith('{'):
                r = json.loads(line)
                rows.setdefault(r['case'], []).append(r)
        return rows
    a, b = load(fa), load(fb)
    va = list(a.values())[0][0]['version']
    vb = list(b.values())[0][0]['version']
    print('A = %s, B = %s; median (IQR); z: Mann-Whitney, positive = B larger; |z| > 3 flagged' % (va, vb))
    print('%-24s %-6s %4s %24s %24s %7s %9s' % ('case', 'metric', 'n', 'A', 'B', 'z', 'time B/A'))
    for name in [c[0] for c in CASES]:
        if name not in a or name not in b:
            continue
        for m in ('best', 'IGD', 'HV'):
            xa = [r[m] for r in a[name] if r.get(m) is not None]
            xb = [r[m] for r in b[name] if r.get(m) is not None]
            if not xa or not xb:
                continue
            qa, qb = np.percentile(xa, [25, 50, 75]), np.percentile(xb, [25, 50, 75])
            z = _rank_z(xa, xb)
            ta = np.median([r['time'] for r in a[name]])
            tb = np.median([r['time'] for r in b[name]])
            print('%-24s %-6s %4d %24s %24s %7.2f %8.2fx%s' % (
                name, m, min(len(xa), len(xb)), '%.4g (%.3g-%.3g)' % (qa[1], qa[0], qa[2]),
                '%.4g (%.3g-%.3g)' % (qb[1], qb[0], qb[2]), z, tb / ta, '  <--' if abs(z) > 3 else ''))


if __name__ == '__main__':
    if sys.argv[1] == 'list':
        print('\n'.join(c[0] for c in CASES))
    elif sys.argv[1] == 'run':
        run(sys.argv[2], int(sys.argv[3]))
    elif sys.argv[1] == 'cmp':
        cmp(sys.argv[2], sys.argv[3])
