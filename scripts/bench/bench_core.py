# -*- coding: utf-8 -*-
"""Time where the evolution goes: objective evaluation, the Rust core, and the rest of Python.

usage:
  python -I bench_core.py [CASE ...]        # all cases when none is given

Each case runs one template with logging, plotting and output off, under cProfile, and splits the
time into evalVars, the built-in functions of _geatpy_core, and everything else. The test functions
are cheap numpy code, so the core's share is higher than with a real (slow) objective. Set
GEATPY_NUM_THREADS (or call ea.set_num_threads) to compare thread counts; run it from outside the
repository (or with -I) so the installed geatpy is used. Python 3.6 compatible.
"""
import cProfile
import pstats
import sys
import time
import warnings

warnings.filterwarnings('ignore')

CASES = [
    # name, template, problem, N, MAXGEN
    ('NSGA2_N100', 'moea_NSGA2_templet', 'DTLZ2', 100, 300),
    ('NSGA2_N1000', 'moea_NSGA2_templet', 'DTLZ2', 1000, 60),
    ('NSGA2_N5000', 'moea_NSGA2_templet', 'DTLZ2', 5000, 12),
    ('NSGA2_ZDT1_N1000', 'moea_NSGA2_templet', 'ZDT1', 1000, 100),  # two objectives, one big front
    ('NSGA3_N105', 'moea_NSGA3_templet', 'DTLZ2', 105, 300),
    ('NSGA3_N1035', 'moea_NSGA3_templet', 'DTLZ2', 1035, 40),
    ('NSGA3_N5050', 'moea_NSGA3_templet', 'DTLZ2', 5050, 10),
    ('DE_N100', 'soea_DE_rand_1_bin_templet', 'Rastrigrin', 100, 500),
    ('DE_N10000', 'soea_DE_rand_1_bin_templet', 'Rastrigrin', 10000, 20),
]


def run(case):
    import numpy as np
    import geatpy as ea
    name, templet, prob, n, gens = case
    np.random.seed(0)
    problem = {'DTLZ2': lambda: ea.benchmarks.DTLZ2(M=3, Dim=12), 'ZDT1': lambda: ea.benchmarks.ZDT1(),
               'Rastrigrin': lambda: ea.benchmarks.Rastrigrin(Dim=30)}[prob]()
    algorithm = getattr(ea, templet)(problem, ea.Population(Encoding='RI', NIND=n), MAXGEN=gens, logTras=0)
    profiler = cProfile.Profile()
    start = time.perf_counter()
    profiler.enable()
    ea.optimize(algorithm, verbose=False, drawing=0, outputMsg=False, drawLog=False, saveFlag=False)
    profiler.disable()
    total = time.perf_counter() - start
    stats = pstats.Stats(profiler).stats
    evals = sum(v[3] for k, v in stats.items() if k[2] == 'evalVars')  # cumulative time
    core = {}
    for (path, _, func), v in stats.items():
        # operators show up as "<built-in method _geatpy_core.ndsortESS>", the indicators (always run
        # at the end of ea.optimize for M > 1) as "<built-in method indicator.HV>"
        if path == '~' and (func.startswith('<built-in method indicator.')
                            or (func.startswith('<built-in method ') and '_geatpy_core.' in func)):
            fn = func.rstrip('>').rsplit('.', 1)[-1]
            core[fn] = core.get(fn, 0) + v[2]
    core_total = sum(core.values())
    hot = sorted(core.items(), key=lambda kv: -kv[1])[:2]
    # cProfile slows the Python side down, so shares are of the profiled total
    profiled = sum(v[2] for v in stats.values())
    print('%-12s threads=%d  total %7.3f s | eval %4.0f%% | core %4.0f%% | other %4.0f%% | %s' % (
        name, ea.get_num_threads() if hasattr(ea, 'get_num_threads') else 1, total,
        100 * evals / profiled, 100 * core_total / profiled, 100 * (profiled - evals - core_total) / profiled,
        ', '.join('%s %.0f%%' % (k, 100 * t / profiled) for k, t in hot)))
    sys.stdout.flush()


if __name__ == '__main__':
    wanted = sys.argv[1:]
    for case in CASES:
        if not wanted or case[0] in wanted:
            run(case)
