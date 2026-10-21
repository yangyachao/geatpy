"""optimize(progress=True) draws a one-line progress bar and leaves the results unchanged."""
import contextlib
import io

import numpy as np

import geatpy as ea


class Sphere(ea.Problem):
    def __init__(self):
        ea.Problem.__init__(self, 'Sphere', 1, [1], 4, [0] * 4, [-1] * 4, [1] * 4)

    def evalVars(self, X):
        return (X ** 2).sum(1, keepdims=True)


def _run(progress, verbose=False):
    alg = ea.soea_DE_rand_1_bin_templet(Sphere(), ea.Population('RI', NIND=10), MAXGEN=30, logTras=10)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        res = ea.optimize(alg, seed=2, verbose=verbose, drawing=0, outputMsg=False, drawLog=False,
                          saveFlag=False, progress=progress)
    return res, out.getvalue()


def test_progress_bar_does_not_change_results():
    (a, out_a), (b, out_b) = _run(False), _run(True)
    assert out_a == ''
    np.testing.assert_array_equal(a['ObjV'], b['ObjV'])
    np.testing.assert_array_equal(a['Vars'], b['Vars'])
    assert out_b.startswith('\r[') and out_b.endswith('100.0%  gen 30  ' + out_b.rsplit('  ', 1)[1])
    assert out_b.endswith('s\n')


def test_progress_bar_gives_way_to_the_log_table():
    _, out = _run(True, verbose=True)
    _, plain = _run(False, verbose=True)
    # erasing the bar leaves the table exactly as without it, apart from carriage returns and spaces
    strip = lambda s: [line.split('\r')[-1].rstrip() for line in s.split('\n')]
    assert [l for l in strip(out) if l] == [l for l in strip(plain) if l]
