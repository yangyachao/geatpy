# -*- coding: utf-8 -*-
"""Record the behaviour of whichever geatpy core is importable into a parity JSON file.

usage: python gen_golden.py OUT.json [--reps N] [--skip-templates] [--merge]

--merge keeps the sections of an existing OUT.json that are not regenerated (e.g. templates).
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'tests'))
import numpy as np  # noqa: E402
import geatpy as ea  # noqa: E402
import parity_cases as pc  # noqa: E402


def main():
    out = sys.argv[1]
    reps = int(sys.argv[sys.argv.index('--reps') + 1]) if '--reps' in sys.argv else 5
    res = {'core': getattr(ea, '__version__', '?'), 'numpy': np.__version__, 'deterministic': {}, 'random': {},
           'templates': {}}
    if '--merge' in sys.argv and os.path.exists(out):
        with open(out) as fh:
            res['templates'] = json.load(fh).get('templates', {})
    for name, f in pc.deterministic_cases(ea).items():
        res['deterministic'][name] = pc.run_case(f)
    for name, (f, tol) in pc.random_stats(ea).items():
        r = pc.run_case(f)
        r['tol'] = tol
        res['random'][name] = r
    if '--skip-templates' not in sys.argv:
        res['templates'] = {}
        for key, name, mk, enc, nind, gen in pc.template_runs(ea):
            vals, err, t0 = [], None, time.time()
            for _ in range(reps):
                try:
                    vals.append(pc.run_template(ea, name, mk, enc, nind, gen))
                except Exception as exc:
                    err = '%s: %s' % (type(exc).__name__, str(exc)[:200])
                    break
            res['templates'][key] = {'values': vals, 'error': err, 'seconds': time.time() - t0}
            print(key, 'error' if err else 'median=%.4g' % np.median(vals), flush=True)
    with open(out, 'w') as fh:
        json.dump(res, fh, indent=1, sort_keys=True)


if __name__ == '__main__':
    main()
