"""Mean coordination of the scaffold nets of the unbiased control (controls/unbiased_control.json, 32 floppy nets x
(1 dual + 3 non-dual cages)) against rigidity (SM Section S3.4 and main text).

Mean coordination: vertex degree averaged over the cell, 2E/V of the RCSR graph (the same in any cell). Threshold 6:
rigid / total cages with mean coordination >= 6 and < 6 for dual and non-dual cages, and Fisher's exact test of dual
against non-dual among the cages with mean coordination >= 6.

  python controls/unbiased_coordination.py --rcsr DIR [--out DIR]   (stored result: controls/unbiased_coordination.json)
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from scipy.stats import fisher_exact
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import nets as E, composite as C


def coordination_rows():
    u = json.loads((HERE / 'unbiased_control.json').read_text(encoding='utf8')); rows = []
    for r in u['rows']:
        g = C.graph(r['scaffold']); deg = np.zeros(g['n'])
        for i, j, s in g['edges']: deg[i] += 1; deg[j] += 1
        rows.append(dict(floppy=r['floppy'], scaffold=r['scaffold'], kind=r['kind'], rigid=bool(r['rigid']), scaffold_V=g['n'],
                         scaffold_E=len(g['edges']), mean_coordination=float(deg.mean()), mean_coordination_2E_over_V=2 * len(g['edges']) / g['n'],
                         degree_min=int(deg.min()), degree_max=int(deg.max()), scaffold_M=r['scaffold_M']))
    return rows


def tab(rows, thr):
    out = {}
    for kind in ('dual', 'nondual'):
        for side, f in ((f'>={thr}', lambda z: z >= thr), (f'<{thr}', lambda z: z < thr)):
            sel = [x for x in rows if x['kind'] == kind and f(x['mean_coordination'])]
            out[f'{kind} {side}'] = dict(rigid=sum(x['rigid'] for x in sel), total=len(sel))
    hi = {k: out[f'{k} >={thr}'] for k in ('dual', 'nondual')}
    p = fisher_exact([[hi['dual']['rigid'], hi['dual']['total'] - hi['dual']['rigid']],
                      [hi['nondual']['rigid'], hi['nondual']['total'] - hi['nondual']['rigid']]])[1]
    return dict(cells=out, fisher_p_dual_vs_nondual_among_high_coordination=float(p))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=str(HERE.parent / 'outputs' / 'controls'))
    E.add_argument(ap); args = ap.parse_args(); E.apply_argument(args)
    rows = coordination_rows(); T = tab(rows, 6)
    res = dict(definition='mean degree = 2E/V of RCSR graph (graph() conventional cell; identical in any cell); threshold 6',
               table=T, n_rows=len(rows), n_distinct_scaffolds=len({x['scaffold'] for x in rows}),
               coordination_exactly_6_count=sum(abs(x['mean_coordination'] - 6) < 1e-9 for x in rows),
               min_over_threshold_margin=sorted(set(round(x['mean_coordination'], 3) for x in rows if 5.5 < x['mean_coordination'] < 6.5)),
               rows=rows)
    p = Path(args.out) / 'unbiased_coordination.json'; p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1), encoding='utf8')
    print(json.dumps(T, indent=1)); print('written', p)


if __name__ == '__main__':
    main()
