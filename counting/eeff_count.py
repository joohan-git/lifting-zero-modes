"""Effective-row count per eigenvalue of S for the ten flat dual pairs, in the common primitive cell (SM Section S2.7).

For each eigenvalue a of S with eigenspace basis B_a: E_eff, the number of rows of R_F(q) B_a that do not vanish;
the rank of R_F(q) B_a; d_a = n_a - rank; and the bound max(0, n_a - E_eff). Four random wavevectors per level
(one generator, seed 11, drawn in order over the pairs). The rank-resolved count of rank_count.py is stored next to
each pair as ref.

  python counting/eeff_count.py --rcsr DIR [--out DIR]      (stored result: counting/eeff_count.json)
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import scipy.linalg as la
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'classifier'))
from common import nets as E
from common import composite as C

PAIRS = [('bcu', 'nbo'), ('hxg-d', 'hxg'), ('nia', 'nia-d'), ('she-d', 'she'), ('bsn', 'lvt'), ('qtz-x', 'uom'),
         ('pcu', 'urk'), ('hex', 'bnn'), ('fcu', 'flu'), ('acs', 'gra')]


def count():
    from sector_tools import build
    ds = json.loads((ROOT / 'screen' / 'dual_screen.json').read_text(encoding='utf8'))['rows']
    rc = json.loads((HERE / 'rank_count.json').read_text(encoding='utf8'))['rows']
    out = {}; rng = np.random.default_rng(11)
    for sc, fl in PAIRS:
        r = [x for x in ds if x.get('scaffold') == sc and x.get('floppy') == fl][0]
        p = build(sc, fl, r['shift']); S = p['S']
        w, V = la.eigh(S); vals = []
        for x in w:
            if not vals or abs(x - vals[-1][0]) > 1e-7: vals.append([x, []])
        groups = [(float(v), V[:, np.where(abs(w - v) < 1e-7)[0]]) for v, _ in vals]
        pf = C.prep(p['fl']); res = []
        for v, B in groups:
            ee, rk = [], []
            for q in rng.random((4, 3)):
                RB = C.compat(pf, q) @ B
                ee.append(int(np.sum(la.norm(RB, axis=1) > 1e-10)))
                sv = la.svd(RB, compute_uv=False) if RB.size else np.zeros(0)
                rk.append(int(np.sum(sv > 1e-8 * max(1, sv[0] if len(sv) else 1))))
            res.append(dict(a=round(v, 6), n=B.shape[1], E_eff=sorted(set(ee)), rank=sorted(set(rk)), dim=B.shape[1] - max(rk),
                            bound=max(0, B.shape[1] - max(ee))))
        key = f'{sc} ⊗ {fl}'
        out[key] = dict(E=p['fl']['e'], levels=res, ref=rc.get(key))
        print(sc, fl, 'E', p['fl']['e'], [(x['a'], x['n'], x['E_eff'], x['dim'], x['bound']) for x in res], flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=str(ROOT / 'outputs' / 'counting'))
    E.add_argument(ap); args = ap.parse_args(); E.apply_argument(args)
    out = count(); p = Path(args.out) / 'eeff_count.json'; p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding='utf8'); print('written', p)


if __name__ == '__main__':
    main()
