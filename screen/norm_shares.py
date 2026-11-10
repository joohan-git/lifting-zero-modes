"""Norm shares of the mechanisms on the vertex classes of a floppy net (SM Section S2.4).

For each degree class c of the floppy net, the eigenvalues of Pi P_c Pi on ker R_F(q) at 200 random
wavevectors (P_c projects on the vertices of class c). A constant value is a fixed norm share;
fcu (x) flu gives 2/3 on the fourfold class.

  python screen/norm_shares.py --rcsr DIR
"""
import argparse
import collections
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import nets as E
from common import composite as C
from common.records import find

PAIRS = [('fcu', 'flu'), ('she-d', 'she'), ('acs', 'gra')]


def shares(sc, fl, nq=200):
    p = C.pair(sc, fl, find(sc, fl)['shift']); f = p['fl']; pf = C.prep(f)
    deg = collections.Counter()
    for i, j, s in f['edges']: deg[i] += 1; deg[j] += 1
    classes = sorted(set(deg.values())); rng = np.random.default_rng(3); out = {}
    vals = {c: [] for c in classes}; lifted = []
    for q in rng.random((nq, 3)):
        Z, _ = C.null(C.compat(pf, q))
        for c in classes:
            Pc = np.diag(np.repeat([1.0 if deg[i] == c else 0.0 for i in range(f['n'])], 3))
            vals[c].append(np.linalg.eigvalsh(Z.conj().T @ Pc @ Z))
        lifted.append(np.linalg.eigvalsh(Z.conj().T @ p['S'] @ Z))
    for c in classes:
        v = np.concatenate(vals[c]); out[f'degree_{c}'] = dict(min=float(v.min()), max=float(v.max()))
    lv = np.concatenate(lifted)
    return dict(nq=nq, classes=classes, norm_share=out, lifted_min_over_kc=float(lv.min()), lifted_max_over_kc=float(lv.max()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=str(ROOT / 'outputs' / 'screen'))
    E.add_argument(ap); args = ap.parse_args(); E.apply_argument(args)
    res = {}
    for sc, fl in PAIRS:
        res[f'{sc}|{fl}'] = shares(sc, fl); print(sc, fl, res[f'{sc}|{fl}'], flush=True)
    C.dump(Path(args.out) / 'norm_shares.json', res)


if __name__ == '__main__':
    main()
