"""Rank-resolved count d_a = dim(E_a cap ker R_F(q)) = n_a - rank(R_F(q) B_a) for every row of the counting table
(SM Section S2.7): the ten flat dual pairs, the five flat pairs of the first prospective test and the planar pair.

Pairs are rebuilt as in counting.py (same records, shifts, common primitive cell); eigenvalues of S are clustered
with the same tolerance (1e-8 ||S||_2), so each entry aligns with one on-site eigenvalue of counting.json. B_a is an
orthonormal basis of the eigenspace E_a. 12 generic wavevectors (seed 26110707); singular values above
1e-8 max(1, ||R_F||_2) count towards the rank, which must not vary between wavevectors.

  python counting/rank_count.py --rcsr DIR [--out DIR]       (stored result: counting/rank_count.json)
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import scipy.linalg as la
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import counting as K
from counting import C, E

TOL = 1e-8
Q_SEED = 26110707
DEFINITION = ('dim(E_a cap ker R_F(q)) = n_a - rank(R_F(q) B_a); eigenvalue clusters of S at 1e-8 ||S||_2 as counting.py; '
              '12 generic q, seed 26110707, rank threshold 1e-8 max(1, ||R_F||_2); pairs rebuilt by counting/counting.py')


def clusters(w, scale):
    groups = []
    for i in np.argsort(w):
        if not groups or abs(w[i] - np.mean(w[groups[-1]])) > 1e-8 * max(scale, 1e-300): groups.append([i])
        else: groups[-1].append(i)
    return groups


def rank_resolved(p):
    f = p['fl']; d = f['d']; S = p['S']; scale = float(la.norm(S, 2))
    w, V = la.eigh(S); groups = clusters(w, scale)
    pf = C.prep(f); recs = None
    for q in np.random.default_rng(Q_SEED).random((12, d)):
        R = C.compat(pf, q); thr = TOL * max(1.0, la.norm(R, 2)); r = []
        for idx in groups:
            sv = la.svdvals(R @ V[:, idx]); rk = int(np.sum(sv > thr))
            r.append(dict(value=float(np.mean(w[idx])), n=len(idx), rank=rk, dim=len(idx) - rk))
        assert recs is None or [(x['n'], x['rank']) for x in r] == [(x['n'], x['rank']) for x in recs], 'rank varies with q'
        recs = r
    return recs


def rows():
    """The pairs of counting.rows(), keeping the composites."""
    built = []; calc = K.calculate
    K.calculate = lambda label, p, *a: built.append((label, p)) or calc(label, p, *a)
    try:
        K.rows()
    finally:
        K.calculate = calc
    return {label: rank_resolved(p) for label, p in built}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=str(HERE.parent / 'outputs' / 'counting'))
    E.add_argument(ap); args = ap.parse_args(); E.apply_argument(args)
    out = rows()
    for label, r in out.items():
        print(label.replace('⊗', 'x'), [(round(x['value'], 5), x['n'], x['rank'], x['dim']) for x in r], flush=True)
    p = Path(args.out) / 'rank_count.json'; p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(definition=DEFINITION, rows=out), indent=1), encoding='utf8'); print('written', p)


if __name__ == '__main__':
    main()
