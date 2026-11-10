"""Second-order bandwidth of nbo caged by bcu, with and without axial bonds (main text; SM Section S4,
Table S3).

One Im-3m cubic cell (a = 1): bcu at 2a, nbo at 6b, contacts nbo-bcu at a/2 along the square normals.
Unit bond stiffness, contact stiffness k_c. At 1,500 random wavevectors the six full-composite modes
with the largest weight on the nbo normals are taken; the deviation |omega^2/k_c - 2| is summarized.
'bcu+axial' adds bonds along the cube axes to the bcu scaffold.

  python second_order/nbo_scaling.py [--nq 1500]
"""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
bcu = [np.array(p) for p in [(0, 0, 0), (.5, .5, .5)]]
nbo = [np.array(p) for p in [(0, .5, .5), (.5, 0, .5), (.5, .5, 0), (.5, 0, 0), (0, .5, 0), (0, 0, .5)]]
X = bcu + nbo; tag = ['B'] * 2 + ['N'] * 6; n = len(X)
shifts = [np.array(s) for s in itertools.product([-1, 0, 1], repeat=3)]


def bonds_of(kind, dist):
    out = []
    for i, j in itertools.product(range(n), range(n)):
        if not tag[i] == tag[j] == kind: continue
        for s in shifts:
            if abs(np.linalg.norm(X[j] + s - X[i]) - dist) < 1e-6 and (i < j or (i == j and tuple(s) > (0, 0, 0))):
                out.append((i, j, s))
    return out


B_bcu = bonds_of('B', np.sqrt(3) / 2)
B_nbo = bonds_of('N', 0.5)
B_con = [(i, j, s) for i in range(n) for j in range(n) if tag[i] == 'N' and tag[j] == 'B'
         for s in shifts if abs(np.linalg.norm(X[j] + s - X[i]) - 0.5) < 1e-6]
B_ax = [(i, i, s) for i in range(n) if tag[i] == 'B' for s in [np.array(v) for v in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]]]
assert (len(B_bcu), len(B_nbo), len(B_con)) == (8, 12, 12)


def build(q, k_c, k_ax=0.0):
    D = np.zeros((3 * n, 3 * n), complex)
    for blist, k in [(B_bcu, 1.0), (B_nbo, 1.0), (B_con, k_c), (B_ax, k_ax)]:
        for i, j, s in blist:
            d = X[j] + s - X[i]; u = d / np.linalg.norm(d); ph = np.exp(2j * np.pi * q @ s); Kb = k * np.outer(u, u)
            si = slice(3 * i, 3 * i + 3); sj = slice(3 * j, 3 * j + 3)
            D[si, si] += Kb; D[sj, sj] += Kb; D[si, sj] -= Kb * ph; D[sj, si] -= Kb * np.conj(ph)
    return D


normals = []
for i in range(n):
    if tag[i] == 'N':
        halves = [a for a in range(3) if abs(X[i][a] - 0.5) < 1e-9]
        nrm = [a for a in range(3) if a not in halves][0] if len(halves) == 2 else halves[0]
        v = np.zeros(3 * n); v[3 * i + nrm] = 1; normals.append(v)
P = np.array(normals).T


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--nq', type=int, default=1500)
    ap.add_argument('--out', default=str(ROOT / 'outputs' / 'second_order'))
    args = ap.parse_args()
    rng = np.random.default_rng(4); qs = rng.random((args.nq, 3))
    dist = np.linalg.norm((qs + 0.5) % 1 - 0.5, axis=1)
    out = {}
    for lab, kax in [('bcu', 0.0), ('bcu+axial', 1.0)]:
        res = {}
        for k_c in [0.001, 0.003, 0.01, 0.03]:
            vals = []; dq = []
            for q, dd in zip(qs, dist):
                w2, V = np.linalg.eigh(build(q, k_c, kax))
                idx = np.argsort(-np.sum(np.abs(P.T @ V) ** 2, axis=0))[:6]
                vals.append(w2[idx] / k_c); dq.append(np.full(6, dd))
            v = np.concatenate(vals); d = np.concatenate(dq); dev = np.abs(v - 2.0)
            res[str(k_c)] = dict(q05_q50_q95=[round(float(x), 4) for x in np.percentile(v, [5, 50, 95])],
                                 frac_within_5pct_of_2kc=round(float(np.mean(dev < 0.1)), 4),
                                 frac_within_1pct=round(float(np.mean(dev < 0.02)), 4),
                                 median_dev_far_from_Gamma=round(float(np.median(dev[d > 0.25])), 5))
        out[lab] = res
    print(json.dumps(out, indent=1))
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / 'nbo_scaling.json').write_text(json.dumps(out, indent=1), encoding='utf8')


if __name__ == '__main__':
    main()
