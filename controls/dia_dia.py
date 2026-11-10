"""Homo-interpenetrated control dia (x) dia (main text; SM Section S5.1 and Table S7).

Two dia nets, the second shifted by (1/2,1/2,1/2) in the cubic cell (cuprite arrangement), unit
bond stiffness, contacts of stiffness k_c between nearest vertices of the two nets. Full composite
spectra at 300 random wavevectors; quantiles of the low (lifted) modes in units of k_c.

  python controls/dia_dia.py
"""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
fcc = [np.array(v) for v in [(0, 0, 0), (0, .5, .5), (.5, 0, .5), (.5, .5, 0)]]
d1 = [f for f in fcc] + [f + .25 for f in fcc]
d2 = [(p + .5) % 1 for p in d1]
P = d1 + d2; tag = [1] * 8 + [2] * 8; n = 16
sh = [np.array(s) for s in itertools.product([-1, 0, 1], repeat=3)]


def bonds(dist, cond):
    out = []
    for i, j in itertools.product(range(n), repeat=2):
        if not cond(i, j): continue
        for s in sh:
            if abs(np.linalg.norm(P[j] + s - P[i]) - dist) < 1e-6 and (i < j or (i == j and tuple(s) > (0, 0, 0))): out.append((i, j, s))
    return out


L = np.sqrt(3) / 4
B1 = bonds(L, lambda i, j: tag[i] == tag[j] == 1); B2 = bonds(L, lambda i, j: tag[i] == tag[j] == 2)
dmin = min(np.linalg.norm(P[j] + s - P[i]) for i in range(8) for j in range(8, 16) for s in sh)
CON = bonds(dmin, lambda i, j: tag[i] != tag[j])


def D(q, k_c):
    M = np.zeros((3 * n, 3 * n), complex)
    for blist, k in [(B1, 1.0), (B2, 1.0), (CON, k_c)]:
        for i, j, s in blist:
            d = P[j] + s - P[i]; u = d / np.linalg.norm(d); Kb = k * np.outer(u, u); ph = np.exp(2j * np.pi * q @ s)
            si = slice(3 * i, 3 * i + 3); sj = slice(3 * j, 3 * j + 3)
            M[si, si] += Kb; M[sj, sj] += Kb; M[si, sj] -= Kb * ph; M[sj, si] -= Kb * np.conj(ph)
    return M


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=str(ROOT / 'outputs' / 'controls'))
    args = ap.parse_args()
    print('bonds', len(B1), len(B2), 'contacts', len(CON), 'nearest contact', round(dmin, 4))
    rng = np.random.default_rng(1); out = {}
    for k_c in [0.003, 0.01, 0.03]:
        low = []; zeros = []
        for q in rng.random((300, 3)):
            w = np.linalg.eigvalsh(D(q, k_c)); low += list(w[w < 0.3] / k_c); zeros.append(int(np.sum(w < 1e-9)))
        v = np.array(low); v = v[v < 20]
        out[str(k_c)] = dict(unlifted_median=int(np.median(zeros)), n_low=len(v), q05_q50_q95=[round(float(x), 4) for x in np.percentile(v, [5, 50, 95])],
                             hist=np.histogram(v, bins=[0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 6, 10, 20])[0].tolist())
    print(json.dumps(out, indent=1))
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / 'dia_dia.json').write_text(json.dumps(out, indent=1), encoding='utf8')


if __name__ == '__main__':
    main()
