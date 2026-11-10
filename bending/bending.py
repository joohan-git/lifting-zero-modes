"""Joint bending added at first order to the ten exactly flat pairs (main text; SM Section S7, Table S8).

Lifted spectrum eig[Z^H (S + K_b(q)) Z] on ker R_F(q), with K_b either angular springs (change of
cos theta between edges at a vertex) or beam-like straightening of collinear edges (transverse kink),
of stiffness k_b in units of k_c. Spread = mean band width over mean band value, 24 random wavevectors.

  python bending/bending.py --rcsr DIR [--pairs bcu:nbo nia:nia-d]
"""
import argparse
import itertools
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import nets as E
from common import composite as C
from common.records import find

PAIRS = [('acs', 'gra'), ('bcu', 'nbo'), ('hex', 'bnn'), ('bsn', 'lvt'), ('fcu', 'flu'), ('hxg-d', 'hxg'), ('nia', 'nia-d'),
         ('pcu', 'urk'), ('qtz-x', 'uom'), ('she-d', 'she')]
RATIOS = [0, 0.01, 0.1, 0.3, 1.0]


def incident(g):
    inc = {i: [] for i in range(g['n'])}
    for b, (i, j, s) in enumerate(g['edges']):
        v = g['unit'][b] * g['length'][b]
        inc[i].append((j, np.array(s), v)); inc[j].append((i, -np.array(s), -v))
    return inc


def bend_matrix(g, q, k_ang, k_beam):
    n = g['n']; inc = incident(g); rows = []; w = []
    for i in range(n):
        for (j, sj, a), (k, sk, b) in itertools.combinations(inc[i], 2):
            la_, lb = np.linalg.norm(a), np.linalg.norm(b); ua, ub = a / la_, b / lb; c = ua @ ub
            pj = np.exp(2j * np.pi * q @ sj); pk = np.exp(2j * np.pi * q @ sk)
            if c < -0.999:      # collinear: kink vector, two transverse components
                Pp = np.eye(3) - np.outer(ua, ua); basis = np.linalg.svd(Pp)[0][:, :2]
                for e in basis.T:
                    r = np.zeros(3 * n, complex)
                    r[3 * j:3 * j + 3] += e * pj / la_; r[3 * i:3 * i + 3] -= e / la_
                    r[3 * k:3 * k + 3] += e * pk / lb; r[3 * i:3 * i + 3] -= e / lb
                    rows.append(r); w.append(k_beam)
            else:               # angle: d cos theta
                r = np.zeros(3 * n, complex)
                ga = (ub - c * ua) / la_; gb = (ua - c * ub) / lb
                r[3 * j:3 * j + 3] += ga * pj; r[3 * i:3 * i + 3] -= ga
                r[3 * k:3 * k + 3] += gb * pk; r[3 * i:3 * i + 3] -= gb
                rows.append(r); w.append(k_ang)
    Rb = np.array(rows); W = np.array(w)
    return (Rb.conj().T * W) @ Rb


def spreads(sc, fl, rng):
    r = find(sc, fl); p = C.pair(sc, fl, r['shift']); f = p['fl']; pf = C.prep(f); res = {}
    rng.random((64, 3))     # the stream first serves the screen of the pair (2 x 12 rank samples, 40 spectra)
    for kind in ['angle', 'beam']:
        for rr in RATIOS:
            evs = []
            for _ in range(24):
                q = rng.random(3); Z, _ = C.null(C.compat(pf, q))
                if Z.shape[1] == 0: continue
                Kb = bend_matrix(f, q, rr if kind == 'angle' else 0.0, rr if kind == 'beam' else 0.0)
                evs.append(np.sort(np.linalg.eigvalsh(Z.conj().T @ (p['S'] + Kb) @ Z)))
            ev = np.array(evs)
            res[f'{kind}_{rr}'] = round(float(np.mean(ev.max(0) - ev.min(0)) / max(np.mean(ev), 1e-12)), 4)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pairs', nargs='*', help='scaffold:floppy (default: the ten exactly flat pairs)')
    ap.add_argument('--out', default=str(ROOT / 'outputs' / 'bending'))
    E.add_argument(ap); args = ap.parse_args(); E.apply_argument(args)
    todo = [tuple(x.split(':')) for x in args.pairs] if args.pairs else PAIRS
    rng = np.random.default_rng(4); out = {}
    for sc, fl in todo:
        out[f'{sc}|{fl}'] = spreads(sc, fl, rng); print(sc, fl, out[f'{sc}|{fl}'], flush=True)
    C.dump(Path(args.out) / 'bending.json', out)


if __name__ == '__main__':
    main()
