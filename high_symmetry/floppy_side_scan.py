"""Floppy-side splitting with a rigid scaffold, computed with the separate implementation
independent_check/engine.py (SM Section S2.2 and Table S4).

On 11 low-index lines and 5 planes, find wavevectors where the floppy net has more mechanisms than
its planar vertices supply while the scaffold is rigid, and test whether the fixed-scaffold first-order
spectrum keeps the generic flat values there.

  python high_symmetry/floppy_side_scan.py --rcsr DIR [--pairs bsn:lvt]
"""
import argparse
import os
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "independent_check"))
sys.path.insert(0, str(ROOT))
import engine as V
from common.composite import dump

PAIRS = [('bcu', 'nbo'), ('bsn', 'lvt'), ('hxg-d', 'hxg'), ('nia', 'nia-d'), ('pcu', 'urk'), ('qtz-x', 'uom'), ('she-d', 'she'),
         ('toc-e', 'hxg'), ('snl', 'lvt'), ('reo', 'urk'), ('sta', 'nia-d')]
DIRS = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, -1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1), (1, 1, -1), (2, 1, 0), (1, 1, 2)]
PLANES = {'qz=0': lambda a, b: (a, b, 0), 'qx=0': lambda a, b: (0, a, b), 'qz=1/2': lambda a, b: (a, b, .5),
          'qx=qy': lambda a, b: (a, a, b), 'qx=-qy': lambda a, b: (a, -a, b)}


def nullspace_dims(fl, unitF):
    stars = [[] for _ in range(fl['V'])]
    for b, (i, j, s) in enumerate(fl['edges']): stars[i].append(unitF[b]); stars[j].append(-unitF[b])
    return [int(np.sum(np.linalg.eigvalsh(np.array(st).T @ np.array(st)) < 1e-9)) for st in stars]


def scan(cg, sc_name, fl_name):
    out, sc, fl, P, S, unitF, unitS = V.screen_pair(cg, sc_name, fl_name)
    nd = nullspace_dims(fl, unitF); nP = sum(nd)
    unit_own, _ = V.bond_geometry(fl); nd_own = nullspace_dims(fl, unit_own)
    generic = np.array(out['bands']); MF = out['mech_F']
    qs = []
    for d in DIRS:
        for t in np.linspace(0, .5, 65)[1:]: qs.append(tuple(t * np.array(d)))
    for nm, f in PLANES.items():
        for a in np.linspace(-.5, .5, 21):
            for b in np.linspace(-.5, .5, 21): qs.append(f(a, b))
    n_extra = 0; n_extra_rigid = 0; n_break_rigid = 0; wit = []
    for q in qs:
        if np.linalg.norm((np.array(q) + .5) % 1 - .5) < 1e-9: continue
        RF = V.rigidity(fl, q, unitF); mF = V.n_mech(RF)
        if mF <= nP: continue
        n_extra += 1
        mS = V.n_mech(V.rigidity(sc, q, unitS))
        if mS > 0: continue
        n_extra_rigid += 1
        w = np.linalg.eigvalsh(V.kernel(RF).conj().T @ S @ V.kernel(RF))
        wl = list(np.round(w, 6)); kept = 0
        for gval in generic:
            hit = [k for k, x in enumerate(wl) if abs(x - gval) < 1e-5]
            if hit: wl.pop(hit[0]); kept += 1
        if kept < MF:
            n_break_rigid += 1
            if len(wit) < 3: wit.append(dict(q=[round(float(x), 4) for x in q], M_F=mF, M_S=mS, kept=kept, spectrum=np.round(np.sort(w), 5).tolist()))
    return dict(M_F_generic=MF, n_planar_common_cell=nP, nullspace_dims=sorted(set(nd)), n_planar_own_cell=sum(nd_own),
                n_q_scanned=len(qs), n_q_MF_gt_P=n_extra, n_of_those_with_rigid_scaffold=n_extra_rigid,
                n_break_with_rigid_scaffold=n_break_rigid, witnesses=wit)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pairs', nargs='*', help='scaffold:floppy (default: the 11 listed pairs)')
    ap.add_argument('--rcsr', help='directory with rcsr3d.cgd (default: $RCSR_DIR)')
    ap.add_argument('--out', default=str(ROOT / 'outputs' / 'high_symmetry'))
    args = ap.parse_args()
    d = args.rcsr or os.environ.get('RCSR_DIR')
    if not d: raise SystemExit('RCSR files needed: pass --rcsr DIR or set RCSR_DIR (see README).')
    cg = V.parse_cgd(str(Path(d) / 'rcsr3d.cgd'))
    todo = [tuple(x.split(':')) for x in args.pairs] if args.pairs else PAIRS
    res = {}
    for a, b in todo:
        res[f'{a}|{b}'] = scan(cg, a, b); print(a, b, res[f'{a}|{b}'], flush=True)
    dump(Path(args.out) / 'floppy_side_scan.json', res)


if __name__ == '__main__':
    main()
