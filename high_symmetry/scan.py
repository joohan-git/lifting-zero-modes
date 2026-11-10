"""Dense scans of high-symmetry lines, planes and zone faces for the flat and near-flat pairs
(Methods; SM Section S3.2 and Table S4).

At every wavevector: scaffold mechanisms M_S; for the exactly flat and small near-flat pairs (and
for all scaffold-zero points) the floppy mechanisms, the fixed-scaffold spectrum and, where the
scaffold has mechanisms, the joint first-order spectrum of the contacts on ker R_F + ker R_S.

  python high_symmetry/scan.py --rcsr DIR --pairs bcu:nbo [--max-q 300]
"""
import argparse
import itertools
import json
import sys
from pathlib import Path
import numpy as np
import scipy.linalg as la
from scipy.spatial import Voronoi
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import nets as E
from common import composite as C
from common.records import screen_records, pair_id

TOL = C.TOL


def selected():
    good = set(json.loads((ROOT / 'screen' / 'dual_screen.json').read_text(encoding='utf8'))['summary']['near_flat'])
    return [r for r in screen_records() if r['cohort'] == 'dual' and r.get('scaffold', '') + '|' + r.get('floppy', '') in good]


def make_q(p):
    A = p['sc']['A']; qs = {}

    def add(q, label, detail=False):
        key = tuple(np.round(q, 10)); entry = qs.setdefault(key, [np.array(q), set(), False]); entry[1].add(label); entry[2] |= detail
    # low-index lines and planes in fractional axes of the conventional cell
    normals = [np.array(v, float) for v in itertools.product((-1, 0, 1), repeat=3) if any(v) and next(x for x in v if x) != -1]
    for v in normals:
        label = 'line_' + ','.join(map(str, v.astype(int)))
        for j, t in enumerate(np.linspace(-.5, .5, 129)): add(t * v, label, j % 16 == 0)
        k = np.argmax(abs(v)); ij = [i for i in range(3) if i != k]
        for off in (0, .5):
            label = 'plane_' + ','.join(map(str, v.astype(int))) + '=' + str(off)
            for ia, a in enumerate(np.linspace(-.5, .5, 25)):
                for ib, b in enumerate(np.linspace(-.5, .5, 25)):
                    q = np.zeros(3); q[ij] = [a, b]; q[k] = (off - v[ij] @ q[ij]) / v[k]
                    if abs(q[k]) <= .50000001: add(q, label, ia % 6 == 0 and ib % 6 == 0)
    # faces and edges of the primitive Wigner-Seitz zone
    _, B = E.sg_ops(C.CG()[p['sc']['name']], p['sc']['setting']); Ap = B @ A; recip = la.inv(Ap).T
    lattice = np.array(list(itertools.product(range(-2, 3), repeat=3))); xyz = lattice @ recip
    center = int(np.where(np.all(lattice == 0, axis=1))[0][0]); vor = Voronoi(xyz)
    for ridge, (a, b) in zip(vor.ridge_vertices, vor.ridge_points):
        if center not in (a, b) or -1 in ridge: continue
        poly = vor.vertices[ridge]; c = poly.mean(0); normal = xyz[b if a == center else a]; n = normal / la.norm(normal)
        e = poly[0] - c; e /= la.norm(e); f = np.cross(n, e)
        poly = poly[np.argsort(np.arctan2((poly - c) @ f, (poly - c) @ e))]
        h = lattice[b if a == center else a]; label = 'BZ_face_' + ','.join(map(str, h))
        for v, w in zip(poly, np.roll(poly, -1, axis=0)):
            for u in (v, (v + w) / 2, c):
                for j, t in enumerate(np.linspace(0, 1, 65)): add(t * u @ A.T, 'BZ_radial_' + label, j % 8 == 0)
            for ia in range(13):
                for ib in range(13 - ia): add(((1 - (ia + ib) / 12) * c + ia / 12 * v + ib / 12 * w) @ A.T, label, ia % 3 == 0 and ib % 3 == 0)
            for j, t in enumerate(np.linspace(0, 1, 65)): add(((1 - t) * v + t * w) @ A.T, 'BZ_edge_' + label, j % 8 == 0)
    return list(qs.values())


def scan_one(r, gen, max_q=None):
    p = C.pair(r['scaffold'], r['floppy'], r['shift'])
    qlist = make_q(p); pf, ps = C.prep(p['fl']), C.prep(p['sc'])
    refs = np.array(gen['branch_means']); m0 = gen['M_F'][0]
    data = dict(scaffold=r['scaffold'], floppy=r['floppy'], rows=[], total_q=len(qlist),
                dense_floppy_kernel=bool(p['fl']['n'] <= 28), rank_relative_tolerance=TOL,
                sampling='129 per low-index line; 25x25 low-index planes at offset 0,1/2; primitive Wigner-Seitz faces barycentric denominator 12 and edges 65')
    for q, labels, detail in qlist[:max_q]:
        R = C.compat(ps, q); sv = la.svdvals(R, check_finite=False); rank = int(np.sum(sv > TOL * max(1, sv[0]))); ms = ps[-1] - rank
        row = dict(q=np.round(q, 10).tolist(), strata=sorted(labels), M_S=ms, sigma_min_S=float(sv[-1]))
        if p['fl']['n'] <= 28 or ms or detail:
            Z, rf = C.null(C.compat(pf, q)); w = la.eigvalsh(Z.conj().T @ p['S'] @ Z); mf = Z.shape[1]
            row.update(M_F=mf, self_stress_F=pf[0].size // 3 - rf, fixed_zero=int(np.sum(w < 1e-8)),
                       fixed_min=float(w.min()), fixed_max=float(w.max()))
            if ms:
                W, rs = C.null(R); ZZ = la.block_diag(Z, W); CW = C.contact_matrix(p, q) @ ZZ
                joint = la.eigvalsh(CW.conj().T @ CW)
                row.update(joint_min=float(joint.min()), joint_max=float(joint.max()), joint_zero=int(np.sum(joint < 1e-8)),
                           joint_spectrum=np.round(joint, 11).tolist())
                # generic flat eigenvalues (with multiplicity) not retained in the joint spectrum
                if gen['spread_rel'] < 1e-7:
                    remaining = list(joint); missing = 0
                    for x in refs:
                        k = int(np.argmin(abs(np.array(remaining) - x))) if remaining else -1
                        if k >= 0 and abs(remaining[k] - x) < 2e-6: remaining.pop(k)
                        else: missing += 1
                    row['generic_flat_multiplicity_missing'] = missing
            if mf != m0 or (ms and gen['spread_rel'] < 1e-7): row['fixed_spectrum'] = np.round(w, 11).tolist()
        data['rows'].append(row)
    data['complete'] = max_q is None or max_q >= len(qlist)
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pairs', nargs='*', help='scaffold:floppy (default: the 17 flat and near-flat pairs)')
    ap.add_argument('--max-q', type=int, help='stop after this many wavevectors (quick test)')
    ap.add_argument('--generic', default=str(Path(__file__).parent / 'generic'), help='directory of generic spectra')
    ap.add_argument('--out', default=str(ROOT / 'outputs' / 'high_symmetry' / 'scans'))
    E.add_argument(ap); args = ap.parse_args(); E.apply_argument(args)
    want = {tuple(x.split(':')) for x in args.pairs} if args.pairs else None
    for r in selected():
        if want and (r['scaffold'], r['floppy']) not in want: continue
        stem = pair_id(r['scaffold'], r['floppy'])
        gen = json.loads((Path(args.generic) / f'{stem}.json').read_text(encoding='utf8'))
        data = scan_one(r, gen, args.max_q)
        C.dump(Path(args.out) / f'{stem}.json', data)
        print(stem, len(data['rows']), 'of', data['total_q'], 'wavevectors', flush=True)


if __name__ == '__main__':
    main()
