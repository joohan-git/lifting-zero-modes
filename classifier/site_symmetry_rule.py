"""Symmetry-only rule on the screened pairs (main text, retrospective test; SM Section S2).

For every floppy vertex: site group in the composite, contact tensor S_i, bond tensor T_i and the
irreducible decomposition of the vector representation. The rule designates a pair flat when S is a
common scalar, when every vertex is planar with its normal an eigenvector of S_i, or when one of the
bipartite or uniaxial structural patterns of Section S2 holds; it abstains when a sector repeats.
It reads geometry only, never a lifted spectrum.

  python classifier/site_symmetry_rule.py --rcsr DIR [--pairs bcu:nbo]
  python classifier/site_symmetry_rule.py count      # counts from the stored files
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np
import scipy.linalg as la
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from common import nets as E
from common import composite as C
from common.records import screen_records


def prediction(p, sites):
    f = p['fl']; ss = np.array([x['S_eigenvalues'] for x in sites]); tt = np.array([x['T_eigenvalues'] for x in sites])
    low = any(x['max_irrep_multiplicity'] > 1 for x in sites)
    reduced = any(x['site_group_single'] != x['site_group_composite'] for x in sites)
    route = None; values = []; typ = 'unclassified'
    if np.max(ss) - np.min(ss) < 1e-7:
        route = 'global_scalar_S'; values = [float(ss.mean())]; typ = '(iii)'
    elif all(x['planar'] and x['normal_S_eigenvector'] for x in sites):
        route = 'local_normals'; values = sorted(set(round(x['normal_stiffness'], 10) for x in sites)); typ = '(i)'
    else:
        col = C.bipartition(f)
        if col is not None and np.max(np.ptp(ss, axis=1)) < 1e-7 and np.max(np.ptp(tt, axis=1)) < 1e-7:
            tv = [tt[col == k].mean() for k in (0, 1)]; sv = [ss[col == k].mean() for k in (0, 1)]
            if all(np.ptp(tt[col == k]) < 1e-7 and np.ptp(ss[col == k]) < 1e-7 for k in (0, 1)):
                route = 'bipartite_isotropic'; values = [float((tv[1] * sv[0] + tv[0] * sv[1]) / sum(tv))]; typ = '(iii)+norm fixing'
    if route is None:
        # equal uniaxial tensors; axial self-translation bonds remove the axial component at generic q
        T = C.tensors(f); w, v = la.eigh(T[0])
        for k in range(3):
            axis = v[:, k]; P = np.eye(3) - np.outer(axis, axis)
            axial = all(abs(np.dot(u, axis)) < 1e-7 or abs(np.dot(u, axis)) > 1 - 1e-7 for u in f['unit'])
            anchored = set(i for b, (i, j, s) in enumerate(f['edges']) if i == j and abs(f['unit'][b] @ axis) > 1 - 1e-7 and any(s))
            vals = [np.trace(P @ p['S'][3 * i:3 * i + 3, 3 * i:3 * i + 3]) / 2 for i in range(f['n'])]
            scalar = all(la.norm(P @ p['S'][3 * i:3 * i + 3, 3 * i:3 * i + 3] @ P - vals[i] * P) < 1e-7 for i in range(f['n']))
            if axial and len(anchored) == f['n'] and scalar and np.ptp(vals) < 1e-7:
                route = 'axial_self_translation'; values = [float(vals[0])]; typ = '(ii)'; break
    if route is None:
        # sector version of norm fixing: planar A sites, transverse A-B bars, axial periodic B-B chains
        T = C.tensors(f); Aids = [i for i, x in enumerate(sites) if x['planar']]; Bids = [i for i in range(f['n']) if i not in Aids]
        if Aids and Bids:
            _, v = la.eigh(T[Aids[0]]); axis = v[:, 0]; P = np.eye(3) - np.outer(axis, axis)
            valid = all(la.norm(T[i] @ axis) < 1e-7 for i in Aids)
            classes = np.array([0 if i in Aids else 1 for i in range(f['n'])]); adj = {i: [] for i in Bids}
            for ib, (i, j, sh) in enumerate(f['edges']):
                axial = abs(f['unit'][ib] @ axis) > 1 - 1e-7; transverse = abs(f['unit'][ib] @ axis) < 1e-7
                if classes[i] != classes[j]: valid &= transverse
                elif classes[i] == 1 and axial:
                    adj[i].append((j, np.array(sh))); adj[j].append((i, -np.array(sh)))
                else: valid = False
            seen = set()
            for root in Bids:
                if root in seen: continue
                stack = [root]; pot = {root: np.zeros(3, int)}; periodic = False; seen.add(root)
                while stack:
                    i = stack.pop()
                    for j, sh in adj[i]:
                        if j not in seen: seen.add(j); pot[j] = pot[i] + sh; stack.append(j)
                        elif np.any(pot[i] + sh - pot[j]): periodic = True
                valid &= periodic
            tv = []; sv = []
            for ids in (Aids, Bids):
                tvals = [np.trace(P @ T[i]) / 2 for i in ids]; svals = [np.trace(P @ p['S'][3 * i:3 * i + 3, 3 * i:3 * i + 3]) / 2 for i in ids]
                valid &= np.ptp(tvals) < 1e-7 and np.ptp(svals) < 1e-7
                for i, tval, sval in zip(ids, tvals, svals):
                    Si = p['S'][3 * i:3 * i + 3, 3 * i:3 * i + 3]
                    valid &= la.norm(P @ T[i] @ P - tval * P) < 1e-7 and la.norm(P @ Si @ P - sval * P) < 1e-7 and la.norm(P @ Si @ axis) < 1e-7
                tv.append(float(np.mean(tvals))); sv.append(float(np.mean(svals)))
            if valid:
                plane = (tv[1] * sv[0] + tv[0] * sv[1]) / sum(tv)
                values = sorted(set([float(plane)] + [float(axis @ p['S'][3 * i:3 * i + 3, 3 * i:3 * i + 3] @ axis) for i in Aids]))
                route = 'sector_bipartite_axial_chains'; typ = 'mixed (i)+norm fixing'
    if route is None and any(x['planar'] for x in sites): typ = 'mixed/unresolved'
    label = 'predicted_flat' if route and not low else ('unpredictable_multiplicity' if low else 'unpredictable_sector')
    return dict(prediction=label, geometry_route=route, conditional_predicted_values=values, type=typ, symmetry_reduced=reduced,
                note='Failure of the sufficient conditions is not a proof of dispersion.')


def effective(geo):
    """A local-normal designation needs at least as many planar vertices as Maxwell mechanisms."""
    if geo.get('geometry_route') == 'local_normals' and 3 * geo['V_F'] - geo['E_F'] > geo['V_F']:
        return 'unpredictable_extra_mechanisms'
    return geo.get('prediction', 'unpredictable_geometry')


def count(rows, records):
    out = Counter()
    for r, ge in zip(records, rows):
        if 'scaffold' not in r: continue
        eligible = r['gen_mech_scaffold'] == 0 and r['unlifted_mech_median'] == 0
        if r['cohort'] != 'dual' or not eligible: continue
        actual = 'exact_flat' if r['band_spread_rel'] < 1e-4 else ('near_flat' if r['band_spread_rel'] < .05 else 'dispersive')
        out[(effective(ge), actual)] += 1
    return [dict(predicted=a, actual=b, count=n) for (a, b), n in sorted(out.items())]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', nargs='?', default='run', choices=['run', 'count'])
    ap.add_argument('--pairs', nargs='*', help='scaffold:floppy (default: all 326 screened pairs)')
    ap.add_argument('--out', default=str(ROOT / 'outputs' / 'classifier'))
    E.add_argument(ap); args = ap.parse_args(); E.apply_argument(args)
    records = screen_records()
    if args.stage == 'count':
        rows = json.loads((HERE / 'site_symmetry.json').read_text(encoding='utf8'))['rows']
        print('dual pairs with a rigid, fully lifting cage (rule outcome, screen outcome, count):')
        for x in count(rows, records): print(' ', x)
        return
    want = {tuple(x.split(':')) for x in args.pairs} if args.pairs else None
    rows = []
    for r in records:
        a, b = r.get('scaffold', r['a']), r.get('floppy', r['b'])
        if want and (a, b) not in want: continue
        try:
            p = C.pair(a, b, r.get('shift', [0, 0, 0])); sites = C.symmetry(p); pred = prediction(p, sites)
            row = dict(cohort=r['cohort'], key=r['key'], scaffold=a, floppy=b, shift=p['shift'], V_F=p['fl']['n'], E_F=p['fl']['e'],
                       V_S=p['sc']['n'], E_S=p['sc']['e'], space_group_floppy=C.CG()[b]['group'], space_group_scaffold=C.CG()[a]['group'],
                       n_contacts=len(p['contacts']), sites=sites, **pred)
        except Exception as e:
            row = dict(cohort=r['cohort'], key=r['key'], error=f'{type(e).__name__}: {e}')
        rows.append(row); print(a, b, row.get('prediction'), row.get('geometry_route'), row.get('error', ''), flush=True)
    C.dump(Path(args.out) / 'site_symmetry.json', dict(rows=rows))


if __name__ == '__main__':
    main()
