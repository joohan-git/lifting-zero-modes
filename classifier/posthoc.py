"""Post hoc diagnosis of the prospective flat pairs on which the classifier abstained (SM S3.6).
Does not change the classifier. For each such pair: the q-independent (compact) part of ker R_F,
the variation of the mechanism projector over q, and the sector composition of the remainder.

  python classifier/posthoc.py --rcsr DIR
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import scipy.linalg as la
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from common import nets as E
from common import composite as C
import flat_classifier as classifier
from pipeline import reduce_pair


def compact_kernel(p):
    g = p['fl']; rows = []; n = 3 * g['n']
    for b, (i, j, s) in enumerate(g['edges']):
        u = g['unit'][b]; v = np.zeros(n); v[3 * i:3 * i + 3] -= u
        if any(s):
            rows.append(v); v = np.zeros(n); v[3 * j:3 * j + 3] += u; rows.append(v)
        else:
            v[3 * j:3 * j + 3] += u; rows.append(v)
    return classifier.kernel(np.array(rows))


def diagnose(r, o):
    p = reduce_pair(C.pair(r['scaffold'], r['floppy'], r['shift'])); Z = compact_kernel(p)
    qs = np.random.default_rng(20261130).random((8, 3)); zs = [classifier.kernel(C.compat(C.prep(p['fl']), q)) for q in qs]
    projectors = [z @ z.conj().T for z in zs]; spread = max(la.norm(P - projectors[0], 2) for P in projectors)
    bonds, T, S, normals = classifier.local_data(p['fl'], p['stars'])
    localnull = [3 - np.linalg.matrix_rank(np.array(bb), tol=1e-8 * la.svdvals(np.array(bb))[0]) for bb in bonds]
    row = dict(id=r['id'], scaffold=r['scaffold'], floppy=r['floppy'], frozen_reason=r['prediction'].get('reason'), M_F=o['M_F'],
               compact_q_independent_kernel_dimension=Z.shape[1], projector_variation_operator_norm=float(spread),
               local_null_dimensions=localnull, local_contact_eigenvalues=[la.eigvalsh(s).tolist() for s in S],
               local_bond_eigenvalues=[la.eigvalsh(t).tolist() for t in T], generic_values=o['branch_means'])
    if Z.shape[1] and Z.shape[1] < o['M_F'][0]:
        classes = {}
        for i in range(p['fl']['n']):
            classes.setdefault(tuple(np.round(np.r_[la.eigvalsh(T[i]), la.eigvalsh(S[i])], 8)), []).append(i)
        classes = list(classes.values()); extended = []
        for z in zs:
            residual = z - Z @ (Z.conj().T @ z); u, sv, v = la.svd(residual, full_matrices=False)
            W = u[:, sv > 1e-8]; weights = []; axial = 0.
            for i in range(p['fl']['n']):
                w, V = la.eigh(T[i]); axis = V[:, 0] if abs(w[1] - w[2]) < 1e-8 else V[:, 2]
                axial += float(la.norm(axis @ W[3 * i:3 * i + 3]) ** 2); weights.append(float(la.norm(W[3 * i:3 * i + 3]) ** 2))
            group = []
            for ids in classes:
                coords = np.array([3 * i + j for i in ids for j in range(3)]); block = W[coords].conj().T @ W[coords]
                fraction = float(np.trace(block).real / W.shape[1])
                group.append(dict(sites=ids, norm_fraction=fraction, operator_residual=float(la.norm(block - fraction * np.eye(W.shape[1]), 2))))
            extended.append(dict(site_norm_weights=weights, group_norm_fractions=group, axial_norm_squared=axial,
                                 values=la.eigvalsh(W.conj().T @ p['S'] @ W).tolist(), compact_cross_norm=float(la.norm(Z.conj().T @ p['S'] @ W))))
        row.update(compact_values=la.eigvalsh(Z.conj().T @ p['S'] @ Z), extended_sector_diagnostics=extended,
                   explanation=f'Route L: {len(normals)} planar sites do not exhaust {o["M_F"][0]} mechanisms; the remaining mechanisms occupy in-plane sectors with different axes, which no route covers.')
    if Z.shape[1] == o['M_F'][0]:
        row['explanation'] = 'A q-independent compact kernel spans all mechanisms; its support extends over several vertices, which the single-site route L does not cover.'
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=str(HERE.parent / 'outputs' / 'classifier'))
    E.add_argument(ap); args = ap.parse_args(); E.apply_argument(args)
    preds = json.loads((HERE / 'predictions_prospective.json').read_text(encoding='utf8'))['rows']
    outs = {o['id']: o for o in json.loads((HERE / 'outcomes_prospective.json').read_text(encoding='utf8'))['rows']}
    rows = []
    for r in preds:
        o = outs.get(r['id'])
        if not o or not o.get('qualified') or not o['shape_exact_flat'] or r['prediction']['prediction'] == 'predicted_flat': continue
        rows.append(diagnose(r, o)); print(r['scaffold'], r['floppy'], rows[-1]['compact_q_independent_kernel_dimension'], flush=True)
    C.dump(Path(args.out) / 'prospective_flat_abstentions.json', rows)


if __name__ == '__main__':
    main()
