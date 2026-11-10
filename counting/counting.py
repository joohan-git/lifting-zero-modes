"""Counting rule of Calugaru et al. (Nat. Phys. 18, 185, 2022; App. I.4) against the flat bands found (SM Table S-counting).

For each flat pair: the on-site contact tensor S = blockdiag_i sum_contacts n n^T; for every eigenvalue a of S
with multiplicity n_a the rule guarantees b_a = max(0, n_a - E_F) flat bands at a (E_F: floppy edges, the
strong bars; not the weak contacts). The first-order flat values are computed at 64 generic wavevectors
(seed 20261132) in the common primitive cell and compared with the eigenvalues of S.

Classes: (i) the count accounts for every flat band; (ii) partial count; (iii) the count is zero and every
flat value is an eigenvalue of S; (iv) the count is zero and at least one flat value is not.

Pairs: the ten exactly flat dual pairs, the five flat pairs of the first prospective test, and the planar
pair hxl (x) hcb (written hcb (x) hxl in the article's planar section).

  python counting/counting.py --rcsr DIR [--out DIR]
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import scipy.linalg as la
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT, ROOT / 'classifier', ROOT / 'drive'):
    sys.path.insert(0, str(p))
from common import nets as E
from common import composite as C
from common.records import screen_records
import flat_classifier as classifier
from pipeline import reduce_pair
from gain import assemble_analytic

Q_SEED = 20261132
DUAL_FLAT = [0, 5, 10, 15, 34, 42, 53, 58, 61, 65]      # screen indices of the ten exactly flat dual pairs
DEFINITION = ('N_L=dV, N_Ltilde=E_F (strong F bars, not weak contacts); A_k=S=blockdiag sum_contact(n n^T). '
              'Dimensionless energy=omega^2/k_c. Bound per S eigenvalue a: max(0,n_a-E_F).')
PROOF = ('dim(E_a(S) intersect ker R(q)) >= n_a-rank R(q) >= n_a-E. A common vector is an exact a eigenvector of the chiral '
         'block Hamiltonian with onsite S and an exact k_c*a eigenvector of R^H R+k_c S. Does not prove exactness for a moving scaffold.')
CELL = ('Counts are in the displayed common primitive cell; conventional counts multiply by conventional_multiplicity. '
        'No constraint rank substituted for E in the literature bound.')


def clusters(values, scale):
    groups = []
    for x in sorted(values):
        if not groups or abs(x - np.mean(groups[-1])) > 1e-8 * max(scale, 1e-300): groups.append([float(x)])
        else: groups[-1].append(float(x))
    return [dict(value=float(np.mean(g)), multiplicity=len(g)) for g in groups]


def calculate(label, p, cohort, identifier, route):
    f = p['fl']; d = f['d']; n = d * f['n']; e = f['e']; S = p['S']; scale = float(la.norm(S, 2))
    onsite = clusters(la.eigvalsh(S), scale)
    for a in onsite: a['count_lower_bound'] = max(0, a['multiplicity'] - e)
    vals = []; ranks = []
    for q in np.random.default_rng(Q_SEED).random((64, d)):
        Z = classifier.kernel(C.compat(C.prep(f), q))
        ranks.append(n - Z.shape[1]); vals.append(la.eigvalsh(Z.conj().T @ S @ Z))
    assert len(set(map(len, vals))) == 1
    vals = np.array(vals); assert np.max(np.ptp(vals, axis=0)) < 1e-8 * scale
    actual = clusters(np.mean(vals, axis=0), scale)
    for a in actual:
        a['is_S_eigenvalue'] = any(abs(a['value'] - b['value']) <= 1e-8 * scale for b in onsite)
        a['count_lower_bound_at_value'] = sum(b['count_lower_bound'] for b in onsite if abs(a['value'] - b['value']) <= 1e-8 * scale)
    total = sum(a['count_lower_bound'] for a in onsite); m = vals.shape[1]
    status = 'i' if total == m else 'ii' if total > 0 else 'iii'
    offsite = any(not a['is_S_eigenvalue'] for a in actual)
    return dict(pair=label, cohort=cohort, identifier=identifier, route=route, d=d, V=f['n'],
                N_L=n, N_Ltilde=e, cell='common primitive' if d == 3 else 'primitive 2D',
                conventional_multiplicity=p.get('cell_multiplicity', 1),
                onsite_eigenvalues=onsite, actual_flat_bands=actual, count_total=total, M_F=m,
                generic_rank=sorted(set(ranks)), nu=n - e, N_ss=[e - r for r in sorted(set(ranks))],
                classification='iv' if offsite and status == 'iii' else status,
                counting_status=status, has_flat_value_not_in_S=offsite,
                max_branch_range=float(np.max(np.ptp(vals, axis=0))), q_seed=Q_SEED, nq=64)


def rows():
    out = []; rec = screen_records()
    retro = json.loads((ROOT / 'classifier' / 'predictions_retrospective.json').read_text(encoding='utf8'))['rows']
    for ix in DUAL_FLAT:
        r = rec[ix]; p = reduce_pair(C.pair(r['scaffold'], r['floppy'], r['shift']))
        out.append(calculate(r['scaffold'] + ' ⊗ ' + r['floppy'], p, 'dual', ix, retro[ix]['prediction']['route']))
    pros = json.loads((ROOT / 'classifier' / 'predictions_prospective.json').read_text(encoding='utf8'))['rows']
    outs = {o['id']: o for o in json.loads((ROOT / 'classifier' / 'outcomes_prospective.json').read_text(encoding='utf8'))['rows']}
    for r in pros:
        o = outs[r['id']]
        if o.get('qualified') and o.get('shape_exact_flat'):
            p = reduce_pair(C.pair(r['scaffold'], r['floppy'], r['shift']))
            out.append(calculate(r['scaffold'] + ' ⊗ ' + r['floppy'], p, 'test_1', r['id'], r['prediction']['route']))
    assert sum(r['cohort'] == 'test_1' for r in out) == 5
    p = assemble_analytic('hcb'); p['S'] = np.zeros((4, 4))
    for i, j, s, v, length in p['contacts']: p['S'][2 * i:2 * i + 2, 2 * i:2 * i + 2] += np.outer(v, v)
    out.append(calculate('hxl ⊗ hcb (2D; manuscript hcb⊗hxl)', p, '2D_reference', None, 'scalar_S'))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=str(ROOT / 'outputs' / 'counting'))
    E.add_argument(ap); args = ap.parse_args(); E.apply_argument(args)
    out = rows(); d = Path(args.out)
    C.dump(d / 'counting.json', dict(definition=DEFINITION, proof=PROOF, cell_warning=CELL, rows=out)); C.csvout(d / 'counting.csv', out)
    for r in out: print(r['pair'].replace('⊗', 'x'), r['N_L'], r['N_Ltilde'], r['count_total'], r['classification'], flush=True)


if __name__ == '__main__':
    main()
