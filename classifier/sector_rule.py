"""Post hoc class-and-sector rule on the eligible pairs of both prospective tests (SM S3.7).

Written after the outcomes were known; reported as a diagnosis, not as a prospective result. Neither rule reads
the spectrum of Pi S Pi; both use the geometry, the mechanisms ker R_F(q) at 3 random wavevectors and S.

  sector rule  For every vertex split R^3 into the eigenspaces (sectors) of S_i; vertex classes are orbits refined
               by the S eigenvalues, so that each class-sector P_{c,sigma} has one value s_{c,sigma}. Then
               Pi S Pi = sum s_{c,sigma} Pi P_{c,sigma} Pi identically. The band is called flat when the Gram
               operators A_{c,sigma}(q) = Z(q)^H P_{c,sigma} Z(q) commute pairwise and their joint spectrum does not
               depend on q; the predicted values are sum s_{c,sigma} lambda_{c,sigma} over the joint eigenvectors.
  local rule   Coarser form: S_i compressed onto the local mechanism subspace of each vertex must be scalar
               (class-level fallback for rigid clusters), with commuting class Gram operators A_c(q) of
               q-independent joint spectrum.

Outcomes: test 1 stored outcomes; test 2 stored or computed outcomes (see sector_tools.eligible_pairs), and for
every eligible test-2 pair the 32-wavevector flatness test of pipeline.spectral_stats is repeated here.

  python classifier/sector_rule.py --rcsr DIR [--out DIR]
"""
import argparse
import sys
from collections import Counter
from pathlib import Path
import numpy as np
import scipy.linalg as la
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from common import nets as E
from common import composite as C
from pipeline import spectral_stats
from sector_tools import build, classes, kernel, sector_of, eligible_pairs

NQ, Q_SEED, MIX_SEED = 3, 7, 1


def _joint(Zs, gram, m, tol):
    """Commutation and q-independence of the joint spectrum of the Gram operators Z^H P Z; returns tuples or a reason."""
    tuples_q = []
    for Z in Zs:
        A = gram(Z); r = np.random.default_rng(MIX_SEED).random(len(A))
        for a in range(len(A)):
            for b in range(a):
                if la.norm(A[a] @ A[b] - A[b] @ A[a]) > tol: return None, 'gram_operators_do_not_commute'
        w, V = la.eigh(sum(x * Ac for x, Ac in zip(r, A)))
        tuples_q.append(sorted(tuple(np.round([np.real(V[:, k].conj() @ Ac @ V[:, k]) for Ac in A], 6)) for k in range(m)))
    for t in tuples_q[1:]:
        if np.max(np.abs(np.array(t) - np.array(tuples_q[0]))) > 1e-5: return None, 'joint_spectrum_varies_with_q'
    return tuples_q[0], None


def _kernels(p):
    qs = np.random.default_rng(Q_SEED).random((NQ, 3)); Zs = [kernel(p, q) for q in qs]; m = {Z.shape[1] for Z in Zs}
    return Zs, (m.pop() if len(m) == 1 and 0 not in m else None)


def sector_rule(p, tol=1e-6):
    g = p['classifier_graph']; n = g['n']
    cls, bonds, T, Sloc, normals = classes(p); ncls = int(cls.max()) + 1
    Zs, m = _kernels(p)
    if m is None: return dict(call='abstain', reason='kernel_dimension_varies_or_zero')
    projs = []; svals = []
    for c in range(ncls):
        ids = np.where(cls == c)[0]; w0 = la.eigvalsh(Sloc[ids[0]]); groups = []
        for k, a in enumerate(w0):
            if groups and abs(a - groups[-1][0]) < tol: groups[-1][1].append(k)
            else: groups.append([a, [k]])
        for a, ks in groups:
            P = np.zeros((3 * n, 3 * n))
            for i in ids:
                w, V = la.eigh(Sloc[i]); sel = [k for k in range(3) if abs(w[k] - a) < tol]
                if len(sel) != len(ks): return dict(call='abstain', reason='S_eigenvalues_differ_within_class')
                P[3 * i:3 * i + 3, 3 * i:3 * i + 3] = V[:, sel] @ V[:, sel].T
            projs.append(P); svals.append(float(a))
    tup, reason = _joint(Zs, lambda Z: [Z.conj().T @ P @ Z for P in projs], m, tol)
    if reason: return dict(call='abstain', reason=reason, class_sectors=len(projs))
    values = sorted(float(sum(s * lam for s, lam in zip(svals, t))) for t in tup)
    return dict(call='predicted_flat', values=values, class_sectors=len(projs))


def local_rule(p):
    g = p['classifier_graph']; n = g['n']; S = p['S']
    cls, bonds, T, Sloc, normals = classes(p)
    Zs, m = _kernels(p)
    if m is None: return dict(call='abstain', reason='kernel_dimension_varies_or_zero')
    s_i = {}; failed = set()
    for i in range(n):
        locs = np.concatenate([Z[3 * i:3 * i + 3].T for Z in Zs])
        if la.norm(locs) < 1e-9: continue
        Vb = sector_of(np.concatenate([locs.real, locs.imag]), tol=1e-6); d = Vb.shape[1]
        comp = Vb.T @ Sloc[i] @ Vb; val = np.trace(comp) / d
        if la.norm(comp - val * np.eye(d)) > 1e-6 * max(1., la.norm(Sloc[i])): failed.add(i)
        else: s_i[i] = val
    key = {}; new = np.full(n, -1)
    for i in range(n):
        k = (int(cls[i]), round(s_i.get(i, -1.), 6)); key.setdefault(k, len(key)); new[i] = key[k]
    cls = new; ncls = int(cls.max()) + 1; s_c = {}; idxs = []
    for c in range(ncls):
        ids = np.where(cls == c)[0]; idx = np.concatenate([np.arange(3 * i, 3 * i + 3) for i in ids]); idxs.append(idx)
        vals = [s_i[i] for i in ids if i in s_i]
        if not any(i in failed for i in ids):
            if vals: s_c[c] = float(np.mean(vals))
            continue
        ok = True; sv = []
        for Z in Zs:
            A = Z[idx].conj().T @ Z[idx]; B = Z[idx].conj().T @ S[np.ix_(idx, idx)] @ Z[idx]
            if la.norm(A) < 1e-12: continue
            s = np.trace(B).real / np.trace(A).real; sv.append(s)
            if la.norm(B - s * A) > 1e-6 * max(1., la.norm(B)): ok = False
        if not ok or np.ptp(sv) > 1e-6:
            return dict(call='abstain', reason='S_compression_not_scalar', vertices=[int(i) for i in ids])
        s_c[c] = float(np.mean(sv))
    tup, reason = _joint(Zs, lambda Z: [Z[idx].conj().T @ Z[idx] for idx in idxs], m, 1e-7)
    if reason: return dict(call='abstain', reason=reason)
    return dict(call='predicted_flat', values=sorted(float(sum(s_c.get(c, 0.) * lam for c, lam in enumerate(t))) for t in tup))


def summary(rows, key):
    calls = [r for r in rows if r[key]['call'] == 'predicted_flat']
    return dict(eligible=len(rows), flat=sum(r['flat'] for r in rows), dispersive=sum(not r['flat'] for r in rows),
                calls=len(calls), calls_matching_spectrum=sum(1 for r in calls if (r[key].get('max_deviation_from_spectrum') is not None and r[key]['max_deviation_from_spectrum'] < 1e-6)),
                calls_on_flat=sum(r['flat'] for r in calls), calls_on_dispersive=sum(not r['flat'] for r in calls),
                flat_abstained=[dict(test=r['test'], scaffold=r['scaffold'], floppy=r['floppy'], reason=r[key]['reason']) for r in rows if r['flat'] and r[key]['call'] != 'predicted_flat'],
                dispersive_abstention_reasons=dict(Counter(r[key]['reason'] for r in rows if not r['flat'] and r[key]['call'] != 'predicted_flat')))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=str(HERE.parent / 'outputs' / 'classifier'))
    E.add_argument(ap); args = ap.parse_args(); E.apply_argument(args)
    rows = []
    for r in eligible_pairs():
        p = build(r['scaffold'], r['floppy'], r['shift'])
        row = dict(test=r['test'], id=r['id'], scaffold=r['scaffold'], floppy=r['floppy'], flat=r['flat'],
                   frozen_prediction=r['frozen_prediction'], frozen_route=r['frozen_route'])
        row['sector_rule'] = sector_rule(p); row['local_rule'] = local_rule(p)
        called = any(row[k]['call'] == 'predicted_flat' for k in ('sector_rule', 'local_rule'))
        if r['test'] == 'stratified_1000' or called:
            st = spectral_stats(p)
            if r['test'] == 'stratified_1000': row['flat_recomputed'] = bool(st['shape_exact_flat'])
            for k in ('sector_rule', 'local_rule'):
                if row[k]['call'] == 'predicted_flat':
                    a = np.sort(row[k]['values']); b = np.sort(np.asarray(st.get('branch_means', [])))
                    row[k]['max_deviation_from_spectrum'] = float(np.max(np.abs(a - b))) if len(a) == len(b) else None
        rows.append(row)
        print(r['test'], r['scaffold'], r['floppy'], 'flat' if r['flat'] else 'dispersive', row['sector_rule']['call'], row['local_rule']['call'], flush=True)
    out = dict(note='Post hoc rules written after the outcomes were known; a diagnosis, not a prospective result. '
                    '3 random wavevectors per pair (seed 7), tolerance 1e-6; values in units of k_c.',
               sector_rule=summary(rows, 'sector_rule'), local_rule=summary(rows, 'local_rule'),
               stratified_flatness_recomputed_disagreements=[dict(scaffold=r['scaffold'], floppy=r['floppy']) for r in rows if 'flat_recomputed' in r and r['flat_recomputed'] != r['flat']],
               rows=rows)
    C.dump(Path(args.out) / 'sector_rule.json', out)


if __name__ == '__main__':
    main()
