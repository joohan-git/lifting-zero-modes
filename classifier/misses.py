"""Post hoc diagnosis of the flat pairs that the frozen classifier missed in the two prospective tests (SM S3.7).

Does not change the classifier, and none of it counts as a prospective success. The six missed pairs are the
eligible, exactly flat pairs of both tests on which the frozen classifier abstained. They are diagnosed together
with the route-L pairs of the tests and the two collective atlas pairs fcu/flu and acs/gra. For every flat
first-order value (common primitive cell, 10 random wavevectors):
  support    class weights f_c, per-site weights, off-site blocks of the band projector (single-site or collective);
  sectors    span of the local mode components at each vertex, with the bond tensor T, the inter-class bond
             tensor T' (bonds to other classes only) and the contact tensor S on that sector;
  balance    inter-class balance: shares f_c proportional to 1/t'_c and value sum_c f_c s_c; the same with
             the full bond tensors t_c for comparison;
  rigid      rigid-body and translation-only fits of the modes on connected pieces of the support and on
             the intra-class clusters, with the summed S and T' of each piece;
  counting   for values in the spectrum of S: n_a, E_eff, rank of R_F on E_a and dim(E_a cap ker R_F);
  cage       the band of D_F + k_c S with the scaffold fixed, k_c/k = 0.1, 1, 3 (mean, width over q, overlap);
  mechanism  single_site, eigenvalue (value in the spectrum of S), inter_class_balance (shares and value as
             predicted from t'), fixed_local_line (every vertex moves along one q-independent line) or other.

  python classifier/misses.py --rcsr DIR [--out DIR]
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import scipy.linalg as la
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from common import nets as E
from common import composite as C
from sector_tools import build, classes, kernel, sector_of, exact_fraction, load, eligible_pairs

NQ = 10
SEED = 2026
KC = (0.1, 1.0, 3.0)
# fixed order: one random stream (SEED) is consumed pair by pair in this order
DIAGNOSED = [('ttr', 'omy'), ('sta', 'smv'), ('wfm', 'wih'), ('pdb', 'xav'), ('ith-e', 'urg'), ('tca', 'hyb'),
             ('kpl', 'uro'), ('ckq', 'nfa'), ('svd', 'knc'), ('wfm', 'hua'), ('mob', 'scp'), ('xfs', 'nfc'),
             ('fcu', 'flu'), ('acs', 'gra')]


def cluster(g, support):
    """Connected pieces of a vertex set over floppy edges, unwrapped to lattice images; True if a piece closes
    a loop with a net lattice shift (no finite cluster)."""
    adj = {i: [] for i in support}
    for a, b, s in g['edges']:
        if a in adj and b in adj: adj[a].append((b, np.array(s))); adj[b].append((a, -np.array(s)))
    pos = {}; infinite = False; comps = []
    for root in support:
        if root in pos: continue
        pos[root] = np.zeros(3, int); stack = [root]; comp = [root]
        while stack:
            i = stack.pop()
            for j, s in adj[i]:
                if j not in pos: pos[j] = pos[i] + s; stack.append(j); comp.append(j)
                elif np.any(pos[i] + s - pos[j]): infinite = True
        comps.append(sorted(comp))
    return pos, infinite, comps


def rigid_fit(g, comp, pos, data_modes, qs):
    """Largest relative residual of a rigid-body (translation + rotation) and of a translation-only fit of the
    band's modes restricted to the piece comp, over modes and wavevectors (Bloch phases on the images)."""
    pts = np.asarray(g['pts']); A = g['A']
    r = np.array([(pts[i] + pos[i]) @ A for i in comp]); r0 = r.mean(0)
    Am = np.vstack([np.hstack([np.eye(3), np.array([[0, x[2], -x[1]], [-x[2], 0, x[0]], [x[1], -x[0], 0]])]) for x in r - r0])
    At = np.vstack([np.eye(3)] * len(comp)); rig = 0.; tra = 0.
    for q, X in zip(qs, data_modes):
        for k in range(X.shape[1]):
            u = np.array([X[3 * i:3 * i + 3, k] * np.exp(2j * np.pi * np.dot(q, pos[i])) for i in comp]).ravel()
            nu = la.norm(u)
            if nu < 1e-12: continue
            x = la.lstsq(Am, u)[0]; rig = max(rig, la.norm(Am @ x - u) / nu)
            x = la.lstsq(At, u)[0]; tra = max(tra, la.norm(At @ x - u) / nu)
    d = la.norm(r[:, None] - r[None], axis=2)[np.triu_indices(len(comp), 1)]
    return rig, tra, sorted(set(np.round(d, 5).tolist()))


def counting(p, qs, value):
    """Eigenspace-resolved counting at the eigenvalues a of S within 1e-4 of value."""
    S = p['S']; w, V = la.eigh(S); E_ = p['fl']['e']; pf = C.prep(p['fl'])
    groups = []
    for i, a in enumerate(w):
        if groups and abs(a - groups[-1][0]) < 1e-6: groups[-1][1].append(i)
        else: groups.append([a, [i]])
    out = []
    for a, idx in groups:
        if abs(a - value) > 1e-4: continue
        Pa = V[:, idx]; per_q = set()
        for q in qs:
            RP = C.compat(pf, q) @ Pa; rk = int(np.sum(la.svdvals(RP) > 1e-8)); eeff = int(np.sum(la.norm(RP, axis=1) > 1e-10))
            per_q.add((rk, eeff))
        (rk, eeff), = per_q if len(per_q) == 1 else (min(per_q),)
        out.append(dict(eigenvalue=float(a), n_a=len(idx), E_eff=eeff, generic_count=max(0, len(idx) - eeff), naive_count=max(0, len(idx) - E_),
                        rank_R_F_on_E_a=rk, dim_E_a_cap_kernel=len(idx) - rk, same_at_all_q=len(per_q) == 1))
    return out


def diagnose(sc, fl, shift, rng):
    p = build(sc, fl, shift); g = p['classifier_graph']; n = g['n']; S = p['S']
    cls, bonds, T, Sloc, normals = classes(p); ncls = int(cls.max()) + 1
    qs = rng.random((NQ, 3)); pf = C.prep(p['fl'])
    data = []
    for q in qs:
        Z = kernel(p, q); w, U = la.eigh(Z.conj().T @ S @ Z); data.append((q, Z, w, Z @ U))
    m = data[0][1].shape[1]
    vals = sorted(set(round(float(x), 4) for _, _, w, _ in data for x in w)); groups = []
    for v in vals:
        if groups and abs(v - groups[-1]) < 2e-4: continue
        groups.append(v)
    Tinter = np.zeros_like(T)
    for k, (a, b, s) in enumerate(g['edges']):
        if cls[a] != cls[b]:
            vv = np.outer(g['unit'][k], g['unit'][k]); Tinter[a] += vv; Tinter[b] += vv
    spec_S = la.eigvalsh(S)
    row = dict(scaffold=sc, floppy=fl, shift=np.asarray(p['shift']).tolist(), V_F=n, E_F=p['fl']['e'], M_F=m, classes=[], bands=[])
    for c in range(ncls):
        ids = np.where(cls == c)[0]; i = ids[0]
        row['classes'].append(dict(cls=c, vertices=ids.tolist(), z=len(bonds[i]), contacts=len(p['stars'][i]),
                                   T_eigenvalues=la.eigvalsh(T[i]).tolist(), S_eigenvalues=la.eigvalsh(Sloc[i]).tolist(), planar=bool(i in normals),
                                   intra_class_edges=sum(1 for a, b, s in g['edges'] if cls[a] == c and cls[b] == c)))
    for v in groups:
        modes_q = [M[:, np.where(abs(w - v) < 2e-4)[0]] for q, Z, w, M in data]
        mv = {x.shape[1] for x in modes_q}
        if len(mv) != 1:
            row['bands'].append(dict(value=v, multiplicity_varies=sorted(mv))); continue
        mv = mv.pop()
        value = float(np.mean([np.diag(X.conj().T @ S @ X).real.mean() for X in modes_q]))
        weights = []; offd = []; offsite = []; PR = []; site_w = []; site_q = []
        for X in modes_q:
            Pi = X @ X.conj().T; W = []
            for c in range(ncls):
                idx = np.concatenate([np.arange(3 * i, 3 * i + 3) for i in np.where(cls == c)[0]])
                A = X[idx].conj().T @ X[idx]; f = np.trace(A).real / mv; W.append(f); offd.append(la.norm(A - f * np.eye(mv)))
            weights.append(W)
            offsite.append(max((la.norm(Pi[3 * i:3 * i + 3, 3 * j:3 * j + 3]) for i in range(n) for j in range(n) if i != j), default=0.))
            wv = (np.abs(X) ** 2).reshape(n, 3, mv).sum(1); PR.append(float(np.mean(1. / (wv ** 2).sum(0))))
            blocks = [Pi[3 * i:3 * i + 3, 3 * i:3 * i + 3] for i in range(n)]
            site_w.append([np.trace(b).real / mv for b in blocks])
            site_q.append([np.trace(Sloc[i] @ b).real / max(np.trace(b).real, 1e-300) for i, b in enumerate(blocks)])
        weights = np.array(weights); wmean = weights.mean(0); site_w = np.array(site_w); site_q = np.array(site_q)
        support = [i for i in range(n) if site_w[:, i].mean() > 1e-9]
        collective = max(offsite) >= 1e-7
        band = dict(value=value, exact_fraction=exact_fraction(value), multiplicity=mv, in_spectrum_of_S=bool(np.min(np.abs(spec_S - value)) < 1e-4),
                    class_weights=wmean.tolist(), class_weight_q_variation=float(np.ptp(weights, axis=0).max()),
                    class_gram_residual=float(max(offd)), offsite_block_norm=float(max(offsite)), collective=bool(collective),
                    participation_sites_per_mode=float(np.mean(PR)), support_sites=support,
                    site_weights=site_w.mean(0)[support].tolist(), site_weight_q_variation=float(np.ptp(site_w[:, support], axis=0).max()),
                    equal_site_weights=bool(np.ptp(site_w.mean(0)[support]) < 1e-6),
                    local_quotients=site_q.mean(0)[support].tolist(), local_quotient_q_variation=float(np.ptp(site_q[:, support], axis=0).max()),
                    sectors=[])
        involved = [c for c in range(ncls) if wmean[c] > 1e-6]; tsec = {}; tfull = {}; ssec = {}
        for c in involved:
            dims = set(); res = dict(T=0., T_inter=0., S=0.); vals_ = dict(T=[], T_inter=[], S=[]); intra_in = 0; dirs = []
            for i in np.where(cls == c)[0]:
                locs = np.concatenate([X[3 * i:3 * i + 3].T for X in modes_q])
                Vb = sector_of(np.concatenate([locs.real, locs.imag])); d = Vb.shape[1]; dims.add(d)
                if d == 0: continue
                if d == 1: dirs.append(Vb[:, 0].tolist())
                for key, M_ in (('T', T[i]), ('T_inter', Tinter[i]), ('S', Sloc[i])):
                    val = np.trace(Vb.T @ M_ @ Vb) / d; vals_[key].append(val); res[key] = max(res[key], la.norm(M_ @ Vb - val * Vb))
                for k, (a, b, s) in enumerate(g['edges']):
                    if a == i and cls[b] == c and la.norm(Vb.T @ g['unit'][k]) > 1e-8: intra_in += 1
            tsec[c] = float(np.mean(vals_['T_inter'])); tfull[c] = float(np.mean(vals_['T'])); ssec[c] = float(np.mean(vals_['S']))
            sec = dict(cls=c, sector_dimensions=sorted(dims), t=tfull[c], t_residual=float(res['T']), t_inter=tsec[c], t_inter_residual=float(res['T_inter']),
                       s=ssec[c], s_residual=float(res['S']), s_spread=float(np.ptp(vals_['S'])), intra_class_bonds_in_sector=intra_in)
            if dirs and max(dims) == 1: sec['line_directions'] = dirs
            band['sectors'].append(sec)
        if collective and len(involved) > 1:
            for key, tt in (('balance_inter_class', tsec), ('balance_full_bond_tensors', tfull)):
                inv = np.array([1. / tt[c] if tt[c] > 1e-9 else np.inf for c in involved])
                if np.all(np.isfinite(inv)):
                    f = inv / inv.sum(); pred = float(sum(x * ssec[c] for x, c in zip(f, involved)))
                    band[key] = dict(classes=involved, shares=f.tolist(), shares_fraction=[exact_fraction(x, 50) for x in f],
                                     measured_shares=[float(wmean[c]) for c in involved], predicted_value=pred, matches_value=bool(abs(pred - value) < 1e-4),
                                     matches_shares=bool(np.max(np.abs(f - wmean[involved])) < 1e-6))
                else:
                    band[key] = dict(classes=involved, applicable=False, note='a sector tensor vanishes (single-site normal)')
        # rigid-unit tests (collective bands): connected pieces of the support, and intra-class clusters inside it
        pos, infinite, comps = cluster(g, support) if collective else ({}, False, []); pieces = []
        for comp in comps:
            rig, tra, dist = rigid_fit(g, comp, pos, modes_q, qs)
            pieces.append(dict(kind='support', sites=comp, rigid_residual=rig, translation_residual=tra,
                               summed_S_eigenvalues=la.eigvalsh(sum(Sloc[i] for i in comp)).tolist(), pair_distances=dist[:6]))
        for c in (involved if collective else []):
            ids = [i for i in np.where(cls == c)[0] if i in support]
            if len(ids) < 2 or ids == support: continue
            cpos, cinf, ccomps = cluster(g, ids)
            for comp in ccomps:
                if len(comp) < 2 or cinf: continue
                rig, tra, dist = rigid_fit(g, comp, cpos, modes_q, qs)
                pieces.append(dict(kind='class_cluster', cls=c, sites=comp, rigid_residual=rig, translation_residual=tra,
                                   summed_S_eigenvalues=la.eigvalsh(sum(Sloc[i] for i in comp)).tolist(),
                                   summed_T_inter_eigenvalues=la.eigvalsh(sum(Tinter[i] for i in comp)).tolist(), pair_distances=dist[:6]))
        band['support_loop_with_lattice_shift'] = bool(infinite); band['rigid_units'] = pieces
        if band['in_spectrum_of_S']: band['counting'] = counting(p, qs[:3], value)
        band['_modes'] = modes_q
        row['bands'].append(band)
    # fixed cage: D_F + k_c S with the scaffold frozen; branches assigned by overlap with the first-order band
    for band in row['bands']:
        if '_modes' not in band: continue
        cage = []
        for kc in KC:
            vq = []; ov_ = []
            for (q, Z, w, M), X in zip(data, band['_modes']):
                R = C.compat(pf, q); e, Vv = la.eigh(R.conj().T @ R + kc * S)
                ov = la.norm(X.conj().T @ Vv, axis=0) ** 2; sel = np.argsort(-ov)[:band['multiplicity']]
                vq.append(np.sort(e[sel])); ov_.append(ov[sel].mean())
            vq = np.array(vq)
            cage.append(dict(kc_over_k=kc, mean=float(vq.mean()), width=float(np.ptp(vq, axis=0).max()), overlap=float(np.mean(ov_))))
        band['fixed_cage'] = cage; del band['_modes']
        band['exact_in_fixed_cage'] = bool(max(c['width'] for c in cage) < 1e-12)
        bal = band.get('balance_inter_class', {})
        band['mechanism'] = ('single_site' if not band['collective'] else 'eigenvalue' if band['in_spectrum_of_S']
                             else 'inter_class_balance' if bal.get('matches_value') and bal.get('matches_shares')
                             else 'fixed_local_line' if all(s['sector_dimensions'] == [1] for s in band['sectors']) else 'other')
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=str(HERE.parent / 'outputs' / 'classifier'))
    E.add_argument(ap); args = ap.parse_args(); E.apply_argument(args)
    elig = eligible_pairs()
    missed = [(r['test'], r['scaffold'], r['floppy']) for r in elig if r['flat'] and r['frozen_prediction'] != 'predicted_flat']
    designated = [(r['test'], r['scaffold'], r['floppy']) for r in elig if r['flat'] and r['frozen_prediction'] == 'predicted_flat']
    shifts = {(r['scaffold'], r['floppy']): r['shift'] for r in elig}
    tests = {(r['scaffold'], r['floppy']): r['test'] for r in elig}
    for r in load(HERE.parent / 'screen' / 'dual_screen.json')['rows']:
        if r.get('scaffold'): shifts.setdefault((r['scaffold'], r['floppy']), r['shift'])
    rng = np.random.default_rng(SEED); rows = []
    for sc, fl in DIAGNOSED:
        row = diagnose(sc, fl, shifts[(sc, fl)], rng)
        row['cohort'] = tests.get((sc, fl), 'dual_atlas')
        row['role'] = ('missed' if (row['cohort'], sc, fl) in missed else 'designated_route_L' if (row['cohort'], sc, fl) in designated else 'atlas_reference')
        rows.append(row)
        print(sc, fl, row['role'], [(round(b['value'], 4), b.get('multiplicity')) for b in row['bands']], flush=True)
    out = dict(note='Post hoc diagnosis of outcomes already known; not a prospective result. Values in units of k_c; '
                    'common primitive cell; 10 random wavevectors (seed 2026), fixed cage at k_c/k = 0.1, 1, 3.',
               eligible_pairs=len(elig), eligible_flat=len(missed) + len(designated),
               missed_flat_pairs=[dict(test=t, scaffold=a, floppy=b) for t, a, b in missed],
               designated_flat_pairs=[dict(test=t, scaffold=a, floppy=b) for t, a, b in designated], pairs=rows)
    C.dump(Path(args.out) / 'misses.json', out)


if __name__ == '__main__':
    main()
