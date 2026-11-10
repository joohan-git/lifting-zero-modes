"""Fixed cage: exact flat bands of nbo, bnn and flu, the sigma tight-binding band of flu, and the
second-order bandwidth coefficients of lvt and uom (SM S2.6; the sigma band in S9).

Five dual pairs (scaffold, floppy net): bcu|nbo, hex|bnn, fcu|flu, bsn|lvt, qtz-x|uom. Both nets are
built in the conventional cell of the RCSR file; the floppy net is placed at the relative origin found
by the screen (classifier/site_symmetry.json, key 'shift'). Graph lengths use the scaffold metric scaled
by its mean edge length. The cage is held fixed: each floppy site i couples with stiffness k_c to its
nearest scaffold sites (all images within 1.05 x the shortest distance), giving the local contact tensor
S_i = sum_c c c^T; the floppy bonds have unit stiffness and give R(q)^dagger R(q).

  * nbo, bnn, flu: the full matrix R^dagger R + k_c S is diagonalized at k_c = 0.1, 1, 3, and the cluster of
    eigenvalues nearest the predicted value (2 k_c for nbo, 12/7 k_c for bnn, the two roots of the 2 x 2
    fluorite problem for flu) is compared with the prediction; 'bandwidth' is the spread over all q.
  * flu, sigma tight binding: the hopping matrix sum_bonds e^{2 pi i q.s} u u^T has flat bands at
    +-4 sqrt(2)/3.
  * lvt, uom: the local normals n_i (lowest eigenvector of the bond tensor T_i) span the kernel at generic q,
    but they are not eigenvectors of S_i. The second-order coefficients beta(q) are the eigenvalues of
    -N^T S R^+ S N (R^+ the pseudo-inverse of R^dagger R); the branch-width coefficient is
    max over branches of (max_q beta - min_q beta) on the wavevectors with spectral gap g(q) >= cutoff.
    Full diagonalization at k_c = 1e-5 ... 1e-2 checks width ~ k_c^2 with this coefficient.

Wavevectors: 1,024 points of a scrambled Sobol sequence (seed 20261107) plus nine special points.
The kernel is the singular-value threshold 1e-9 of R(q).

  python fixed_cage/fixed_cage.py --rcsr DIR [--out DIR]

Outputs (default outputs/fixed_cage/): results.json, bandwidths.csv, scaling.csv, second_order.csv,
exact_bands.csv, sigma_tb.csv, witnesses.json.
"""
import os
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '1'
import argparse
import csv
import itertools
import json
import sys
from pathlib import Path
import numpy as np
from scipy.linalg import block_diag
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import nets as E

CASES = [('bcu', 'nbo'), ('hex', 'bnn'), ('fcu', 'flu'), ('bsn', 'lvt'), ('qtz-x', 'uom')]
SEED = 20261107
SPECIAL = np.vstack([np.zeros(3), np.eye(3) * .5, np.full(3, .5), np.array([[.25, .25, 0], [.1, .1, .2], [1 / 3, 1 / 3, 2 / 3]])])
EXACT_KC = [.1, 1., 3.]
SCALING_KC = np.logspace(-5, -2, 7)
GAP_CUTOFFS = [0., .01, .05, .1, .3]


def conv(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    raise TypeError(type(x).__name__)


def dump(out, name, obj):
    (out / name).write_text(json.dumps(obj, indent=2, default=conv), encoding='utf-8')


def csvout(out, name, rows):
    with (out / name).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def site_records():
    rows = json.loads((ROOT / 'classifier' / 'site_symmetry.json').read_text(encoding='utf8'))['rows']
    return {(r.get('scaffold'), r.get('floppy')): r for r in rows}


def model(source, sc, fl, record):
    scaffold = E.build(source[sc], primitive=False)
    active = E.build(source[fl], primitive=False)
    shift = np.array(record['shift'])
    metric = scaffold['A'] * scaffold['original_mean_length']
    pos = active['pts'] + shift
    edges = active['edges']
    vec = np.array([(pos[j] + sh - pos[i]) @ metric for i, j, sh in edges])
    unit = vec / np.linalg.norm(vec, axis=1)[:, None]
    n = active['n']; T = np.zeros((n, 3, 3)); S = T.copy()
    for (i, j, sh), v in zip(edges, unit):
        T[i] += np.outer(v, v); T[j] += np.outer(v, v)
    images = np.array([p + sh for p in scaffold['pts'] for sh in itertools.product(range(-2, 3), repeat=3)])
    counts = []
    for i, p in enumerate(pos):
        d = (images - p) @ metric; ell = np.linalg.norm(d, axis=1)
        ix = (ell <= 1.05 * ell.min() + 1e-12)
        if ell.min() < 1e-9: raise ValueError('overlapping sites')
        c = d[ix] / ell[ix, None]; S[i] = c.T @ c; counts.append(len(c))
    # agreement of the contact tensors with the per-site eigenvalues stored by the classifier
    residual = 0.
    for i, site in enumerate(record['sites']):
        residual = max(residual, np.max(abs(np.linalg.eigvalsh(S[i]) - site['S_eigenvalues'])))
    return dict(sc=sc, fl=fl, n=n, edges=edges, unit=unit, S=block_diag(*S), T=block_diag(*T), localT=T, localS=S, shift=shift,
                counts=counts, prior_tensor_eigenvalue_residual=residual)


def matrices(m, q):
    r = np.zeros((len(m['edges']), 3 * m['n']), complex)
    hop = np.zeros((3 * m['n'], 3 * m['n']), complex)
    for b, ((i, j, sh), v) in enumerate(zip(m['edges'], m['unit'])):
        z = np.exp(2j * np.pi * np.dot(q, sh)); r[b, 3 * i:3 * i + 3] -= v; r[b, 3 * j:3 * j + 3] += z * v
        outer = z * np.outer(v, v)
        hop[3 * i:3 * i + 3, 3 * j:3 * j + 3] += outer
        hop[3 * j:3 * j + 3, 3 * i:3 * i + 3] += outer.conj().T
    return r, r.conj().T @ r, hop


def spectrum(r):
    _, s, vh = np.linalg.svd(r, full_matrices=True)
    rank = np.count_nonzero(s > 1e-9)
    z = vh[rank:].conj().T
    ev = s[:rank] ** 2
    pos = vh[:rank].conj().T
    pinv = (pos / ev) @ pos.conj().T
    return z, float(ev.min()), pinv


def roots(k):
    """Two fluorite flat-band energies at contact stiffness k (t_A = 4/3, t_B = 8/3, s_A = 4/3, s_B = 2)."""
    a = 4 / 3 + k * 4 / 3; b = 8 / 3 + k * 2
    disc = np.sqrt((a - b) ** 2 + 4 * (4 / 3) * (8 / 3))
    return np.array([(a + b - disc) / 2, (a + b + disc) / 2])


def nearest_cluster(ev, target, count):
    return np.sort(ev[np.argsort(abs(ev - target))[:count]])


def run(out):
    source = E.read_cgd(); records = site_records()
    qs = qmc.Sobol(3, scramble=True, seed=SEED).random_base2(10)
    exact_rows = []; second_rows = []; tb_rows = []; summary = {}; raw = {}; models = {}
    tb_zero_counts = []; tb_reweighted_residual = []; tb_half_norm_residual = []
    for sc, fl in CASES:
        m = model(source, sc, fl, records[(sc, fl)]); models[fl] = m
        allq = np.vstack([qs, SPECIAL]); qdata = []; first = []; dims = []; gaps = []
        n = m['n']; localN = np.zeros((3 * n, n))
        for i, t in enumerate(m['localT']):
            _, v = np.linalg.eigh(t); localN[3 * i:3 * i + 3, i] = v[:, 0]
        normal_res = np.linalg.norm((np.eye(3 * n) - localN @ localN.T) @ m['S'] @ localN, 2)
        for qi, q in enumerate(allq):
            r, d, h = matrices(m, q); z, gap, pinv = spectrum(r)
            dims.append(z.shape[1]); gaps.append(gap)
            p = np.linalg.eigvalsh(z.conj().T @ m['S'] @ z)
            if qi < len(qs): first.append(p)
            if fl in ['nbo', 'bnn', 'flu']:
                vals = ([2.] if fl == 'nbo' else [12 / 7] if fl == 'bnn' else [])
                for k in EXACT_KC:
                    ev = np.linalg.eigvalsh(d + k * m['S'])
                    targets = [k * x for x in vals] if fl != 'flu' else roots(k)
                    for branch, target in enumerate(targets):
                        # the predicted value is fixed; the whole spectrum is searched, the band need not be lowest
                        count = 6 if fl == 'nbo' else 1 if fl == 'bnn' else z.shape[1]
                        sel = nearest_cluster(ev, target, count)
                        exact_rows.append(dict(net=fl, qindex=qi, q1=q[0], q2=q[1], q3=q[2], kc=k, branch=branch, target=target,
                                               cluster_min=min(sel), cluster_max=max(sel), max_error=max(abs(sel - target)),
                                               lowest_index=int(np.argmin(abs(ev - target))), count=count, special=qi >= len(qs)))
                if fl == 'flu':
                    target = 4 * np.sqrt(2) / 3
                    ev, hv = np.linalg.eigh(h)
                    tb_zero_counts.append(int(np.count_nonzero(abs(ev) < 1e-9)))
                    classA = np.diag(m['T']) < 2
                    scale = np.sqrt(np.diag(m['T']))
                    rz = scale[:, None] * z
                    tb_reweighted_residual.append(np.linalg.norm(h @ rz - target * rz))
                    ix = np.argsort(abs(ev - target))[:z.shape[1]]; hz = hv[:, ix]
                    tb_half_norm_residual.append(np.linalg.norm(hz[classA].conj().T @ hz[classA] - .5 * np.eye(z.shape[1]), 2))
                    for sign in [-1, 1]:
                        sel = nearest_cluster(ev, sign * target, z.shape[1])
                        tb_rows.append(dict(qindex=qi, q1=q[0], q2=q[1], q3=q[2], sign=sign, target=sign * target, cluster_min=min(sel),
                                            cluster_max=max(sel), max_error=max(abs(sel - sign * target)), count=z.shape[1]))
            else:
                # the local normal basis is exact and q-independent; generic kernel dimension only
                if z.shape[1] != n: continue
                s = float(np.trace(localN.T @ m['S'] @ localN) / n)
                b = -localN.T @ m['S'] @ pinv @ m['S'] @ localN
                beta = np.linalg.eigvalsh(b)
                qdata.append(dict(qindex=qi, q=q, gap=gap, beta=beta, first=s, kernel_residual=np.linalg.norm(r @ localN)))
                for bi, value in enumerate(beta):
                    second_rows.append(dict(net=fl, qindex=qi, q1=q[0], q2=q[1], q3=q[2], gap=gap, branch=bi, beta=value, first=s))
        summ = dict(vertices=n, edges=len(m['edges']), shift=m['shift'], contact_counts=m['counts'],
                    prior_tensor_eigenvalue_residual=m['prior_tensor_eigenvalue_residual'],
                    kernel_dim_histogram={str(i): dims.count(i) for i in set(dims)}, gap_quantiles=np.quantile(gaps[:len(qs)], [0, .1, .5, .9, 1]),
                    normal_S_mixing_norm=normal_res, first_generic_min=min(x.min() for x in first), first_generic_max=max(x.max() for x in first))
        if qdata:
            raw[fl] = qdata
            summ['coefficient_domains'] = {}
            for cutoff in GAP_CUTOFFS:
                dat = [x for x in qdata if x['qindex'] < len(qs) and x['gap'] >= cutoff]
                if not dat: continue
                b = np.array([x['beta'] for x in dat])
                summ['coefficient_domains'][str(cutoff)] = dict(nq=len(dat), beta_min=b.min(), beta_max=b.max(), branch_width_coefficients=np.ptp(b, axis=0),
                                                                 maximum_branch_width_coefficient=max(np.ptp(b, axis=0)), multiplet_envelope_coefficient=b.max() - b.min())
        summary[fl] = summ
        print(fl, 'done', flush=True)

    # full diagonalization on the same wavevectors (gap >= 0.1) against the second-order coefficients
    scaling = []; witnesses = {}
    for fl, qdata in raw.items():
        m = models[fl]; n = m['n']
        dat = [x for x in qdata if x['qindex'] < len(qs) and x['gap'] >= .1]
        if not dat: raise ValueError('empty fixed-gap sample')
        coeff = np.array([x['beta'] for x in dat]); s = dat[0]['first']
        ib = np.unravel_index(np.argmin(coeff), coeff.shape)[0]
        jb = np.unravel_index(np.argmax(coeff), coeff.shape)[0]
        witnesses[fl] = [dat[ib], dat[jb]]
        for k in SCALING_KC:
            energy = []; err = []
            for x in dat:
                _, d, _ = matrices(m, x['q']); ev = np.linalg.eigvalsh(d + k * m['S'])[:n]
                energy.append(ev); err.append(max(abs(ev - (k * s + k * k * x['beta']))))
            energy = np.array(energy); width = np.ptp(energy, axis=0)
            scaling.append(dict(net=fl, kc=k, nq=len(dat), gap_cutoff=.1, maximum_branch_width=max(width), width_over_kc_squared=max(width) / k ** 2,
                                multiplet_envelope=energy.max() - energy.min(), max_second_order_energy_error=max(err), coefficient=max(np.ptp(coeff, axis=0))))
        arr = [x for x in scaling if x['net'] == fl]
        fit = arr[2:5]          # k_c = 1e-4 ... 1e-3: roundoff and higher orders both small
        summary[fl]['full_bandwidth_scaling_exponent'] = np.polyfit(np.log([x['kc'] for x in fit]), np.log([x['maximum_branch_width'] for x in fit]), 1)[0]
        summary[fl]['full_second_order_validation'] = arr
        # approach to the soft set: lvt along q = (t, t, 2t) (a rank-loss line), uom towards Gamma along a generic ray
        soft = []
        for delta in [1e-1, 3e-2, 1e-2, 3e-3, 1e-3]:
            q = np.array([.137, .137 + delta, .274]) if fl == 'lvt' else delta * np.array([.37, .61, .89])
            r, d, _ = matrices(m, q); z, gap, pinv = spectrum(r)
            if z.shape[1] != n: continue
            beta = np.linalg.eigvalsh(-z.conj().T @ m['S'] @ pinv @ m['S'] @ z)
            soft.append(dict(delta=delta, q=q, gap=gap, beta_min=beta.min(), beta_max=beta.max()))
        summary[fl]['soft_approach'] = soft

    agg = []
    for fl in ['nbo', 'bnn', 'flu']:
        for k in EXACT_KC:
            for branch in ([0, 1] if fl == 'flu' else [0]):
                rr = [x for x in exact_rows if x['net'] == fl and x['kc'] == k and x['branch'] == branch]
                agg.append(dict(net=fl, kc=k, branch=branch, energy=rr[0]['target'], sample_count=len(rr),
                                bandwidth=max(x['cluster_max'] for x in rr) - min(x['cluster_min'] for x in rr), max_target_error=max(x['max_error'] for x in rr),
                                lowest_index_min=min(x['lowest_index'] for x in rr), lowest_index_max=max(x['lowest_index'] for x in rr)))
    summary['exact_bandwidths'] = agg
    pos_rows = [x for x in tb_rows if x['sign'] == 1]; neg_rows = [x for x in tb_rows if x['sign'] == -1]
    summary['sigma_TB'] = dict(expected=4 * np.sqrt(2) / 3,
                               positive_bandwidth=max(x['cluster_max'] for x in pos_rows) - min(x['cluster_min'] for x in pos_rows),
                               negative_bandwidth=max(x['cluster_max'] for x in neg_rows) - min(x['cluster_min'] for x in neg_rows),
                               max_error=max(x['max_error'] for x in tb_rows), zero_band_count_histogram={str(i): tb_zero_counts.count(i) for i in set(tb_zero_counts)},
                               generic_zero_counts=sorted(set(tb_zero_counts[:len(qs)])), reweighted_kernel_residual=max(tb_reweighted_residual),
                               TB_equal_norm_share_residual=max(tb_half_norm_residual))
    # Counterexamples to broader wording (not to the hypotheses). A one-dimensional bipartite chain has
    # t_A = t_B = 2 but no mechanism at generic q: its sigma bands are +-|1 + exp(2 pi i q)|, not flat +-2.
    summary['counterexamples'] = dict(chain_q=[0, .25, .5], chain_positive_energy=[2, np.sqrt(2), 0],
                                      mixed_local_q_independent_model='D0=diag(0,1), S=[[1,0.3],[0.3,2]] at every cell: Sn not parallel n, yet both exact bands are flat')
    # Three-orbital simple cubic lattice in a 2x2x2 cell: bipartite and T_A = T_B = 2I, but no all-q mechanisms.
    positions = list(itertools.product([0, 1], repeat=3)); hop = np.zeros((24, 24), complex)
    q = np.array([.17, .31, .43])
    for i, p in enumerate(positions):
        for axis in range(3):
            nxt = list(p); nxt[axis] = (nxt[axis] + 1) % 2; j = positions.index(tuple(nxt))
            phase = np.exp(2j * np.pi * q[axis]) if p[axis] == 1 else 1.
            hop[3 * i + axis, 3 * j + axis] += phase; hop[3 * j + axis, 3 * i + axis] += phase.conjugate()
    ev = np.linalg.eigvalsh(hop)
    summary['counterexamples']['pcu_three_orbital'] = dict(q=q, t_A=2, t_B=2, TB_spectrum=ev, gap_of_RdaggerR=min(2 - ev),
                                                          distance_to_claimed_plus2=min(abs(ev - 2)), distance_to_claimed_minus2=min(abs(ev + 2)))
    assert max(x['max_error'] for x in exact_rows) < 1e-10
    assert summary['sigma_TB']['max_error'] < 1e-10
    for fl in ['lvt', 'uom']:
        assert summary[fl]['coefficient_domains']['0.1']['maximum_branch_width_coefficient'] > 1e-3
    out.mkdir(parents=True, exist_ok=True)
    csvout(out, 'exact_bands.csv', exact_rows); csvout(out, 'bandwidths.csv', agg)
    csvout(out, 'second_order.csv', second_rows); csvout(out, 'scaling.csv', scaling)
    csvout(out, 'sigma_tb.csv', tb_rows); dump(out, 'witnesses.json', witnesses); dump(out, 'results.json', summary)
    for fl in ['lvt', 'uom']:
        print(fl, 'maximum branch-width coefficient (gap >= 0.1):', round(summary[fl]['coefficient_domains']['0.1']['maximum_branch_width_coefficient'], 4))
    print('largest exact bandwidth:', max(x['bandwidth'] for x in agg), '| sigma TB bandwidths:',
          summary['sigma_TB']['positive_bandwidth'], summary['sigma_TB']['negative_bandwidth'])
    print('written to', out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    E.add_argument(ap)
    ap.add_argument('--out', default=str(ROOT / 'outputs' / 'fixed_cage'))
    args = ap.parse_args(); E.apply_argument(args)
    run(Path(args.out))


if __name__ == '__main__':
    main()
