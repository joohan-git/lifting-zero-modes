"""Flatness at finite stiffness ratio for the ten exactly flat pairs: the eight whose band splits on a measure-zero
set and fcu (x) flu and she-d (x) she, which split on none (main text; SM Section S3.5, Table S5); and the drive on
the eight splitting pairs (SM Section S8.2). The two non-splitting pairs were run later with this script, unchanged,
on the same wavevectors; zone_fraction_summary.csv lists them last.

static: full composite at k_c/k_s = 1e-3, 1e-2, 3e-2 on 8,192 wavevectors (eight scrambled Sobol
blocks of 1,024). Full eigenstates are assigned to the first-order reference bands by maximal overlap
with the mechanism projectors; a wavevector counts when any assigned branch deviates by more than
1% (5%) from its first-order value.
gain: largest Floquet exponent at Omega = 2 omega_ref for each flat reference value, on 256 Sobol
wavevectors plus split points from the scans.

  python finite_stiffness/zone_fractions.py static --rcsr DIR --pair bcu:nbo [--nq 64]
  python finite_stiffness/zone_fractions.py gain --rcsr DIR --pair bcu:nbo [--nq 8]
  python finite_stiffness/zone_fractions.py summarize --raw DIR     # tables from raw per-pair files
"""
import argparse
import itertools
import json
import sys
from pathlib import Path
import numpy as np
import scipy.linalg as la
from scipy.optimize import linear_sum_assignment
from scipy.stats import qmc, t as student_t
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'classifier')); sys.path.insert(0, str(ROOT / 'drive'))
from common import nets as E
from common import composite as C
from common.records import find, pair_id
from flat_classifier import kernel, local_data
from gain import monodromy

PAIRS = [('acs', 'gra'), ('bcu', 'nbo'), ('hex', 'bnn'), ('bsn', 'lvt'), ('hxg-d', 'hxg'), ('nia', 'nia-d'), ('pcu', 'urk'), ('qtz-x', 'uom')]
STATIC_PAIRS = PAIRS + [('fcu', 'flu'), ('she-d', 'she')]     # static mode only; the drive was run on the eight
KCS = [.001, .01, .03]
THRESHOLDS = [.01, .05]


def setup(sc, fl):
    r = find(sc, fl); p = C.pair(sc, fl, r['shift'])
    _, B = E.sg_ops(C.CG()[sc], p['sc']['setting']); p = C.primitive(p, B)
    n = 3 * p['fl']['n']; S = np.zeros((n, n)); stars = [[] for _ in range(p['fl']['n'])]
    for i, j, s, v, l in p['contacts']:
        S[3 * i:3 * i + 3, 3 * i:3 * i + 3] += np.outer(v, v); stars[i].append(v)
    p['S'] = S; p['stars'] = stars
    z = kernel(C.compat(C.prep(p['fl']), [.1731, .2717, .3923])); w = la.eigvalsh(z.conj().T @ S @ z)
    ref = np.round(w, 10); refs = np.unique(ref); counts = [int(sum(ref == c)) for c in refs]
    return p, B, refs, counts


def target(p, q, refs, counts):
    """Generic mechanism bundle at q (exact local normals when they exhaust it), rotated to the
    eigenbasis of the first-order operator, padded with zeros on the scaffold."""
    bonds, T, S, normals = local_data(p['fl'], p['stars']); m = sum(counts); n = 3 * p['fl']['n']
    if len(normals) == m:
        z = np.zeros((n, m), complex)
        for j, (i, v) in enumerate(normals.items()): z[3 * i:3 * i + 3, j] = v
    else:
        z = kernel(C.compat(C.prep(p['fl']), q))
        if z.shape[1] != m: z = kernel(C.compat(C.prep(p['fl']), np.array(q) + 1e-5 * np.array([.731, .419, .293])))
    if z.shape[1] != m: raise ValueError('generic mechanism dimension not recovered')
    w, V = la.eigh(z.conj().T @ p['S'] @ z); Q = z @ V
    if max(abs(w - np.repeat(refs, counts))) > 1e-6: raise ValueError('first-order reference mismatch')
    return np.pad(Q, ((0, 3 * p['sc']['n']), (0, 0)))


def measure(D, Q, kc, refs, counts):
    lam, U = la.eigh(D, check_finite=False); overlap = abs(Q.conj().T @ U) ** 2
    weights = []; offset = 0
    for count in counts:
        weights.append(overlap[offset:offset + count].sum(0)); offset += count
    cost = np.repeat(np.array(weights), counts, axis=0)
    rows, cols = linear_sum_assignment(-cost)
    chosen = lam[cols] / kc; ref = np.repeat(refs, counts); deviation = abs(chosen - ref) / ref
    leakage = [sum(float(w[abs(lam / kc - c) / c > delta].sum()) for c, w in zip(refs, weights)) / sum(counts) for delta in THRESHOLDS]
    return dict(max_deviation=float(max(deviation)), mean_deviation=float(np.mean(deviation)),
                selected_squared_over_kc=chosen.tolist(), spectral_weight_outside=leakage,
                selected_projector_weight=float(cost[rows, cols].sum() / sum(counts)))


def gamma_distance(p, qs):
    reciprocal = la.inv(p['fl']['A']).T; unit = abs(la.det(reciprocal)) ** (1 / 3)
    sh = np.array(list(itertools.product(range(-1, 2), repeat=3)))
    return np.min(la.norm((qs[:, None, :] + sh[None, :, :]) @ reciprocal, axis=2), axis=1) / unit


def static(sc, fl, nq=None):
    p, B, refs, counts = setup(sc, fl)
    qs = np.vstack([qmc.Sobol(3, scramble=True, seed=20261201 + j).random_base2(10) for j in range(8)])
    if nq: qs = qs[:nq]
    data = dict(scaffold=sc, floppy=fl, q=qs.tolist(), nq=len(qs), reference_squared_over_kc=refs.tolist(), reference_multiplicities=counts,
                kc_over_ks=KCS, threshold_relative=THRESHOLDS, gamma_distance=gamma_distance(p, qs).tolist(), rows=[], kF=1, kS=1, mass=1, stress=0)
    pf, ps = C.prep(p['fl']), C.prep(p['sc'])
    for q in qs:
        R = la.block_diag(C.compat(pf, q), C.compat(ps, q)); base = R.conj().T @ R; Cm = C.contact_matrix(p, q); pump = Cm.conj().T @ Cm
        Q = target(p, q, refs, counts); data['rows'].append([measure(base + kc * pump, Q, kc, refs, counts) for kc in KCS])
    data['complete'] = len(data['rows']) == 8192
    return data


def gain(sc, fl, nq=None):
    p, B, refs, counts = setup(sc, fl)
    qs = np.vstack([qmc.Sobol(3, scramble=True, seed=20261231 + j).random_base2(5) for j in range(8)])
    samples = [dict(q=q.tolist(), kind='BZ_uniform', replicate=i // 32) for i, q in enumerate(qs)]
    scan = HERE.parent / 'high_symmetry' / 'scans' / f'{pair_id(sc, fl)}.json'
    special = [dict(q=[0., 0., 0.], kind='Gamma', q_conventional=[0., 0., 0.])]
    if scan.exists():
        rows = json.loads(scan.read_text(encoding='utf8'))['rows']
        cand = [r for r in rows if r.get('generic_flat_multiplicity_missing', 0) > 0 and la.norm(np.array(r['q']) % 1) > 1e-8]
        if len(cand) > 64: cand = [cand[i] for i in np.random.default_rng(SPECIAL_SEED[(sc, fl)]).choice(len(cand), 64, replace=False)]
        special += [dict(q=(np.array(r['q']) @ B.T % 1).tolist(), kind='nonGamma_split_stratum', q_conventional=r['q'], strata=r.get('strata')) for r in cand]
    samples = (samples[:nq] if nq else samples) + (special[:1] if nq else special)
    data = dict(scaffold=sc, floppy=fl, samples=samples, rows=[], kc_over_ks=KCS, reference_squared_over_kc=refs.tolist(), h=.1, damping=0, steps=256,
                drive='Omega=2*sqrt(kc*reference); each distinct flat reference driven separately', step_checks=[])
    for iq, s in enumerate(samples):
        gains = []
        for kc in KCS:
            D, P = C.full_matrices(p, s['q'], kc); lam, U = la.eigh(D); Pe = U.conj().T @ P @ U; gg = []
            for c in refs:
                wr = np.sqrt(kc * c); g = monodromy(lam, Pe, 2 * wr, .1, 256) / wr; gg.append(g)
                if iq < 3 or s['kind'] == 'Gamma':
                    g2 = monodromy(lam, Pe, 2 * wr, .1, 512) / wr
                    data['step_checks'].append(dict(q_index=iq, kc=kc, reference=float(c), gain=g, gain_double_steps=g2, error=abs(g - g2)))
            gains.append(gg)
        data['rows'].append(gains)
    data['complete'] = nq is None
    return data


# seeds of the split-point subsample (one per pair, fixed when the scans were subsampled)
SPECIAL_SEED = {('acs', 'gra'): 20261215, ('bcu', 'nbo'): 20261220, ('hex', 'bnn'): 20261225, ('bsn', 'lvt'): 20261230,
                ('hxg-d', 'hxg'): 20261257, ('nia', 'nia-d'): 20261268, ('pcu', 'urk'): 20261273, ('qtz-x', 'uom'): 20261276}


def replicate_stats(values):
    a = np.array(values); means = a.reshape(8, -1).mean(1); mean = float(means.mean()); se = float(means.std(ddof=1) / np.sqrt(8))
    half = float(student_t.ppf(.975, 7) * se)
    return dict(mean=mean, replicate_means=means.tolist(), standard_error=se, ci95=[max(0., mean - half), min(1., mean + half)])


def summarize(raw, out):
    rows = []
    for path in sorted((Path(raw) / 'static').glob('*.json')):
        d = json.loads(path.read_text(encoding='utf8'))
        if not d['complete']: continue
        dev = np.array([[x['max_deviation'] for x in r] for r in d['rows']]); rho = np.array(d['gamma_distance'])
        weights = np.array([[x['spectral_weight_outside'] for x in r] for r in d['rows']])
        selection = np.array([[x['selected_projector_weight'] for x in r] for r in d['rows']])
        for ik, kc in enumerate(d['kc_over_ks']):
            for it, delta in enumerate(d['threshold_relative']):
                hit = dev[:, ik] > delta; inside = rho <= 2 * np.sqrt(kc)
                st = replicate_stats(hit.astype(float)); gs = replicate_stats((hit & inside).astype(float))
                ns = replicate_stats((hit & ~inside).astype(float)); ws = replicate_stats(weights[:, ik, it])
                rows.append(dict(scaffold=d['scaffold'], floppy=d['floppy'], kc=kc, threshold=delta, nq=d['nq'],
                                 volume_fraction=st['mean'], volume_ci95=st['ci95'], replicate_fractions=st['replicate_means'],
                                 gamma_contribution=gs['mean'], outside_gamma_contribution=ns['mean'], outside_gamma_ci95=ns['ci95'],
                                 gamma_rho_coefficient_1_contribution=float(np.mean(hit & (rho <= np.sqrt(kc)))),
                                 gamma_rho_coefficient_4_contribution=float(np.mean(hit & (rho <= 4 * np.sqrt(kc)))),
                                 fixed_gamma_rho_005_contribution=float(np.mean(hit & (rho <= .05))),
                                 outside_fixed_gamma_005=float(np.mean(hit & (rho > .05))),
                                 reference_spectral_weight_outside=ws['mean'], reference_spectral_weight_ci95=ws['ci95'],
                                 selected_projector_weight_mean=float(selection[:, ik].mean()),
                                 selected_projector_weight_p05=float(np.quantile(selection[:, ik], .05))))
    C.csvout(Path(out) / 'zone_fraction_summary.csv', rows)
    grows = []
    for path in sorted((Path(raw) / 'gain').glob('*.json')):
        d = json.loads(path.read_text(encoding='utf8'))
        if not d.get('complete'): continue
        a = np.array(d['rows']); mask = np.array([s['kind'] == 'BZ_uniform' for s in d['samples']])
        for ik, kc in enumerate(d['kc_over_ks']):
            for ir, ref in enumerate(d['reference_squared_over_kc']):
                g = a[mask, ik, ir]; st = replicate_stats((g > 1e-5).astype(float)); ms = replicate_stats(g / (d['h'] / 4))
                grows.append(dict(scaffold=d['scaffold'], floppy=d['floppy'], kc=kc, reference_squared_over_kc=ref, nq=int(mask.sum()),
                                  gain_q_fraction=st['mean'], gain_fraction_ci95=st['ci95'], mean_gain_over_reference=float(g.mean()),
                                  mean_gain_fraction_of_Mathieu=ms['mean'], mean_gain_fraction_ci95=ms['ci95'],
                                  gain_quantiles_05_50_95=np.quantile(g, [.05, .5, .95]).tolist(),
                                  max_step_doubling_error_over_reference=max(x['error'] for x in d['step_checks'])))
    C.csvout(Path(out) / 'gain_other_pairs_summary.csv', grows)
    print('rows', len(rows), len(grows), '->', out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['static', 'gain', 'summarize'])
    ap.add_argument('--pair', help='scaffold:floppy (default: all ten for static, the eight splitting pairs for gain)')
    ap.add_argument('--nq', type=int, help='first N wavevectors only (quick test)')
    ap.add_argument('--raw', default=str(ROOT / 'outputs' / 'finite_stiffness' / 'raw'))
    ap.add_argument('--out', default=str(ROOT / 'outputs' / 'finite_stiffness'))
    E.add_argument(ap); args = ap.parse_args(); E.apply_argument(args)
    if args.mode == 'summarize':
        summarize(args.raw, args.out); return
    todo = [tuple(args.pair.split(':'))] if args.pair else (STATIC_PAIRS if args.mode == 'static' else PAIRS)
    for sc, fl in todo:
        data = (static if args.mode == 'static' else gain)(sc, fl, args.nq)
        C.dump(Path(args.raw) / args.mode / f'{pair_id(sc, fl)}.json', data)
        if args.mode == 'static':
            dev = np.array([[x['max_deviation'] for x in r] for r in data['rows']])
            print(sc, fl, 'nq', len(dev), 'fraction outside 1% at k_c = 1e-3:', float(np.mean(dev[:, 0] > .01)), flush=True)
        else:
            print(sc, fl, 'samples', len(data['rows']), flush=True)


if __name__ == '__main__':
    main()
