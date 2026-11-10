"""Parametric drive k_c(t) = k_c [1 + h cos(Omega t)] of the full composite (main text; Methods; SM S8).

Linear gain = largest Floquet exponent of the full monodromy matrix over one drive period, from exact
free drift in the eigenbasis of D(q) and midpoint kicks of the contact Hessian k_c C^H C (cross terms
between floppy and scaffold vertices included; the geometric term vanishes at rest length = length).
k_c = 1e-3, h = 0.1, no damping; gains in units of the reference frequency omega_ref = sqrt(c k_c).

Models: nbo = bcu (x) nbo, flu = fcu (x) flu (RCSR embedding, primitive cell), hcb = hxl (x) hcb,
dia = dia (x) dia (built here).

  python drive/gain.py spectrum nbo --rcsr DIR [--nq 256]   # 175 drive frequencies, uniform q
  python drive/gain.py spectrum hcb --nq 8
  python drive/gain.py special --rcsr DIR                    # zone centre and other special points
  python drive/gain.py count                                 # counts from the stored files
"""
import argparse
import itertools
import json
import sys
from pathlib import Path
import numpy as np
import scipy.linalg as la
from scipy.integrate import solve_ivp
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from common import nets as E
from common import composite as C

REF = {'nbo': 2, 'hcb': 1.5, 'flu': 14 / 9, 'dia': 2}
KC = .001; H = .1


def nearest_edges(A, pts):
    d = len(A); cand = []
    for i, j in itertools.product(range(len(pts)), repeat=2):
        for s in itertools.product(range(-1, 2), repeat=d):
            v = (pts[j] + s - pts[i]) @ A; l = la.norm(v)
            if l > 1e-8: cand.append((l, E.canon(i, j, s)))
    l0 = min(x[0] for x in cand); return sorted(set(e for l, e in cand if l < 1.001 * l0))


def assemble_analytic(name):
    if name == 'hcb':
        A = np.array([[1., 0], [.5, np.sqrt(3) / 2]]); fp = np.array([[1 / 3, 1 / 3], [2 / 3, 2 / 3]]); sp = np.zeros((1, 2))
    else:
        A = np.eye(3); fcc = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
        fp = np.vstack([fcc, fcc + .25]); sp = (fp + .5) % 1
    fl = C.from_positions(name, A, fp, nearest_edges(A, fp)); sc = C.from_positions('hxl' if name == 'hcb' else 'dia', A, sp, nearest_edges(A, sp))
    cs = []; d = len(A)
    for i, p in enumerate(fp):
        candidates = []
        for j, x in enumerate(sp):
            for s in itertools.product(range(-1, 2), repeat=d):
                v = (x + s - p) @ A; l = la.norm(v)
                if l > 1e-8: candidates.append((j, s, v, l))
        dn = min(c[3] for c in candidates)
        for j, s, v, l in candidates:
            if l < 1.05 * dn: cs.append((i, j, np.array(s), v / l, float(l)))
    p = dict(fl=fl, sc=sc, contacts=cs)
    return p if name == 'hcb' else C.primitive(p, np.array([[0, .5, .5], [.5, 0, .5], [.5, .5, 0]]))


def get_model(name):
    if name in ('hcb', 'dia'): return assemble_analytic(name)
    sc, fl = {'nbo': ('bcu', 'nbo'), 'flu': ('fcu', 'flu')}[name]
    p = C.pair(sc, fl, [0, 0, 0]); _, B = E.sg_ops(C.CG()[sc], p['sc']['setting']); return C.primitive(p, B)


def monodromy(lam, P, omega, h, steps):
    n = len(lam); dt = 2 * np.pi / omega / steps; w = np.sqrt(np.maximum(lam, 0)); a = w * dt / 2
    c = np.cos(a); s = np.sinc(a / np.pi) * dt / 2; ws = -w * np.sin(a)
    X = np.c_[np.eye(n), np.zeros((n, n))].astype(complex); V = np.c_[np.zeros((n, n)), np.eye(n)].astype(complex)
    for j in range(steps):
        X, V = c[:, None] * X + s[:, None] * V, ws[:, None] * X + c[:, None] * V
        V -= dt * h * np.cos(2 * np.pi * (j + .5) / steps) * (P @ X)
        X, V = c[:, None] * X + s[:, None] * V, ws[:, None] * X + c[:, None] * V
    mu = la.eigvals(np.r_[X, V], check_finite=False)
    return max(0, float(np.log(np.max(np.abs(mu))) * omega / (2 * np.pi)))


RATIOS = np.unique(np.round(np.r_[np.linspace(.5, 3.5, 101), np.linspace(1.90, 2.10, 81)], 8))


def spectrum(model, nq, steps=192):
    p = get_model(model); d = p['fl']['d']; wf = np.sqrt(REF[model] * KC)
    qs = np.random.default_rng(C.SEED + 7).random((nq, d))
    data = dict(model=model, kc=KC, kF=1., kS=1., mass=1., h=H, damping=0., omega_reference=wf, reference_squared_over_kc=REF[model],
                reference_meaning='flat first-order frequency' if model != 'dia' else 'reference sqrt(2 kc); dia has no flat frequency',
                q_sampling='uniform primitive reciprocal parallelepiped', seed=C.SEED + 7, q=qs.tolist(), Omega_over_reference=RATIOS.tolist(),
                steps=steps, gain_over_reference=[], step_check=[],
                pump='full contact Hessian kc*C(q)^H*C(q), including F-S cross terms; geometric term exactly zero because l0=l')
    for iq, q in enumerate(qs):
        D, P = C.full_matrices(p, q, KC); w, U = la.eigh(D); P = U.conj().T @ P @ U
        data['gain_over_reference'].append([monodromy(w, P, r * wf, H, steps) / wf for r in RATIOS])
        if iq < 4:
            for r in (1., 1.975, 2., 2.025, 2.5):
                g1 = monodromy(w, P, r * wf, H, steps) / wf; g2 = monodromy(w, P, r * wf, H, steps * 2) / wf
                data['step_check'].append(dict(q_index=iq, ratio=r, gain=g1, gain_double_steps=g2, absolute_difference=abs(g1 - g2)))
    G = np.array(data['gain_over_reference']); data['quantiles_05_50_95'] = np.quantile(G, [.05, .5, .95], axis=0).tolist()
    data['positive_q_fraction'] = (G > 1e-5).mean(0).tolist(); data['complete'] = True
    return data


def special(models):
    rows = []
    for name in models:
        p = get_model(name); wf = np.sqrt(KC * REF[name]); ratios = RATIOS
        if name == 'hcb':
            points = [('Gamma', [0, 0]), ('M', [.5, 0]), ('K', [2 / 3, 1 / 3]), ('P', [1 / 3, 1 / 3])]
        else:
            B = np.array([[0, .5, .5], [.5, 0, .5], [.5, .5, 0]]) if name == 'dia' else E.sg_ops(C.CG()['bcu' if name == 'nbo' else 'fcu'])[1]
            points = [(label, (np.array(q) @ B.T).tolist()) for label, q in
                      [('Gamma', [0, 0, 0]), ('conv_110_quarter', [.25, .25, 0]), ('conv_100_quarter', [.25, 0, 0]), ('conv_111_quarter', [.25, .25, .25])]]
        for label, q in points:
            D, P = C.full_matrices(p, q, KC); lam, U = la.eigh(D); PP = U.conj().T @ P @ U
            G = np.array([monodromy(lam, PP, r * wf, H, 192) / wf for r in ratios]); i = int(np.argmin(abs(ratios - 2)))
            extra = {}
            if label == 'Gamma':
                n = len(lam); omega = 2 * wf; T = 2 * np.pi / omega

                def rhs(t, y):
                    Y = y.reshape(2 * n, 2 * n); return np.r_[Y[n:], -(D + H * np.cos(omega * t) * P) @ Y[:n]].ravel()
                sol = solve_ivp(rhs, (0, T), np.eye(2 * n, dtype=complex).ravel(), method='DOP853', rtol=2e-10, atol=2e-12)
                mu = la.eigvals(sol.y[:, -1].reshape(2 * n, 2 * n)); extra['adaptive_gain_at_ratio_2'] = max(0, float(np.log(max(abs(mu))) / T / wf))
            rows.append(dict(model=name, label=label, q_primitive=q, Omega_over_reference=ratios.tolist(), gain_over_reference=G.tolist(),
                             gain_at_ratio_2=float(G[i]), gain_at_ratio_2_double_steps=monodromy(lam, PP, 2 * wf, H, 384) / wf, **extra,
                             static_eigenvalues_over_kc=(lam / KC).tolist(), included_in_uniform_BZ_quantiles=False))
            print(name, label, rows[-1]['gain_at_ratio_2'], flush=True)
    return rows


def count(directory):
    """Wavevectors that grow at Omega = 2 omega_ref (g > 1e-5) and those within 4% of h/4."""
    for m in ['nbo', 'flu', 'hcb', 'dia']:
        d = json.loads((Path(directory) / f'{m}.json').read_text(encoding='utf8'))
        i = d['Omega_over_reference'].index(2.0); g = np.array([row[i] for row in d['gain_over_reference']]); ref = d['h'] / 4
        print(m, 'wavevectors', len(g), 'growing', int(np.sum(g > 1e-5)), 'within 4% of h/4', int(np.sum(abs(g / ref - 1) < 0.04)),
              'median g/omega', round(float(np.median(g)), 5))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('task', choices=['spectrum', 'special', 'count'])
    ap.add_argument('model', nargs='?', choices=list(REF))
    ap.add_argument('--nq', type=int, default=256)
    ap.add_argument('--steps', type=int, default=192)
    ap.add_argument('--out', default=str(ROOT / 'outputs' / 'drive'))
    E.add_argument(ap); args = ap.parse_args(); E.apply_argument(args)
    if args.task == 'count':
        count(HERE / 'gain'); return
    if args.task == 'spectrum':
        if not args.model: raise SystemExit('give a model: ' + ' '.join(REF))
        data = spectrum(args.model, args.nq, args.steps); C.dump(Path(args.out) / 'gain' / f'{args.model}.json', data)
        count(Path(args.out) / 'gain') if all((Path(args.out) / 'gain' / f'{m}.json').exists() for m in REF) else \
            print(args.model, 'growing at Omega = 2 omega_ref:', sum(r[list(RATIOS).index(2.0)] > 1e-5 for r in data['gain_over_reference']), 'of', args.nq)
    else:
        C.dump(Path(args.out) / 'gain_special_q.json', special([args.model] if args.model else list(REF)))


if __name__ == '__main__':
    main()
