"""Fabrication-tolerance ensemble: 3 x 3 x 3 periodic supercells of the beam-frame composite with independent random
errors on every contact stiffness and every floppy-strut bending stiffness, k -> k (1 + delta xi), xi uniform on
[-1, 1], delta = 0, 1, 3, 10 %. Twelve seeds, two supercell wavevectors per seed, each drawn so that all 27 folded
wavevectors have a floppy-net gap >= 0.05 (fcu/flu: the best of 200 draws, recorded as min_folded_gap). The lifted band is the 27 M supercell modes with the largest overlap with
the supercell mechanism space. Reported per delta: relative width of the lifted density of states (standard deviation
and 5-95 % range over the mean, pooled), and the participation number of the lifted eigenvectors over floppy nodes,
divided by the number of mechanism-carrying nodes in the supercell (1 = extended, 1/N = one node), with the
equivalent length xi = a (PR / n_mc)^(1/3), n_mc mechanism nodes per cell.
k_c = 1e-3, k_b = k_c, thick cage (10, 1), kf_a = 1, k_cr = 100.
Run: python metamaterial/sim_tolerance.py --rcsr DIR   (stored result: metamaterial/tolerance.json)
"""
import json
from multiprocessing import Pool
import numpy as np
import scipy.linalg as la
import sim_core as S

KC = 1e-3
CASES = [
    dict(name='nbo_lock', pair=("bcu", "nbo"), design='lock'),
    dict(name='nbo_free', pair=("bcu", "nbo"), design='rigid'),
    dict(name='nbo_release', pair=("bcu", "nbo"), design='release'),
    dict(name='niad_lock', pair=("nia", "nia-d"), design='lock'),
    dict(name='niad_free', pair=("nia", "nia-d"), design='rigid'),
    dict(name='bnn_release', pair=("hex", "bnn"), design='release'),
    dict(name='flu_release', pair=("fcu", "flu"), design='release'),
]
DELTA = [0.0, 0.01, 0.03, 0.1]
SEEDS = 12
NQ = 2
N = 3


def supercell_Q(m, rng, tries=200):
    """First draw whose 27 folded wavevectors all have gap >= 0.05; if none in `tries` draws (fcu/flu, whose gap
    closes on much of the zone), the draw with the largest smallest folded gap. Returns Q and that smallest gap."""
    best = (-1.0, None)
    for _ in range(tries):
        Q = rng.random(3)
        g = min(m.gap((Q + np.array(G)) / N) for G in np.ndindex(N, N, N))
        if g >= 0.05:
            return Q, g
        if g > best[0]:
            best = (g, Q)
    return best[1], best[0]


def job(args):
    case, delta, seed = args
    m = S.Model(*case['pair']); lat = S.Lattice(m, (N,) * 3, periodic=True)
    inst = lat.instances(); nc, nfb = lat.count['c'], lat.count['f']
    rng = np.random.default_rng(1000 * seed + 7)
    cm = 1 + delta * rng.uniform(-1, 1, nc); bm = 1 + delta * rng.uniform(-1, 1, nfb)
    Zw = np.zeros(m.nf); r2 = np.random.default_rng(2)
    for _ in range(16):
        Z = m.Z(r2.random(3) - 0.5)[:, :m.M]; Zw += np.sum(np.abs(Z.reshape(m.nf, 3, -1)) ** 2, axis=(1, 2))
    mc = Zw > 1e-8 * Zw.max(); n_mc = int(mc.sum())
    qrng = np.random.default_rng(seed)
    vals, prs, gaps = [], [], []
    for _ in range(NQ):
        Q, gq = supercell_Q(m, qrng); gaps.append(float(gq))
        K = lat.assemble(Q, contact_mult=cm, bend_mult=bm, kc=KC, kf_a=1.0, kf_b=KC, ks_a=10.0, ks_b=1.0,
                         design=case['design'], kcr=100.0)
        Ke = lat.condensed(K); w, v = la.eigh(Ke)
        Z = lat.mech(Q); nM = m.M * lat.ncell
        wt = np.sum(np.abs(Z.conj().T @ v) ** 2, axis=0); idx = np.sort(np.argsort(-wt)[:nM])
        fl = np.where(lat.is_floppy)[0]
        for k in idx:
            u = v[:, k].reshape(lat.nn, 3); p = np.sum(np.abs(u[fl]) ** 2, axis=1)
            prs.append(float(p.sum() ** 2 / np.sum(p ** 2)))
        vals += (w[idx] / KC).tolist()
    return dict(name=case['name'], delta=delta, seed=seed, values=vals, pr=prs, min_gap=min(gaps), n_mc_cell=n_mc, n_mc=n_mc * lat.ncell)


if __name__ == '__main__':
    jobs = [(c, d, s) for c in CASES for d in DELTA for s in range(SEEDS)]
    with Pool(14) as P:
        rows = P.map(job, jobs, chunksize=1)
    out = dict(params=dict(kc=KC, kb=KC, ks=(10.0, 1.0), kf_a=1.0, kcr=100.0, N=N, seeds=SEEDS, nq=NQ, delta=DELTA,
                           gap_min=0.05), cases={})
    for c in CASES:
        rec = []
        for d in DELTA:
            rr = [r for r in rows if r['name'] == c['name'] and r['delta'] == d]
            v = np.concatenate([r['values'] for r in rr]); pr = np.concatenate([r['pr'] for r in rr])
            n_mc, n_cell = rr[0]['n_mc'], rr[0]['n_mc_cell']
            # per-seed widths for an error bar
            ws = [float(np.std(r['values']) / np.mean(r['values'])) for r in rr]
            rec.append(dict(delta=d, min_folded_gap=float(min(r['min_gap'] for r in rr)), mean=float(v.mean()), rel_std=float(v.std() / v.mean()),
                            rel_range_5_95=float((np.quantile(v, .95) - np.quantile(v, .05)) / v.mean()),
                            rel_std_seed_mean=float(np.mean(ws)), rel_std_seed_sd=float(np.std(ws)),
                            pr_mean=float(pr.mean()), pr_median=float(np.median(pr)), pr_over_nmc=float(pr.mean() / n_mc),
                            xi_cells=float((pr.mean() / n_cell) ** (1 / 3)), n_mc=n_mc, n_mc_cell=n_cell,
                            hist=np.histogram(v, bins=60)[0].tolist(), hist_edges=np.histogram(v, bins=60)[1].tolist()))
            print(c['name'], d, rec[-1]['rel_std'], rec[-1]['pr_over_nmc'], rec[-1]['xi_cells'], flush=True)
        out['cases'][c['name']] = dict(case=c, rows=rec)
    json.dump(out, open(S.out('tolerance.json'), 'w'), indent=1)
    print('done')
