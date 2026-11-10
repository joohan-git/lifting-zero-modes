"""Finite specimen in the time domain: 6 x 6 x 6 conventional cells of the beam-frame composite, cage clamped outside
the box (cage edges, contacts and locks that leave the box end on ground), mobile cage inside (k_s = 10 / 1),
floppy edges that leave the box removed. Rotations are condensed statically (sparse LU), unit nodal masses.

The condensed stiffness is diagonalized once and the response to a point force is integrated exactly mode by mode
(zero-order hold on the force, step T_on / 64), so every quality factor costs one pass. Damping is modal with
zeta = 1 / (2Q) for every mode. The force acts on the floppy node nearest the specimen centre, along its mechanism
polarization, as a Gaussian-windowed sine: sigma_t = 4 periods of the on-band carrier, centred at 4 sigma_t.
On-band carrier: the local-density-weighted mean of omega over modes with omega^2 < 20 k_c at the driven node.
Off-band carrier: the peak of the driven-node local density of states above omega^2 = 20 k_c (stiff floppy-net and
cage branches), same envelope. Energy per node e_i = |v_i|^2 / 2 + u_i . (K u)_i / 2 (clipped at zero) gives the
participation number N = (sum e)^2 / sum e^2 and the energy-weighted radius from the driven node.
k_c = 1e-3, k_b = k_c unless stated.
Run: python metamaterial/sim_time.py --rcsr DIR [case names]   (stored result: metamaterial/time.json, all eight
cases; the stored file was assembled from two runs of four cases each, which write time_<names>.json)
"""
import json, sys
import numpy as np
import scipy.linalg as la
import sim_core as S

NC = 6
KC = 1e-3
CASES = [
    dict(name='nbo_lock', pair=("bcu", "nbo"), design='lock', kf_a=1.0, kb=KC),
    dict(name='nbo_lock_kfa10', pair=("bcu", "nbo"), design='lock', kf_a=10.0, kb=KC),
    dict(name='nbo_free', pair=("bcu", "nbo"), design='rigid', kf_a=1.0, kb=KC),
    dict(name='nbo_release', pair=("bcu", "nbo"), design='release', kf_a=1.0, kb=KC),
    dict(name='niad_lock', pair=("nia", "nia-d"), design='lock', kf_a=1.0, kb=KC),
    dict(name='niad_free', pair=("nia", "nia-d"), design='rigid', kf_a=1.0, kb=KC),
    dict(name='bnn_free', pair=("hex", "bnn"), design='rigid', kf_a=1.0, kb=KC),
    dict(name='bnn_release', pair=("hex", "bnn"), design='release', kf_a=1.0, kb=KC),
]
QS = [np.inf, 200.0, 50.0]


def polarization(m, i):
    rng = np.random.default_rng(5); P = np.zeros((3, 3), complex)
    for _ in range(32):
        Z = m.Z(rng.random(3) - 0.5)[:, :m.M]; Zi = Z[3 * i:3 * i + 3]; P += Zi @ Zi.conj().T
    w, v = la.eigh(P.real)
    return v[:, -1]


def integrate(w2, Phi, b, wc, sigma, t0, tend, dt, Q, nout):
    """Exact modal integration with zero-order hold. Returns times, u and v (translations) at nout output times."""
    om = np.sqrt(np.maximum(w2, 1e-14)); z = 0.0 if not np.isfinite(Q) else 1.0 / (2 * Q)
    od = om * np.sqrt(1 - z * z); p = Phi.T @ b
    e = np.exp(-z * om * dt); c, s = np.cos(od * dt), np.sin(od * dt)
    # state transition for x'' + 2 z om x' + om^2 x = f
    a11 = e * (c + z * om / od * s); a12 = e * s / od
    a21 = -e * om * om / od * s; a22 = e * (c - z * om / od * s)
    g1 = (1 - a11) / (om * om); g2 = -a21 / (om * om)
    nst = int(np.ceil(tend / dt)); every = max(1, nst // nout)
    x = np.zeros_like(om); y = np.zeros_like(om); T, U, V = [], [], []
    for n in range(nst + 1):
        t = n * dt
        if n % every == 0:
            T.append(t); U.append(Phi @ x); V.append(Phi @ y)
        f = np.exp(-0.5 * ((t + 0.5 * dt - t0) / sigma) ** 2) * np.sin(wc * (t + 0.5 * dt))
        x, y = a11 * x + a12 * y + g1 * p * f, a21 * x + a22 * y + g2 * p * f
    return np.array(T), np.array(U), np.array(V)


def run(case):
    sc, fl = case['pair']; m = S.Model(sc, fl)
    lat = S.Lattice(m, (NC,) * 3, periodic=False)
    K = lat.assemble(kc=KC, kf_a=case['kf_a'], kf_b=case['kb'], ks_a=10.0, ks_b=1.0, design=case['design'], kcr=100.0)
    Ke = lat.condensed(K).real
    w2, Phi = la.eigh(Ke)
    nn = lat.nn; pos = lat.pos; centre = pos.mean(0)
    fl_nodes = np.where(lat.is_floppy)[0]
    # driven node: floppy node nearest the centre that carries a mechanism
    Zw = np.zeros(m.nf); rng = np.random.default_rng(2)
    for _ in range(16):
        Z = m.Z(rng.random(3) - 0.5)[:, :m.M]; Zw += np.sum(np.abs(Z.reshape(m.nf, 3, -1)) ** 2, axis=(1, 2))
    carriers = fl_nodes[Zw[lat.node_base[fl_nodes]] > 1e-8 * Zw.max()]
    i0 = int(carriers[np.argmin(np.linalg.norm(pos[carriers] - centre, axis=1))])
    pol = polarization(m, lat.node_base[i0])
    b = np.zeros(3 * nn); b[3 * i0:3 * i0 + 3] = pol
    ld = (Phi.T @ b) ** 2
    low = w2 < 20 * KC
    w_on = float(np.sqrt(np.sum(ld[low] * w2[low]) / np.sum(ld[low])))
    hi = ~low
    hist, edges = np.histogram(np.sqrt(np.maximum(w2[hi], 0)), bins=200, weights=ld[hi])
    k = int(np.argmax(hist)); w_off = float(0.5 * (edges[k] + edges[k + 1]))
    # spectral content of the on-band band at the driven node
    lw = ld[low] / ld[low].sum(); wl = np.sqrt(np.maximum(w2[low], 0))
    a_cell = float(np.linalg.norm(m.A[0]))
    T_on = 2 * np.pi / w_on; sigma = 4 * T_on; t0 = 4 * sigma; tend = t0 + 60 * T_on; dt = T_on / 64
    res = dict(case=case, n_nodes=int(nn), n_dof=int(3 * nn), driven_node=i0, polarization=pol.tolist(),
               w_on=w_on, w_off=w_off, sigma_t=sigma, t0=t0, t_end=tend, a_cell=a_cell,
               ldos_low_fraction=float(ld[low].sum() / ld.sum()),
               ldos_low_w_mean=float(np.sum(lw * wl)), ldos_low_w_std=float(np.sqrt(np.sum(lw * (wl - np.sum(lw * wl)) ** 2))),
               burst_rel_bandwidth=float(1 / (sigma * w_on)),
               n_low_modes=int(low.sum()), runs=[])
    d = np.linalg.norm(pos - pos[i0], axis=1) / a_cell
    for label, wc in (('on', w_on), ('off', w_off)):
        for Q in QS:
            if label == 'off' and np.isfinite(Q) and Q < 100:
                continue
            T, U, V = integrate(w2, Phi, b, wc, sigma, t0, tend, dt, Q, 240)
            KU = U @ Ke
            e = 0.5 * (V.reshape(len(T), nn, 3) ** 2).sum(2) + 0.5 * (U * KU).reshape(len(T), nn, 3).sum(2)
            e = np.maximum(e, 0); tot = e.sum(1) + 1e-300
            PR = tot ** 2 / ((e ** 2).sum(1) + 1e-300); R = np.sqrt((e * d[None, :] ** 2).sum(1) / tot)
            frac1 = (e[:, d <= 1.0].sum(1)) / tot
            order = np.argsort(d); cum = np.cumsum(e[:, order], axis=1) / tot[:, None]
            R50 = d[order][np.argmax(cum >= 0.5, axis=1)]; R90 = d[order][np.argmax(cum >= 0.9, axis=1)]
            snaps = {}
            for ts in (t0, t0 + 3 * sigma, tend):
                kk = int(np.argmin(abs(T - ts))); snaps[f'{T[kk]:.1f}'] = e[kk].tolist()
            res['runs'].append(dict(carrier=label, w=wc, Q=None if not np.isfinite(Q) else Q, t=T.tolist(), energy=tot.tolist(),
                                    PR=PR.tolist(), radius=R.tolist(), R50=R50.tolist(), R90=R90.tolist(), frac_within_1cell=frac1.tolist(), snapshots=snaps))
            late = T > t0 + 3 * sigma
            print(case['name'], label, Q, f'PR_late {PR[late].mean():.2f} R_late {R[late].mean():.3f} R90 {R90[late].mean():.2f} frac1 {frac1[late].mean():.3f}', flush=True)
    res['pos'] = (pos / a_cell).tolist(); res['is_floppy'] = lat.is_floppy.tolist()
    res['spectrum_low'] = (w2[low] / KC).tolist(); res['ldos_low'] = ld[low].tolist()
    return res


if __name__ == '__main__':
    names = sys.argv[1:] or [c['name'] for c in CASES]
    out = {}
    for c in CASES:
        if c['name'] in names:
            out[c['name']] = run(c)
            json.dump(out, open(S.out('time.json' if len(names) == len(CASES) else f"time_{'_'.join(names)}.json"), 'w'))
    print('done')
