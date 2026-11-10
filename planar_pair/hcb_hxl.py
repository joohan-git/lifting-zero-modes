"""Planar dual pair hxl (x) hcb: closed form and full-composite checks (main text; SM Sections S5.2, S6).

Model: hcb vertices A at 0 and B at (1,0), hexagon centre C (hxl vertex) at (-1,0), lattice vectors
a1 = (3/2, sqrt3/2), a2 = (3/2, -sqrt3/2); unit masses, hcb bonds k_f = 1, hxl bonds k_s = 1, contacts k_c.

  python planar_pair/hcb_hxl.py first-order     # S_i = (3/2) k_c 1 and the fixed-scaffold band
  python planar_pair/hcb_hxl.py quantiles       # single-band quantiles, 400 random wavevectors (Table S7)
  python planar_pair/hcb_hxl.py gtilde          # second-order coefficient along Gamma-M (Section S6.3)
"""
import argparse
import itertools
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
S3 = math.sqrt(3)
a1 = np.array([1.5, S3 / 2]); a2 = np.array([1.5, -S3 / 2])
SITES = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([-1.0, 0.0])]   # A, B (hcb), C (hxl)
SH = [(i, j) for i, j in itertools.product([-1, 0, 1], repeat=2)]


def _nbrs(i, j, dist):
    out = []
    for si, sj in SH:
        d = SITES[j] + si * a1 + sj * a2 - SITES[i]
        if abs(np.linalg.norm(d) - dist) < 1e-9: out.append((np.array([si, sj]), d))
    return out


FB = [(0, 1, sh, d, 'f') for sh, d in _nbrs(0, 1, 1.0)]
SB = []
for b in [(2, 2, sh, d, 's') for sh, d in _nbrs(2, 2, S3)]:
    if not any(np.allclose(b[3], -c[3]) for c in SB): SB.append(b)
CB = [(i, 2, sh, d, 'c') for i in (0, 1) for sh, d in _nbrs(i, 2, 1.0)]
BONDS = FB + SB + CB
assert (len(FB), len(SB), len(CB)) == (3, 3, 6)


def D_of(q, kc, kf=1.0, ks=1.0):
    D = np.zeros((6, 6), complex)
    for (i, j, sh, d, kd) in BONDS:
        k = {'f': kf, 's': ks, 'c': kc}[kd]; u = d / np.linalg.norm(d)
        ph = np.exp(2j * np.pi * (q @ sh)); K = k * np.outer(u, u)
        si, sj = slice(2 * i, 2 * i + 2), slice(2 * j, 2 * j + 2)
        D[si, si] += K; D[sj, sj] += K; D[si, sj] -= K * ph; D[sj, si] -= K * np.conj(ph)
    return D


def compat_hcb(q):
    R = []
    for (i, j, sh, d, kd) in FB:
        u = d / np.linalg.norm(d); ph = np.exp(2j * np.pi * (q @ sh))
        row = np.zeros(4, complex); row[2 * j:2 * j + 2] += u * ph; row[2 * i:2 * i + 2] -= u; R.append(row)
    return np.array(R)


def mech(q):
    """hcb mechanism (kernel of the hcb compatibility matrix), padded with zeros on the scaffold."""
    v = np.linalg.svd(compat_hcb(q))[2][-1].conj()
    return np.concatenate([v, np.zeros(2)])


Bm = 2 * np.pi * np.linalg.inv(np.array([a1, a2])).T          # rows b1, b2
red = lambda qc: np.linalg.solve(Bm.T, qc)                   # Cartesian -> reduced
MP = Bm[0] / 2                                               # M point


def contact_tensor():
    S = np.zeros((2, 2))
    for (i, j, sh, d, kd) in CB:
        if i == 0: u = d / np.linalg.norm(d); S += np.outer(u, u)
    return S


def first_order(nq=200):
    S = contact_tensor(); Sfull = np.kron(np.eye(2), S)
    rng = np.random.default_rng(0); vals = []; dims = []
    for q in rng.random((nq, 2)):
        R = compat_hcb(q); _, s, vh = np.linalg.svd(R); Z = vh[int(np.sum(s > 1e-8 * s[0])):].conj().T
        dims.append(Z.shape[1]); vals.append(np.linalg.eigvalsh(Z.conj().T @ Sfull @ Z))
    v = np.concatenate(vals)
    return dict(S_A_over_kc=S.tolist(), mechanisms_per_cell=sorted(set(dims)), lifted_min_over_kc=float(v.min()),
                lifted_max_over_kc=float(v.max()), nq=nq)


def overlap_band(q, kc):
    w, V = np.linalg.eigh(D_of(q, kc))
    return w[int(np.argmax(np.abs(mech(q).conj() @ V) ** 2))]


def quantiles():
    rng = np.random.default_rng(0); QS = rng.random((400, 2))
    out = dict(rule='eigenvalue with the largest overlap with the hcb mechanism, one per wavevector', n_wavevectors=len(QS), seed=0)
    for kc in [0.003, 0.01, 0.03]:
        v = np.array([overlap_band(q, kc) / kc for q in QS])
        out[f'{kc:g}'] = dict(q05_q50_q95=[float(x) for x in np.percentile(v, [5, 50, 95])])
    return out


def gtilde():
    uM = MP / np.linalg.norm(MP); QM = float(np.linalg.norm(MP))

    def gt(qabs, kc):
        return (1.5 * kc - overlap_band(red(qabs * uM), kc)) / kc ** 2
    KCS = [1e-4, 1e-5]; POINTS = [1.0, 0.6, 0.3]
    out = dict(model='hxl (x) hcb, k_s = k_f = 1, unit masses, unit hcb edge length', path='Gamma-M', M_point_abs_q=QM, kc=KCS,
               g_at_points={f'{p:g}': [gt(p, kc) for kc in KCS] for p in POINTS})
    qs = np.linspace(0.2, 0.6, 21); g = np.array([gt(x, KCS[0]) for x in qs])
    Cc = float(np.sum(g / qs ** 2) / np.sum(1 / qs ** 4)); slope = float(np.polyfit(np.log(qs), np.log(g), 1)[0])
    out.update(fit_range=[float(qs[0]), float(qs[-1])], fit_kc=KCS[0], C_inverse_square=Cc, loglog_slope=slope,
               fit_max_rel_residual=float(np.max(np.abs(g - Cc / qs ** 2) / g)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('task', choices=['first-order', 'quantiles', 'gtilde'])
    ap.add_argument('--out', default=str(ROOT / 'outputs' / 'planar_pair'))
    args = ap.parse_args()
    res = {'first-order': first_order, 'quantiles': quantiles, 'gtilde': gtilde}[args.task]()
    name = {'first-order': 'first_order.json', 'quantiles': 'hcb_band_quantiles.json', 'gtilde': 'gtilde_values.json'}[args.task]
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / name).write_text(json.dumps(res, indent=1), encoding='utf8')
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
