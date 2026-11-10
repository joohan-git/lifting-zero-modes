"""Band of the planar pair hcb (x) hxl along Gamma-K-M-Gamma and over the zone (SM Section S6), full composite.

Model: A at 0, B at (1,0) (hcb), hexagon centre C at (-1,0) (hxl); a1 = (3/2, sqrt3/2), a2 = (3/2, -sqrt3/2);
unit masses; hcb bonds k_f = 1, hxl bonds k_s = 1, contacts k_c = 0.01. At each wavevector the lifted band is the
eigenvalue with the largest overlap with the hcb mechanism. Also: transverse sound speed c_T, pocket edge
q_p = sqrt(3k_c/2)/c_T, zone median of omega^2/k_c over 2,000 random wavevectors (seed 0), and the largest
deviation from 3/2 between K and M in units of k_c^2.

  python planar_pair/planar_band.py [--out FILE]      (stored result: planar_pair/planar_band.json)
"""
import argparse
import itertools
import json
import math
from pathlib import Path
import numpy as np

KC = 0.01
S3 = math.sqrt(3)

# ====================================================================== model (hcb in hxl)
a1 = np.array([1.5, S3 / 2]); a2 = np.array([1.5, -S3 / 2])
SITES = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([-1.0, 0.0])]   # A, B (hcb), C (hxl)
SH = [(i, j) for i, j in itertools.product([-1, 0, 1], repeat=2)]


def _nbrs(i, j, dist):
    out = []
    for si, sj in SH:
        d = SITES[j] + si * a1 + sj * a2 - SITES[i]
        if abs(np.linalg.norm(d) - dist) < 1e-9:
            out.append((np.array([si, sj]), d))
    return out


FB = [(0, 1, sh, d, 'f') for sh, d in _nbrs(0, 1, 1.0)]
_sb = [(2, 2, sh, d, 's') for sh, d in _nbrs(2, 2, S3)]
SB = []
for b in _sb:
    if not any(np.allclose(b[3], -c[3]) for c in SB):
        SB.append(b)
CB = [(i, 2, sh, d, 'c') for i in (0, 1) for sh, d in _nbrs(i, 2, 1.0)]
BONDS = FB + SB + CB
assert (len(FB), len(SB), len(CB)) == (3, 3, 6)


def D_of(q, kc=KC, kf=1.0, ks=1.0):
    D = np.zeros((6, 6), complex)
    for (i, j, sh, d, kd) in BONDS:
        k = {'f': kf, 's': ks, 'c': kc}[kd]; u = d / np.linalg.norm(d)
        ph = np.exp(2j * np.pi * (q @ sh)); K = k * np.outer(u, u)
        si, sj = slice(2 * i, 2 * i + 2), slice(2 * j, 2 * j + 2)
        D[si, si] += K; D[sj, sj] += K; D[si, sj] -= K * ph; D[sj, si] -= K * np.conj(ph)
    return D


def mech(q):
    """hcb mechanism (kernel of the hcb compatibility matrix), padded with zeros on the scaffold."""
    R = []
    for (i, j, sh, d, kd) in FB:
        u = d / np.linalg.norm(d); ph = np.exp(2j * np.pi * (q @ sh))
        row = np.zeros(4, complex); row[2 * j:2 * j + 2] += u * ph; row[2 * i:2 * i + 2] -= u; R.append(row)
    v = np.linalg.svd(np.array(R))[2][-1].conj()
    return np.concatenate([v, np.zeros(2)])


Bm = 2 * np.pi * np.linalg.inv(np.array([a1, a2])).T          # rows b1, b2
red = lambda qc: np.linalg.solve(Bm.T, qc)                   # Cartesian -> reduced
G0 = np.zeros(2); KP = (2 * Bm[0] + Bm[1]) / 3; MP = Bm[0] / 2  # Gamma, K (|K| = 4pi/3sqrt3), M
assert abs(np.linalg.norm(KP) - 4 * math.pi / (3 * S3)) < 1e-12
CORN = [G0, KP, MP, G0]
SEGL = [np.linalg.norm(CORN[i + 1] - CORN[i]) for i in range(3)]
XC = np.concatenate([[0], np.cumsum(SEGL)])                  # path coordinate of corners


def path_pt(s):
    i = min(int(np.searchsorted(XC, s, side='right') - 1), 2)
    t = (s - XC[i]) / SEGL[i]
    return CORN[i] + t * (CORN[i + 1] - CORN[i])


def solve(s, kc=KC):
    q = red(path_pt(s)); w, V = np.linalg.eigh(D_of(q, kc))
    ov = np.abs(mech(q).conj() @ V) ** 2
    return w, int(np.argmax(ov)), float(ov.max())


SF = np.linspace(0, XC[-1], 600)
ALLW, MW = [], []
for s in SF:
    w, k, o = solve(s); ALLW.append(w); MW.append(w[k])
ALLW, MW = np.array(ALLW) / KC, np.array(MW) / KC
SMK = np.linspace(0, XC[-1], 34)
MKW = np.array([solve(s)[0][solve(s)[1]] for s in SMK]) / KC

# transverse sound speed of the composite and pocket edge q_p = sqrt(3k_c/2)/c_T (SM S6.3)
qs = 1e-3 * MP / np.linalg.norm(MP)
cT = math.sqrt(np.sort(np.linalg.eigvalsh(D_of(red(qs))))[0]) / 1e-3
QP = math.sqrt(1.5 * KC) / cT
rng = np.random.default_rng(0)
zone = []
for r in rng.random((2000, 2)):
    w, V = np.linalg.eigh(D_of(r))                 # same rule as the markers: largest mechanism overlap
    zone.append(w[int(np.argmax(np.abs(mech(r).conj() @ V) ** 2))] / KC)
zone = np.array(zone)
OUT = dict(kc=KC, ks=1.0, path_corners=['G', 'K', 'M', 'G'], path_corner_x=XC.tolist(),
           marker_x=SMK.tolist(), marker_w2_over_kc=MKW.tolist(), cT=cT, q_p=QP,
           q_p_over_sqrt_kc=QP / math.sqrt(KC), zone_median_w2_over_kc=float(np.median(zone)),
           zone_q05_q95=[float(np.quantile(zone, .05)), float(np.quantile(zone, .95))],
           residual_at_KM_max_over_kc2=float(np.max(np.abs(MW[(SF > XC[1]) & (SF < XC[2])] - 1.5)) / KC))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=str(Path(__file__).resolve().parents[1] / 'outputs' / 'planar_pair' / 'planar_band.json'))
    p = Path(ap.parse_args().out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(OUT, indent=1), encoding='utf8'); print('written', p)
