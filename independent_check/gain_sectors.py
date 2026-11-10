"""Floquet gain with the second implementation, resolved into primitive-wavevector sectors (SM Section S8.2).

The conventional-cell composite commutes with the centring translations; its Bloch space at a
conventional wavevector splits into sectors that are the primitive wavevectors folded onto it. In each
sector the full monodromy over one drive period (DOP853, rtol 1e-9) gives the largest Floquet exponent.
k_c = 1e-3, h = 0.1, Omega = 2 sqrt(2 k_c), no damping; bcu (x) nbo (I centring, 24 random conventional
wavevectors) and dia (x) dia (F centring, 40 random), each with three special wavevectors.

  python independent_check/gain_sectors.py --rcsr DIR [--nrand 2] [--pairs bcu:nbo]
"""
import argparse
import os
import sys
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import engine as V
from common.composite import dump

kc, h = 1e-3, 0.1
CENT = {'I': [np.array([.5, .5, .5])], 'F': [np.array([0, .5, .5]), np.array([.5, 0, .5])]}

def translation_ops(sc, fl, t, q, cents):
    """Unitary U_c = e^{2 pi i q.c} T_c on the composite Bloch space; U_c^2 = 1."""
    pts = np.vstack([fl['pts'] + t, sc['pts']]); N = len(pts); ops = []
    for c in cents:
        U = np.zeros((3*N, 3*N), complex)
        for i in range(N):
            x = pts[i] + c; d = V.dfrac(pts, x); j = int(np.argmin(d)); assert d[j] < 1e-6, 'centering not a symmetry'
            s = np.rint(x - pts[j]); U[3*j:3*j+3, 3*i:3*i+3] = np.exp(-2j*np.pi*np.dot(q, s))*np.exp(2j*np.pi*np.dot(q, c))*np.eye(3)
        ops.append(U)
    return ops

def sectors(ops):
    H = sum((3**k)*U for k, U in enumerate(ops)); H = (H + H.conj().T)/2
    w, Q = np.linalg.eigh(H); labels = np.round(w).astype(int); out = []
    for lab in sorted(set(labels)):
        B = Q[:, labels == lab]; out.append((int(lab), B))
    return out

def gain(D0, Pc, Omega, rtol=1e-9):
    N = D0.shape[0]; T = 2*np.pi/Omega
    def f(t, y):
        Y = y.reshape(2*N, 2*N); D = D0 + kc*(1 + h*np.cos(Omega*t))*Pc
        dY = np.empty_like(Y); dY[:N] = Y[N:]; dY[N:] = -D @ Y[:N]; return dY.ravel()
    sol = solve_ivp(f, (0, T), np.eye(2*N, dtype=complex).ravel(), method='DOP853', rtol=rtol, atol=1e-12)
    M = sol.y[:, -1].reshape(2*N, 2*N); return float(np.log(np.abs(np.linalg.eigvals(M)).max())/T)

def run(cg, cases):
    res = {}
    for sc_name, fl_name, cent, nrand in cases:
        out, sc, fl, P, S, unitF, unitS = V.screen_pair(cg, sc_name, fl_name, nq=5)
        wref = np.sqrt(2*kc); Omega = 2*wref; rng = np.random.default_rng(7)
        qs = [np.zeros(3), np.array([.25, .25, 0]), np.array([.5, 0, 0])] + [rng.random(3) for _ in range(nrand)]
        rows = []
        for qi, q in enumerate(qs):
            RF = V.rigidity(fl, q, unitF); RS = V.rigidity(sc, q, unitS); nF = 3*fl['V']; N = nF + 3*sc['V']
            D0 = np.zeros((N, N), complex); D0[:nF, :nF] = RF.conj().T @ RF; D0[nF:, nF:] = RS.conj().T @ RS
            C = V.contact_matrix(fl, sc, P['contacts'], q); Pc = C.conj().T @ C
            ops = translation_ops(sc, fl, P['shift'], q, CENT[cent])
            for lab, B in sectors(ops):
                D0s = B.conj().T @ D0 @ B; Pcs = B.conj().T @ Pc @ B
                # sanity: sector invariance
                comm = max(np.abs(D0 @ B - B @ D0s).max(), np.abs(Pc @ B - B @ Pcs).max())
                w2 = np.linalg.eigvalsh(D0s + kc*Pcs)
                near = int(np.sum(np.abs(np.sqrt(np.abs(w2)) - wref) < 0.05*wref))
                g = gain(D0s, Pcs, Omega)
                rows.append(dict(q=np.round(q, 4).tolist(), sector=lab, dim=B.shape[1], g_over_wref=g/wref, n_modes_near_wref=near,
                                 lowest_w2_over_kc=np.round(np.sort(w2)[:6]/kc, 4).tolist(), block_residual=float(comm)))
                if qi < 3: print(sc_name, fl_name, 'q', rows[-1]['q'], 'sector', lab, 'dim', B.shape[1], 'g/wref', round(g/wref, 6), 'near', near,
                                 'w2/kc', rows[-1]['lowest_w2_over_kc'], 'resid', f'{comm:.1e}', flush=True)
        gs = np.array([r['g_over_wref'] for r in rows if r['q'] != [0, 0, 0]])
        summ = dict(n_primitive_q=len(gs), frac_gain=float(np.mean(gs > 1e-5)), median=float(np.median(gs)), p5=float(np.percentile(gs, 5)), p95=float(np.percentile(gs, 95)))
        print(sc_name, fl_name, 'SUMMARY (random + special, excl. Gamma):', summ, flush=True)
        res[f'{sc_name}|{fl_name}'] = dict(summary=summ, rows=rows)
    return res


CASES = {('bcu', 'nbo'): ('I', 24), ('dia', 'dia'): ('F', 40)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pairs', nargs='*', help='bcu:nbo and/or dia:dia (default: both)')
    ap.add_argument('--nrand', type=int, help='number of random conventional wavevectors (quick test)')
    ap.add_argument('--rcsr', help='directory with rcsr3d.cgd (default: $RCSR_DIR)')
    ap.add_argument('--out', default=str(HERE.parent / 'outputs' / 'independent_check'))
    args = ap.parse_args()
    d = args.rcsr or os.environ.get('RCSR_DIR')
    if not d: raise SystemExit('RCSR files needed: pass --rcsr DIR or set RCSR_DIR (see README).')
    cg = V.parse_cgd(str(Path(d) / 'rcsr3d.cgd'))
    todo = [tuple(x.split(':')) for x in args.pairs] if args.pairs else list(CASES)
    cases = [(s, f, CASES[(s, f)][0], args.nrand if args.nrand is not None else CASES[(s, f)][1]) for s, f in todo]
    dump(Path(args.out) / 'gain_sectors.json', run(cg, cases))


if __name__ == '__main__':
    main()
