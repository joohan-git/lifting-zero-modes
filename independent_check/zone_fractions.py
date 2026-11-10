"""Zone fractions at finite stiffness ratio, repeated with the second implementation (SM Section S3.5).

volume: full composite D(q) = diag(R_F^H R_F, R_S^H R_S) + k_c C^H C in the primitive cell, eight scrambled
Sobol blocks of 1,024 wavevectors, k_c/k_s = 1e-4, 1e-3, 1e-2. Two criteria per wavevector: (A) the full
eigenstates with the largest weight on each first-order reference subspace deviate by more than delta;
(B) fewer eigenvalues than the reference multiplicity lie within the delta window (no assignment).
soft-modes: for qtz-x (x) uom and bcu (x) nbo at k_c = 1e-3 on 4,096 Sobol wavevectors, the 1% violation
with the scaffold free and held fixed, and the lowest scaffold and non-mechanism floppy eigenvalues in
units of the contact scale k_c c.

  python independent_check/zone_fractions.py volume --rcsr DIR [--pairs bcu:nbo] [--nq-block 64] [--eps 1e-3]
  python independent_check/zone_fractions.py soft-modes --rcsr DIR [--nq-block 64]
"""
import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
from scipy.stats import qmc
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import engine as V
import engine_primitive as V2
from common.composite import dump

PAIRS = [('bcu', 'nbo'), ('qtz-x', 'uom'), ('acs', 'gra')]
EPS = [1e-4, 1e-3, 1e-2]; DELTAS = [0.01, 0.05]
NBLK, NPB = 8, 1024

def analyse(pre, q, kc, refs):
    sc, fl, Pl, S, unitF, unitS = pre['sc'], pre['fl'], pre['P'], pre['S'], pre['unitF'], pre['unitS']
    RF = V.rigidity(fl, q, unitF); RS = V.rigidity(sc, q, unitS); nF = 3*fl['V']; N = nF + 3*sc['V']
    D = np.zeros((N, N), complex); D[:nF, :nF] = RF.conj().T @ RF; D[nF:, nF:] = RS.conj().T @ RS
    C = V.contact_matrix(fl, sc, Pl['contacts'], q); D += kc*(C.conj().T @ C)
    lam, vec = np.linalg.eigh(D)
    Z = V.kernel(RF); mu, W = np.linalg.eigh(Z.conj().T @ S @ Z)
    out = {}
    taken = np.zeros(N, bool)
    for c, Mc in refs:
        sel = np.abs(mu - c) < 1e-6*max(c, 1e-12)
        B = Z @ W[:, sel]                      # orthonormal basis of the reference subspace (F block)
        wgt = np.sum(np.abs(B.conj().T @ vec[:nF, :])**2, axis=0)   # ||Pi_c v_k||^2
        wgt_m = np.where(taken, -1, wgt)
        idx = np.argsort(-wgt_m)[:Mc]; taken[idx] = True
        dev = lam[idx]/(kc*c) - 1
        rel_all = np.abs(lam/(kc*c) - 1)
        out[c] = dict(dim_ref=int(sel.sum()), Mc=Mc, max_dev=float(np.abs(dev).max()), min_weight=float(wgt[idx].min()),
                      n_in={d: int(np.sum(rel_all <= d)) for d in DELTAS}, n_in_1e3=int(np.sum(rel_all <= 1e-3)))
    return out, int(Z.shape[1])

def plane_dist(q):
    """Fractional distance to the hexagonal zero-set planes found by the scans for qtz-x (coordinate planes, q_z=1/2, q_x=-q_y)."""
    w = lambda x: np.abs((x + 0.5) % 1 - 0.5)
    return min(w(q[0]), w(q[1]), w(q[2]), w(q[2] - 0.5), w(q[0] + q[1]))

def volume(cg, pairs, nq_block=NPB, eps=EPS):
    res = {}
    for s, f in pairs:
        pre = V2.prepare(cg, s, f)
        # reference values and multiplicities from the fixed-scaffold generic spectrum
        sp = V2.spectra(pre, nq=8)
        vals = np.array(sp['bands']); refs = []
        for c in sorted(set(np.round(vals, 6))):
            refs.append((float(np.mean(vals[np.abs(vals - c) < 1e-5])), int(np.sum(np.abs(vals - c) < 1e-5))))
        rec = dict(mult=pre['sc']['mult'], V_F=pre['fl']['V'], V_S=pre['sc']['V'], M_F=pre['MF'], M_S=sp['mech_S'], refs=refs, n_contacts=sp['n_contacts'], eps={})
        print(s, f, 'primitive V_F', pre['fl']['V'], 'V_S', pre['sc']['V'], 'M_F', pre['MF'], 'refs', refs, flush=True)
        for kc in eps:
            blocks = []; rows = []
            for b in range(NBLK):
                qs = qmc.Sobol(d=3, scramble=True, seed=1000 + b).random(nq_block)
                vA = {d: 0 for d in DELTAS}; vB = {d: 0 for d in DELTAS}; vA_notB = {d: 0 for d in DELTAS}
                for q in qs:
                    o, nk = analyse(pre, q, kc, refs)
                    for d in DELTAS:
                        a = any(x['max_dev'] > d for x in o.values()); bb = any(x['n_in'][d] < x['Mc'] for x in o.values())
                        vA[d] += a; vB[d] += bb; vA_notB[d] += (a and not bb)
                    if kc == 1e-3:
                        rows.append([float(plane_dist(q)), max(x['max_dev'] for x in o.values()), min(x['min_weight'] for x in o.values()),
                                     min(x['n_in'][0.01] - x['Mc'] for x in o.values()), min(x['n_in'][0.05] - x['Mc'] for x in o.values()), nk])
                blocks.append(dict(A={d: vA[d]/nq_block for d in DELTAS}, B={d: vB[d]/nq_block for d in DELTAS}, A_notB={d: vA_notB[d]/nq_block for d in DELTAS}))
            summ = {}
            for key in ('A', 'B', 'A_notB'):
                for d in DELTAS:
                    x = np.array([bl[key][d] for bl in blocks]); summ[f'{key}_{d}'] = dict(mean=float(x.mean()), se=float(x.std(ddof=1)/np.sqrt(NBLK)))
            rec['eps'][str(kc)] = summ
            print(f'  eps={kc}:', {k: f"{v['mean']*100:.2f}+-{v['se']*100:.2f}%" for k, v in summ.items()}, flush=True)
            if kc == 1e-3:
                R = np.array(rows)
                # violation rate (criterion A, 1%) vs distance to the zero-set planes of the scans (bins) and weight statistics
                bins = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
                prof = []
                for lo, hi in zip(bins[:-1], bins[1:]):
                    m = (R[:, 0] >= lo) & (R[:, 0] < hi)
                    if m.sum(): prof.append(dict(bin=[lo, hi], n=int(m.sum()), viol_A_1pct=float(np.mean(R[m, 1] > 0.01)), viol_B_1pct=float(np.mean(R[m, 3] < 0)),
                                                 median_maxdev=float(np.median(R[m, 1])), median_minweight=float(np.median(R[m, 2]))))
                rec['profile_eps1e-3'] = prof
                rec['weight_stats_eps1e-3'] = dict(min_weight_p05=float(np.percentile(R[:, 2], 5)), min_weight_median=float(np.median(R[:, 2])),
                                                   maxdev_p50=float(np.median(R[:, 1])), maxdev_p90=float(np.percentile(R[:, 1], 90)),
                                                   maxdev_p99=float(np.percentile(R[:, 1], 99)), kernel_dims=sorted(set(R[:, 5].astype(int).tolist())))
                print('   profile:', prof, flush=True); print('   weights:', rec['weight_stats_eps1e-3'], flush=True)
        res[f'{s}|{f}'] = rec
    return res


def soft_modes(cg, nq_block=512):
    kc = 1e-3; res = {}
    for s, f, c, Mc in (('qtz-x', 'uom', 1.4705863431008748, 6), ('bcu', 'nbo', 2.0, 3)):
        pre = V2.prepare(cg, s, f); sc, fl, Pl, S, unitF, unitS = pre['sc'], pre['fl'], pre['P'], pre['S'], pre['unitF'], pre['unitS']
        qs = np.vstack([qmc.Sobol(d=3, scramble=True, seed=3000 + b).random(nq_block) for b in range(8)])
        full = []; frozen = []; soft = []; softF = []
        for q in qs:
            RF = V.rigidity(fl, q, unitF); RS = V.rigidity(sc, q, unitS); nF = 3*fl['V']; N = nF + 3*sc['V']
            D = np.zeros((N, N), complex); D[:nF, :nF] = RF.conj().T @ RF; D[nF:, nF:] = RS.conj().T @ RS
            C = V.contact_matrix(fl, sc, Pl['contacts'], q); D += kc*(C.conj().T @ C)
            lam = np.linalg.eigvalsh(D); full.append(int(np.sum(np.abs(lam/(kc*c) - 1) <= 0.01)) < Mc)
            lamF = np.linalg.eigvalsh(RF.conj().T @ RF + kc*S); frozen.append(int(np.sum(np.abs(lamF/(kc*c) - 1) <= 0.01)) < Mc)
            wS = np.linalg.eigvalsh(RS.conj().T @ RS); soft.append(float(wS[0]/(kc*c)))           # lowest scaffold eigenvalue (0 only at zero set)
            wF = np.linalg.eigvalsh(RF.conj().T @ RF); softF.append(float(np.sort(wF)[Mc]/(kc*c)))  # lowest non-mechanism F eigenvalue
        full = np.array(full); frozen = np.array(frozen); soft = np.array(soft); softF = np.array(softF)
        r = dict(viol_full_1pct=float(full.mean()), viol_frozen_scaffold_1pct=float(frozen.mean()),
                 frac_scaffold_lowest_below_10x_ref=float(np.mean(soft < 10)), frac_scaffold_lowest_below_100x_ref=float(np.mean(soft < 100)),
                 frac_F_stiff_lowest_below_10x_ref=float(np.mean(softF < 10)), frac_F_stiff_lowest_below_100x_ref=float(np.mean(softF < 100)),
                 viol_full_given_scaffold_soft100=float(full[soft < 100].mean()) if (soft < 100).any() else None,
                 viol_full_given_scaffold_stiff100=float(full[soft >= 100].mean()) if (soft >= 100).any() else None,
                 median_scaffold_lowest_over_ref=float(np.median(soft)), median_F_stiff_lowest_over_ref=float(np.median(softF)))
        print(s, f, r, flush=True); res[f'{s}|{f}'] = r
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['volume', 'soft-modes'])
    ap.add_argument('--pairs', nargs='*', help='scaffold:floppy (default: bcu:nbo qtz-x:uom acs:gra)')
    ap.add_argument('--nq-block', type=int, help='wavevectors per Sobol block (default 1024 / 512)')
    ap.add_argument('--eps', nargs='*', type=float, help='k_c/k_s values (default 1e-4 1e-3 1e-2)')
    ap.add_argument('--rcsr', help='directory with rcsr3d.cgd (default: $RCSR_DIR)')
    ap.add_argument('--out', default=str(HERE.parent / 'outputs' / 'independent_check'))
    args = ap.parse_args()
    d = args.rcsr or os.environ.get('RCSR_DIR')
    if not d: raise SystemExit('RCSR files needed: pass --rcsr DIR or set RCSR_DIR (see README).')
    cg = V.parse_cgd(str(Path(d) / 'rcsr3d.cgd'))
    if args.mode == 'volume':
        pairs = [tuple(x.split(':')) for x in args.pairs] if args.pairs else PAIRS
        res = volume(cg, pairs, args.nq_block or NPB, args.eps or EPS); dump(Path(args.out) / 'zone_fractions.json', res)
    else:
        res = soft_modes(cg, args.nq_block or 512); dump(Path(args.out) / 'soft_modes.json', res)


if __name__ == '__main__':
    main()
