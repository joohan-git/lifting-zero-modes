"""Final design map of the ten exactly flat dual pairs in the beam frame (sim_core).

Joint designs: rigid (welded hubs), lock (contact rotation lock k_cr = 100), release (moment-free strut ends).
Thick cage k_s,axial = 10, k_s,bend = 1; floppy axial stiffness kf_a = 1 and 10; k_c = 1e-3 and 1e-2;
k_b / k_c = 0.01, 0.1, 0.3, 1. Metric: largest over sorted branches of (branch width / branch mean) on 48 random
wavevectors, reported for the raw draws (seed 20261107, the q set of sweep2) and for draws filtered by the floppy-net
gap g(q) = (M+1)-th eigenvalue of R_F^H R_F >= 0.05.

Second block (residual): for the three pairs whose locked bending is exactly q-independent (bcu/nbo, nia/nia-d,
pcu/urk) the residual width against kf_a (1 ... 100), k_b / k_c and the gap cutoff of the sample.
Run: python metamaterial/sim_design.py --rcsr DIR   (stored result: metamaterial/design_map.json)
"""
import json, sys
from multiprocessing import Pool
import numpy as np
import sim_core as S

KC = [1e-3, 1e-2]
RB = [0.01, 0.1, 0.3, 1.0]
KFA = [1.0, 10.0]
DES = ['rigid', 'lock', 'release']
KSA, KSB = 10.0, 1.0
NQ = 48
GMIN = 0.05
EXACT = [("bcu", "nbo"), ("nia", "nia-d"), ("pcu", "urk")]


def widths(m, qs, **kw):
    L = np.array([m.lifted(q, **kw)[0] / kw['kc'] for q in qs])
    r, bw, mean = S.rel_branch_width(L)
    return r, mean


def run_pair(pair):
    sc, fl = pair; m = S.Model(sc, fl)
    raw, filt, graw, gfilt, rej = m.qsets(NQ, S.SEED, GMIN)
    rows = []
    for qname, qs in (('filtered', filt), ('raw', raw)):
        for kfa in KFA:
            for kc in KC:
                for des in DES:
                    for rb in (RB if des != 'release' else [None]):
                        kb = (rb or 0.0) * kc
                        r, mean = widths(m, qs, kc=kc, kf_a=kfa, kf_b=kb, ks_a=KSA, ks_b=KSB, design=des, kcr=100.0)
                        rows.append(dict(q=qname, kf_a=kfa, kc=kc, design=des, kb_over_kc=rb, rel_width=r,
                                         branch_mean=mean.tolist()))
    out = dict(nf=m.nf, ns=m.ns, M=m.M, gap_raw=graw.tolist(), gap_filtered_min=float(gfilt.min()),
               rejected_draws=int(rej), rows=rows)
    print(sc, fl, flush=True)
    return f'{sc}|{fl}', out


def run_residual(pair):
    sc, fl = pair; m = S.Model(sc, fl)
    raw, filt, graw, gfilt, rej = m.qsets(NQ, S.SEED, GMIN)
    res = dict(kfa_scan=[], cutoff_scan=[], kb_scan=[])
    for kfa in [1, 2, 5, 10, 20, 50, 100]:
        for rb in [0.3, 1.0]:
            r, _ = widths(m, filt, kc=1e-3, kf_a=float(kfa), kf_b=rb * 1e-3, ks_a=KSA, ks_b=KSB, design='lock')
            res['kfa_scan'].append(dict(kf_a=kfa, kb_over_kc=rb, rel_width=r))
    for rb in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]:
        r, _ = widths(m, filt, kc=1e-3, kf_a=1.0, kf_b=rb * 1e-3, ks_a=KSA, ks_b=KSB, design='lock')
        r0, _ = widths(m, filt, kc=1e-3, kf_a=1.0, kf_b=rb * 1e-3, ks_a=KSA, ks_b=KSB, design='rigid')
        res['kb_scan'].append(dict(kb_over_kc=rb, lock=r, rigid=r0))
    # gap cutoff: 48 accepted draws for each cutoff from the same stream
    for g in [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]:
        rng = np.random.default_rng(S.SEED); qs = []
        while len(qs) < NQ:
            q = rng.random(3) - 0.5
            if g == 0.0 or m.gap(q) >= g:
                qs.append(q)
        r, _ = widths(m, qs, kc=1e-3, kf_a=1.0, kf_b=1e-3, ks_a=KSA, ks_b=KSB, design='lock')
        res['cutoff_scan'].append(dict(gmin=g, rel_width=r))
    print('residual', sc, fl, flush=True)
    return f'{sc}|{fl}', res


if __name__ == '__main__':
    with Pool(12) as P:
        design = dict(P.map(run_pair, S.PAIRS))
        residual = dict(P.map(run_residual, EXACT))
    out = dict(params=dict(KC=KC, RB=RB, KFA=KFA, designs=DES, ks_axial=KSA, ks_bend=KSB, nq=NQ, gap_min=GMIN,
                           kcr=100.0, seed=S.SEED, nu=0.3),
               pairs={f'{sc}|{fl}': design[f'{sc}|{fl}'] for sc, fl in S.PAIRS}, residual=residual)
    json.dump(out, open(S.out('design_map.json'), 'w'), indent=1)
    print('done')
