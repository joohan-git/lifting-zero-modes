"""pcu (scaffold, fixed) (x) urk (floppy): width of the lifted branches against k_c with D = D_F + k_c S (k_F = 1),
SM Sections S2.6, S2.7 and Table S16.

Uses model, matrices and spectrum of fixed_cage.py unchanged. Lifted branches: the m eigenvectors of D with the largest
weight on ker R_F(q) (m = kernel dimension). Width of a branch: max_q - min_q of its sorted eigenvalue over the
wavevectors with floppy-net gap >= 0.1. k_c = 0.1, 0.03, 0.01, 0.003; 40 wavevectors (seed 20261107) and a check
on 400 (seed 20261108).

  python fixed_cage/urk_fixed_cage.py --rcsr DIR [--out DIR]     (stored result: fixed_cage/urk_fixed_cage.json)
"""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import numpy as np
import scipy.linalg as la
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import nets as E

KC = [0.1, 0.03, 0.01, 0.003]


def load_fixed_cage():
    spec = importlib.util.spec_from_file_location('fixed_cage_module', HERE / 'fixed_cage.py')
    fc = importlib.util.module_from_spec(spec); spec.loader.exec_module(fc)
    return fc


def run(fc, m, seed, ndraw):
    data = []
    for q in np.random.default_rng(seed).random((ndraw, 3)):
        r, d, _ = fc.matrices(m, q); z, gap, _ = fc.spectrum(r); mm = z.shape[1]
        if mm == 0: continue
        row = dict(q=q, gap=gap, m=mm, ev={})
        for kc in KC:
            ev, U = la.eigh(d + kc * m['S']); w = np.linalg.norm(z.conj().T @ U, axis=0) ** 2
            idx = np.sort(np.argsort(-w)[:mm]); row['ev'][kc] = (ev[idx], float(np.sort(w)[::-1][:mm].min()))
        data.append(row)
    return data


def analyse(data, cutoff=0.1):
    good = [x for x in data if x['gap'] >= cutoff]
    out = dict(n_drawn=len(data), n_used=len(good), kernel_dims=sorted(set(x['m'] for x in good)),
               gap_min_used=min(x['gap'] for x in good),
               min_weight_on_mechanism_space={str(kc): min(x['ev'][kc][1] for x in good) for kc in KC}, per_kc={})
    mm = good[0]['m']; widths = {}
    for kc in KC:
        a = np.array([x['ev'][kc][0] for x in good]); w = np.ptp(a, 0); widths[kc] = w
        out['per_kc'][str(kc)] = dict(branch_mean_over_kc=(a.mean(0) / kc).tolist(), branch_width=w.tolist(),
                                      width_over_kc2=(w / kc ** 2).tolist(), max_width_over_kc2=float((w / kc ** 2).max()))
    kcs = np.array(KC); fit = {}
    for b in range(mm):
        wb = np.array([widths[k][b] for k in KC]); s, c = np.polyfit(np.log(kcs), np.log(wb), 1)
        fit[f'branch{b}'] = dict(slope=float(s), coeff_from_fit_exp_intercept=float(np.exp(c)))
    wmax = np.array([widths[k].max() for k in KC]); s, c = np.polyfit(np.log(kcs), np.log(wmax), 1)
    fit['max_branch'] = dict(slope=float(s), coeff_from_fit_exp_intercept=float(np.exp(c)))
    # coefficient with the slope fixed to 2: width/k_c^2 at the three smallest k_c, and least squares through the origin
    fit['coefficient_slope2_max_branch_smallest3_kc'] = [float(widths[k].max() / k ** 2) for k in KC[1:]]
    fit['coefficient_slope2_LS'] = float(np.sum(wmax * kcs ** 2) / np.sum(kcs ** 4))
    out['fit'] = fit
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=str(HERE.parent / 'outputs' / 'fixed_cage'))
    E.add_argument(ap); args = ap.parse_args(); E.apply_argument(args)
    fc = load_fixed_cage()
    m = fc.model(E.read_cgd(), 'pcu', 'urk', fc.site_records()[('pcu', 'urk')])
    main_ = analyse(run(fc, m, 20261107, 40)); big = analyse(run(fc, m, 20261108, 400))
    res = dict(pair='pcu (scaffold, fixed) x urk (floppy)', shift=m['shift'].tolist(),
               definition='ptp over q of lifted eigenvalue of D_F + k_c S (k_F=1), gap_F>=0.1; coefficient = width/k_c^2 (units 1/k_F)',
               kc_list=KC, main_40q_seed20261107=main_, check_400q_seed20261108=big)
    p = Path(args.out) / 'urk_fixed_cage.json'; p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1), encoding='utf8')
    for nm, r in (('40q', main_), ('400q', big)):
        print(nm, 'used', r['n_used'], 'max width/k_c^2 at k_c = 0.003:', round(r['per_kc']['0.003']['max_width_over_kc2'], 3), 'slope', round(r['fit']['max_branch']['slope'], 3))
    print('written', p)


if __name__ == '__main__':
    main()
