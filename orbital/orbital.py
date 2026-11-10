"""Electronic crystal field: p orbitals on hcb, flu and the Lieb lattice (SM Section S9 and its orbital table).

Model. Each site carries p_x, p_y, p_z; nearest neighbours are joined by sigma hopping of unit magnitude
with on-site energies T_i = sum over bonds u u^T, so that H0(q) = R(q)^dagger R(q) (R the Bloch
compatibility matrix). The ligand field is lambda S, S block diagonal over sites. First-order energies
are lambda eig[Z^dagger S Z], Z an orthonormal basis of the zero band of H0; 'full' energies are the
lowest eigenvalues of H0 + lambda S. Every input below is a constant of this script and is written to
model.json.

  hcb (honeycomb, nearest-neighbour distance 1): A at (0,0), B at (1,0); Bravais rows AV; bond directions
      DELTAS with cell shifts SHIFTS. The p_z orbitals get the independent on-site energy 4 (PZ) and form two
      decoupled bands there. Fields: S_SCALAR = 1; S_U = diag(3/2, 3/2, 0) per site (the triangular dual star,
      route U); S_BROKEN = diag(2.1, 0.9, 0) per site (threefold axis broken); S_L = [[1.5, 0, .4], [0, 1.5, .2],
      [.4, .2, .8]] per site (tilted field: p_z is not an eigenvector of S_L; route L predicts 0.8 for both
      p_z bands near energy 4).
  flu (conventional cubic cell, a = 1): A = 8 tetrahedral sites, B = 4 fcc sites, 32 bonds of length sqrt(3)/4.
      S_C = 4/3 on A and 2 on B (the fcu dual star, route C, shift 14/9); S_CB = S_C + 0.4 diag(1, -1, 0) on every
      site. The finite-field energy of the C case is c_exact(lambda) with (t_A, t_B, s_A, s_B) = (4/3, 8/3, 4/3, 2).
  Lieb: hub A plus majority sites B, C with on-site potentials (1, 1) or (1.4, 0.6); M = (1/2, 1/2) is skipped.

Wavevectors (fractional reciprocal coordinates): hcb and Lieb on the 81 x 81 midpoint grid, flu on the 12^3
midpoint grid of the conventional reciprocal cell (only points with a fourfold zero band enter the widths;
the others are listed in flu_rank_jump_points.json). Path K-M-K' = (1/3,2/3)-(1/2,1/2)-(2/3,1/3), 121 points
per leg plus the end point. Branches are the sorted eigenvalues. The scaling uses lambda = 1e-5 ... 1e-1
(9 values) on the same path; slopes are fitted on lambda = 1e-4 ... 1e-2. Field strength lambda = 0.03; zero
modes are eigenvalues below 1e-9.

  python orbital/orbital.py [--out DIR]

Outputs (default outputs/orbital/): model.json, summary.json, scaling.csv, bands_path.csv, bands_bz.csv,
special_points.json, flu_rank_jump_points.json.
"""
import os
for _k in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_k] = '1'
import argparse
import csv
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.linalg import block_diag

ROOT = Path(__file__).resolve().parents[1]
LAM = .03
TOL = 1e-9

# honeycomb
AV = np.array([[1.5, np.sqrt(3) / 2], [1.5, -np.sqrt(3) / 2]])
DELTAS = np.array([[1., 0.], [-.5, -np.sqrt(3) / 2], [-.5, np.sqrt(3) / 2]])
SHIFTS = np.array([[0, 0], [-1, 0], [0, -1]])
PZ = np.diag([0, 0, 1, 0, 0, 1])
PZ_ONSITE = 4
S_SCALAR = np.eye(6)
S_U = block_diag(*([np.diag([1.5, 1.5, 0.])] * 2))
S_BROKEN = block_diag(*([np.diag([2.1, .9, 0.])] * 2))
S_L = block_diag(*([np.array([[1.5, 0, .4], [0, 1.5, .2], [.4, .2, .8]])] * 2))
L_VALUE = .8

# fluorite
F_A = np.array(list(itertools.product([.25, .75], repeat=3)))
F_B = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
F_POS = np.vstack([F_A, F_B])
F_EDGES = []
for i, a in enumerate(F_A):
    for j, b in enumerate(F_B):
        for sh in itertools.product([-1, 0, 1], repeat=3):
            d = b + sh - a
            if abs(np.dot(d, d) - 3 / 16) < 1e-12:
                F_EDGES.append((i, j + 8, np.array(sh), d / np.linalg.norm(d)))
assert len(F_EDGES) == 32
S_C = np.diag(np.repeat([4 / 3] * 8 + [2.] * 4, 3))
S_CB = S_C + block_diag(*([.4 * np.diag([1., -1., 0.])] * 12))
P_A = np.diag([1.] * 24 + [0.] * 12)

# Lieb on-site potentials of the majority sites
LIEB_CASES = [('scalar', 1., 1.), ('unequal', 1.4, .6)]

# wavevectors
QH = np.array(list(itertools.product((np.arange(81) + .5) / 81, repeat=2)))
QC = np.array(list(itertools.product((np.arange(12) + .5) / 12, repeat=3)))
PATH_NODES = np.array([[1 / 3, 2 / 3], [.5, .5], [2 / 3, 1 / 3]])
PATH = np.concatenate([np.linspace(PATH_NODES[i], PATH_NODES[i + 1], 121, endpoint=False) for i in range(2)] + [PATH_NODES[-1:]])
SCALING_LAMBDA = np.logspace(-5, -1, 9)
WITNESS_Q = [('hcb', [1 / 3, 2 / 3]), ('hcb', [.5, .5]), ('flu', [.173, .287, .419]), ('flu', [.211, .137, .357])]


def conv(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    raise TypeError(type(x).__name__)


def savecsv(out, name, rows):
    with (out / name).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def dump(out, name, obj):
    (out / name).write_text(json.dumps(obj, indent=2, default=conv), encoding='utf-8')


def honey(q):
    r = np.zeros((3, 6), complex)
    r[:, :2] = -DELTAS
    r[:, 3:5] = np.exp(2j * np.pi * (SHIFTS @ q))[:, None] * DELTAS
    return r.conj().T @ r + PZ_ONSITE * PZ, r


def zero_basis(h):
    e, v = np.linalg.eigh(h)
    return v[:, abs(e) < TOL], e


def calc(h, s, lam, n=None):
    z, e = zero_basis(h)
    p = np.linalg.eigvalsh(z.conj().T @ s @ z)
    full = np.linalg.eigvalsh(h + lam * s)
    return p, full[:len(p) if n is None else n], z, e


def fluorite(q):
    r = np.zeros((32, 36), complex)
    for row, (i, j, sh, d) in enumerate(F_EDGES):
        r[row, 3 * i:3 * i + 3] = -d
        r[row, 3 * j:3 * j + 3] = np.exp(2j * np.pi * np.dot(sh, q)) * d
    return r.conj().T @ r, r


def c_exact(lam):
    ta, tb, sa, sb = 4 / 3, 8 / 3, 4 / 3, 2.
    return ((ta + tb) + lam * (sa + sb) - np.sqrt((ta - tb + lam * (sa - sb)) ** 2 + 4 * ta * tb)) / 2


def lieb(q, vb=1., vc=1.):
    x, y = 1 + np.exp(2j * np.pi * q)
    h = np.array([[0, x, y], [x.conjugate(), 0, 0], [y.conjugate(), 0, 0]], complex)
    s = np.diag([0., vb, vc])
    z, _ = zero_basis(h)
    p = np.linalg.eigvalsh(z.conj().T @ s @ z)
    f = np.linalg.eigvalsh(h + LAM * s)
    return p, f[1]


def run(out):
    rows, summary, validations = [], {}, {}
    hstats = {k: {'p': [], 'full': [], 'error': []} for k in ['scalar', 'U', 'U_broken']}
    liebstats = {k: {'p': [], 'full': []} for k in ['scalar', 'unequal']}
    max_kernel_res = max_norm_h = 0.
    for q in QH:
        h, r = honey(q)
        z, _ = zero_basis(h)
        assert z.shape[1] == 1
        max_kernel_res = max(max_kernel_res, np.linalg.norm(r @ z))
        max_norm_h = max(max_norm_h, np.linalg.norm(z[:3].conj().T @ z[:3] - .5 * np.eye(1)))
        for name, s in [('scalar', S_SCALAR), ('U', S_U), ('U_broken', S_BROKEN)]:
            p, f, _, _ = calc(h, s, LAM)
            hs = hstats[name]
            hs['p'].append(p[0]); hs['full'].append(f[0]); hs['error'].append(abs(f[0] - LAM * p[0]))
            rows.append(dict(platform='hcb', case=name, q1=q[0], q2=q[1], q3=0, branch=0, first_order=p[0], full_energy=f[0], lambda_=LAM))
        for name, vb, vc in LIEB_CASES:
            p, f = lieb(q, vb, vc)
            if len(p) != 1: continue          # threefold zero at M
            liebstats[name]['p'].append(p[0]); liebstats[name]['full'].append(f)
            rows.append(dict(platform='Lieb', case=name, q1=q[0], q2=q[1], q3=0, branch=0, first_order=p[0], full_energy=f, lambda_=LAM))
    for k, st in hstats.items():
        summary['hcb_' + k] = dict(nq=len(QH), first_range=[min(st['p']), max(st['p'])], first_width=np.ptp(st['p']), full_width=np.ptp(st['full']), max_error=max(st['error']))
    for k, st in liebstats.items():
        summary['Lieb_' + k] = dict(nq=len(st['p']), first_range=[min(st['p']), max(st['p'])], first_width=np.ptp(st['p']), full_width=np.ptp(st['full']))
    cstats = {k: {'p': [], 'f': [], 'err': []} for k in ['C', 'C_broken']}
    rank_jumps = []
    norm_res = 0.
    for q in QC:
        h, r = fluorite(q)
        z, _ = zero_basis(h)
        if z.shape[1] != 4:                  # extra zero modes: listed separately, not in the widths
            rank_jumps.append(dict(q=q, kernel_dimension=z.shape[1], C_projected=calc(h, S_C, LAM)[0], C_full=calc(h, S_C, LAM)[1],
                                   broken_projected=calc(h, S_CB, LAM)[0], broken_full=calc(h, S_CB, LAM)[1]))
            continue
        max_kernel_res = max(max_kernel_res, np.linalg.norm(r @ z))
        norm_res = max(norm_res, np.linalg.norm(z.conj().T @ P_A @ z - (2 / 3) * np.eye(4), 2))
        for name, s in [('C', S_C), ('C_broken', S_CB)]:
            p, f, _, _ = calc(h, s, LAM)
            st = cstats[name]
            st['p'].append(p); st['f'].append(f); st['err'].append(np.max(abs(f - LAM * p)))
            for b in range(4):
                rows.append(dict(platform='flu', case=name, q1=q[0], q2=q[1], q3=q[2], branch=b, first_order=p[b], full_energy=f[b], lambda_=LAM))
    for k, st in cstats.items():
        p, f = np.array(st['p']), np.array(st['f'])
        summary['flu_' + k] = dict(nq=len(p), first_range=[p.min(), p.max()], first_branch_widths=np.ptp(p, axis=0), full_branch_widths=np.ptp(f, axis=0), max_error=max(st['err']))
    validations.update(kernel_residual=max_kernel_res, hcb_norm_half_residual=max_norm_h, flu_norm_two_thirds_residual=norm_res,
                       flu_exact_finite_formula_residual=np.max(abs(np.array(cstats['C']['f']) - c_exact(LAM))))

    # gapped path K-M-K' (away from Gamma)
    pathrows = []
    for iq, q in enumerate(PATH):
        h, _ = honey(q)
        for name, s in [('scalar', S_SCALAR), ('U', S_U), ('U_broken', S_BROKEN)]:
            p, f, _, _ = calc(h, s, LAM)
            pathrows.append(dict(index=iq, q1=q[0], q2=q[1], case=name, p=p[0], full=f[0] / LAM))
        ev = np.linalg.eigvalsh(h + LAM * S_L)[-2:]
        for b in range(2):
            pathrows.append(dict(index=iq, q1=q[0], q2=q[1], case='L_mix_' + str(b), p=L_VALUE, full=(ev[b] - PZ_ONSITE) / LAM))
    scale = []
    for lam in SCALING_LAMBDA:
        es, ls = [], []
        for q in PATH:
            h, _ = honey(q)
            p, f, _, _ = calc(h, S_BROKEN, lam)
            es.append(abs(f[0] - lam * p[0]))
            ls.append(np.linalg.eigvalsh(h + lam * S_L)[-2:])
        ls = np.array(ls)
        scale.append(dict(lambda_=lam, U_broken_max_error=max(es), L_mix_max_branch_width=max(np.ptp(ls, axis=0)), C_exact_error=abs(c_exact(lam) - lam * 14 / 9)))
    fit = {}
    for key in ['U_broken_max_error', 'L_mix_max_branch_width', 'C_exact_error']:
        fit[key] = np.polyfit(np.log([x['lambda_'] for x in scale[2:7]]), np.log([x[key] for x in scale[2:7]]), 1)[0]
    summary['scaling_slopes'] = fit
    summary['flu_rank_jump_sample_count'] = len(rank_jumps)
    summary['L'] = dict(first_eigenvalues=[L_VALUE, L_VALUE],
                        full_width_at_lambda_0p03=max(np.ptp(np.array([np.linalg.eigvalsh(honey(q)[0] + LAM * S_L)[-2:] for q in QH[::8]]), axis=0)),
                        normal_eigenvector=False)
    special = {}
    for platform, q in [('hcb_Gamma', [0., 0.]), ('flu_Gamma', [0., 0., 0.])]:
        h, r = honey(q) if platform.startswith('hcb') else fluorite(q)
        s = S_U if platform.startswith('hcb') else S_C
        p, f, z, e = calc(h, s, LAM)
        special[platform] = dict(kernel_dimension=z.shape[1], projected=p, full_low=f)
        if platform.startswith('hcb'):
            special['hcb_Gamma_broken'] = dict(projected=calc(h, S_BROKEN, LAM)[0], full_low=calc(h, S_BROKEN, LAM)[1])
    # dual-star tensors from the ligand directions, independently of the spectra
    ligand_h = np.column_stack([np.cos(np.pi + np.arange(3) * 2 * np.pi / 3), np.sin(np.pi + np.arange(3) * 2 * np.pi / 3), np.zeros(3)])
    ligand_a = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / np.sqrt(3)
    ligand_b = np.vstack([np.eye(3), -np.eye(3)])
    validations['dual_star_tensor_residuals'] = [np.linalg.norm(ligand_h.T @ ligand_h - np.diag([1.5, 1.5, 0])), np.linalg.norm(ligand_a.T @ ligand_a - 4 / 3 * np.eye(3)),
                                                 np.linalg.norm(ligand_b.T @ ligand_b - 2 * np.eye(3))]
    dual_fcu = (F_B + .5) % 1
    dual_res = []
    for i, pos in enumerate(F_POS):
        disp = np.array([d + sh - pos for d in dual_fcu for sh in itertools.product([-1, 0, 1], repeat=3)])
        dist = np.linalg.norm(disp, axis=1)
        near = disp[abs(dist - dist.min()) < 1e-10] / dist.min()
        assert len(near) == (4 if i < 8 else 6)
        dual_res.append(np.linalg.norm(near.T @ near - S_C[3 * i:3 * i + 3, 3 * i:3 * i + 3]))
    validations['actual_fcu_dual_nearest_star_residual'] = max(dual_res)
    # the honeycomb Hamiltonian written directly as on-site plus hopping terms
    direct_res = []
    for q in PATH[::11]:
        h, r = honey(q)
        direct = np.diag([1.5, 1.5, PZ_ONSITE, 1.5, 1.5, PZ_ONSITE]).astype(complex)
        for d, sh in zip(DELTAS, SHIFTS):
            term = -np.exp(2j * np.pi * np.dot(sh, q)) * np.outer(d, d)
            direct[:2, 3:5] += term; direct[3:5, :2] += term.conj().T
        direct_res.append(np.linalg.norm(direct - h))
    validations['independent_direct_hopping_residual'] = max(direct_res)
    # gapped wavevectors at lambda = 0.001 (no uniform error bound is claimed at band touchings)
    witnesses = []
    for name, q in WITNESS_Q:
        h, _ = honey(q) if name == 'hcb' else fluorite(q)
        s = S_BROKEN if name == 'hcb' else S_CB
        p, f, z, e = calc(h, s, .001)
        witnesses.append(dict(platform=name, q=q, kernel_dimension=z.shape[1], gap=min(e[e > TOL]), lambda_=.001, projected=p, full=f, max_error=max(abs(f - .001 * p))))
    summary['gapped_witnesses'] = witnesses
    assert max_kernel_res < 1e-10 and norm_res < 1e-10
    assert summary['hcb_U']['first_width'] < 1e-10
    assert summary['hcb_U_broken']['first_width'] > .5
    assert max(summary['flu_C']['first_branch_widths']) < 1e-10
    assert max(summary['flu_C_broken']['first_branch_widths']) > .1
    out.mkdir(parents=True, exist_ok=True)
    savecsv(out, 'bands_bz.csv', rows); savecsv(out, 'bands_path.csv', pathrows); savecsv(out, 'scaling.csv', scale)
    dump(out, 'special_points.json', special)
    dump(out, 'flu_rank_jump_points.json', rank_jumps)
    dump(out, 'model.json', dict(honey_bravais=AV, honey_bond_directions=DELTAS, honey_shifts=SHIFTS, honey_onsite=[1.5, 1.5, PZ_ONSITE], honey_S_U=S_U,
                                 honey_S_broken=S_BROKEN, honey_S_L=S_L, flu_positions=F_POS,
                                 flu_edges=[dict(i=i, j=j, shift=sh, direction=d) for i, j, sh, d in F_EDGES], flu_S_C=S_C, flu_S_broken=S_CB, lambda_=LAM))
    summary.update(lambda_=LAM, validation=validations)
    dump(out, 'summary.json', summary)
    print('hcb route U first-order range:', summary['hcb_U']['first_range'], '| flu route C range:', summary['flu_C']['first_range'])
    print('written to', out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=str(ROOT / 'outputs' / 'orbital'))
    run(Path(ap.parse_args().out))


if __name__ == '__main__':
    main()
