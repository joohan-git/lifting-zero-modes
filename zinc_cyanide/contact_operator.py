"""Contact operator of a real filler on the framework mechanism space.

At one fixed geometry (relaxed filled structure), compute force constants of
  (F) the filled structure and (E) the same framework with the partner atoms deleted.
On the framework atoms, the partner-fixed dynamical matrix is the framework block D_FF of (F).
The contact perturbation is Delta(q) = D_FF^F(q) - D^E(q). Project it onto the mass-weighted
mechanism space Z(q) of the framework (central-force or rigid-unit bond set):
    C(q) = Z^dagger Delta Z,  B(q) = Z^dagger D^E Z  (intrinsic band of the mechanism sector).
The theory predicts that the spectrum of C(q) is q-independent (a rigid shift); B(q) carries
the framework's own dispersion. Full-material spectra are also reported for the filled crystal.
Units: eigenvalues in THz^2 (omega/2pi)^2.

Zn(CN)2 (two interpenetrated dia nets; each net is the partner of the other), as in the article:
  python zinc_cyanide/contact_operator.py --name ZnCN2_rigid --structure zinc_cyanide/ZnCN2_fcc.cif       --pairs Zn-C,Zn-N,C-N --rigid Zn --partner none --partner_net --sc 2,2,2 --out ZnCN2_rigid.json
  python zinc_cyanide/contact_operator.py --name ZnCN2_pin --structure zinc_cyanide/ZnCN2_fcc.cif       --pairs Zn-C,Zn-N,C-N --partner none --partner_net --sc 2,2,2 --out ZnCN2_pin.json
--rigid Zn: Zn(C/N)4 tetrahedra as rigid units; without it, pin-jointed bonds. Force field MACE-MP
(medium-mpa-0, float64, CPU); phonopy finite displacements 0.01 A; relaxation to fmax 1e-3 eV/A with
symmetry fixed. Results: zinc_cyanide/contact_rigid_units.json and contact_pin_joints.json.
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from ase.io import read
import mechanism_bands as rp


def dm_at(ph, q):
    ph.dynamical_matrix.run(q)
    return ph.dynamical_matrix.dynamical_matrix * ph.unit_conversion_factor ** 2


def stats(arr):
    arr = np.asarray(arr)  # (nq, nu) sorted eigenvalues
    m = arr.mean()
    q05, q95 = np.quantile(arr, 0.05), np.quantile(arr, 0.95)
    return dict(mean=float(m), min=float(arr.min()), max=float(arr.max()), q05=float(q05), q95=float(q95),
                abs_spread=float(q95 - q05), rel_spread=float((q95 - q05) / abs(m)) if m else None,
                branch_width=(arr.max(0) - arr.min(0)).tolist(), branch_mean=arr.mean(0).tolist())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--structure", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--pairs", required=True)
    p.add_argument("--partner", required=True, help="species of the partner (deleted in the empty run)")
    p.add_argument("--partner_idx", default="", help="alternatively, comma list of atom indices of the partner")
    p.add_argument("--partner_net", action="store_true", help="partner = bonded component(s) not containing atom 0")
    p.add_argument("--model", default="medium-mpa-0")
    p.add_argument("--sc", default="3,3,3")
    p.add_argument("--disp", type=float, default=0.01)
    p.add_argument("--fmax", type=float, default=1e-3)
    p.add_argument("--norelax", action="store_true")
    p.add_argument("--bondfactor", type=float, default=1.15)
    p.add_argument("--rigid", default="", help="centre species whose neighbours form rigid units")
    p.add_argument("--nq", type=int, default=200)
    p.add_argument("--seed", type=int, default=20261107)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    calc = rp.get_calc(a.model)
    full = read(a.structure)
    if not a.norelax:
        full = rp.relax(full, calc, a.fmax)
    sym = full.get_chemical_symbols()
    if a.partner_net:
        pr = []
        for x in a.pairs.split(","):
            ab, _, r = x.partition(":")
            A, B = ab.split("-")
            pr.append((A, B, float(r)) if r else (A, B))
        bb, _ = rp.framework_bonds(full, pr, a.bondfactor)
        parent = list(range(len(full)))
        def f(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]; u = parent[u]
            return u
        # components of the periodic bond graph, distinguishing images: label (atom, cell) is not
        # tracked here, so use the infinite-net test: atoms connected inside the cell plus through
        # translations that are lattice vectors of the cell belong to one net (cell must separate nets)
        for i, j, d in bb:
            parent[f(i)] = f(j)
        root = f(0)
        pidx = {i for i in range(len(full)) if f(i) != root}
    elif a.partner_idx:
        pidx = set(int(x) for x in a.partner_idx.split(","))
    else:
        ps = set(a.partner.split(","))
        pidx = {i for i, s in enumerate(sym) if s in ps}
    fidx = np.array([i for i in range(len(full)) if i not in pidx])
    empty = full[fidx]
    pairs = []
    for x in a.pairs.split(","):
        ab, _, r = x.partition(":")
        A, B = ab.split("-")
        pairs.append((A, B, float(r)) if r else (A, B))
    bonds, dmin = rp.framework_bonds(empty, pairs, a.bondfactor)
    if a.rigid:
        bonds = rp.rigid_unit_bonds(empty, bonds, set(a.rigid.split(",")))
    sc = [int(x) for x in a.sc.split(",")]
    phF = rp.force_constants(full, calc, sc, a.disp)
    phE = rp.force_constants(empty, calc, sc, a.disp)
    masses = empty.get_masses()
    frame = np.array(sorted({b[0] for b in bonds} | {b[1] for b in bonds}))
    cols = np.concatenate([[3 * i, 3 * i + 1, 3 * i + 2] for i in fidx])
    rng = np.random.default_rng(a.seed)
    qs = rng.random((a.nq, 3)) - 0.5
    # real contact tensors: S_i = -sum_p Phi_ip (all partner images) = -sum_p sqrt(m_i m_p) D_ip(Gamma)
    d0 = dm_at(phF, [0, 0, 0]).real
    mfull = full.get_masses()
    pl = sorted(pidx)
    Sreal = []
    for k, i in enumerate(fidx):
        Si = np.zeros((3, 3))
        for pp in pl:
            Si -= np.sqrt(mfull[i] * mfull[pp]) * d0[3 * i:3 * i + 3, 3 * pp:3 * pp + 3]
        Sreal.append(0.5 * (Si + Si.T))
    Sreal = np.array(Sreal)
    Smw = np.zeros((3 * len(fidx), 3 * len(fidx)))
    for k, i in enumerate(fidx):
        Smw[3 * k:3 * k + 3, 3 * k:3 * k + 3] = Sreal[k] / mfull[i]
    Creal = []
    C, B, Ffix, Ffull, wts, nus = [], [], [], [], [], []
    for q in qs:
        dF = dm_at(phF, q)
        dE = dm_at(phE, q)
        dFF = dF[np.ix_(cols, cols)]
        Z = rp.mech_basis(rp.compat(q, empty, bonds), masses, frame=frame)
        nus.append(Z.shape[1])
        if Z.shape[1] == 0:
            continue
        C.append(np.linalg.eigvalsh(Z.conj().T @ (dFF - dE) @ Z))
        Creal.append(np.linalg.eigvalsh(Z.conj().T @ Smw @ Z))
        B.append(np.linalg.eigvalsh(Z.conj().T @ dE @ Z))
        Ffix.append(np.linalg.eigvalsh(Z.conj().T @ dFF @ Z))
        # full filled crystal: modes with largest weight on the (embedded) mechanism space
        w2, ev = np.linalg.eigh(dF)
        Zfull = np.zeros((dF.shape[0], Z.shape[1]), complex)
        Zfull[cols] = Z
        wt = np.sum(np.abs(Zfull.conj().T @ ev) ** 2, axis=0)
        idx = np.argsort(-wt)[:Z.shape[1]]
        Ffull.append(np.sort(w2[idx]))
        wts.append(float(wt[idx].mean()))
    nu = max(set(nus), key=nus.count)
    keep = [k for k, n in enumerate([n for n in nus if n > 0]) if n == nu]
    sel = lambda L: np.array([L[k] for k in keep])
    out = dict(name=a.name, model=a.model, n_full=len(full), n_frame=len(empty), n_bonds=len(bonds),
               nu_generic=nu, nu_values=sorted(set(nus)), cell=full.cell.cellpar().tolist(),
               contact_operator=stats(sel(C)), contact_real_onsite=stats(sel(Creal)), intrinsic_band=stats(sel(B)),
               contact_tensors={str(int(i)): dict(sym=full.get_chemical_symbols()[i], S=Sreal[k].tolist(), eig=np.linalg.eigvalsh(Sreal[k]).tolist()) for k, i in enumerate(fidx)},
               partner_fixed_band=stats(sel(Ffix)), full_material_band=stats(sel(Ffull)),
               full_mech_weight=float(np.mean([wts[k] for k in keep])))
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: ({kk: v[kk] for kk in ('mean', 'abs_spread', 'rel_spread')} if isinstance(v, dict) and 'mean' in v else (v if not isinstance(v, dict) else '...')) for k, v in out.items()}, indent=1))


if __name__ == "__main__":
    main()
