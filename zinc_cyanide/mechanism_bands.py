"""Framework-mechanism branches in real crystals with an MLIP and phonopy.

For a structure and a set of framework species pairs, this script:
1. relaxes the structure with symmetry fixed (optional);
2. computes harmonic force constants by finite displacements (phonopy);
3. builds the central-force compatibility matrix R(q) of the framework bonds only,
   with the same position-dependent phase convention as the phonopy dynamical matrix;
4. projects every phonon eigenvector onto the mass-weighted mechanism space ker R(q);
5. assigns the nu modes with the largest mechanism weight at each q and reports
   their omega^2 statistics over random and path wavevectors.
"""
import argparse, json, sys
import numpy as np
from ase.io import read
from ase.constraints import FixSymmetry
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

THZ = 1.0  # phonopy frequencies are in THz


def get_calc(model):
    from mace.calculators import mace_mp
    return mace_mp(model=model, default_dtype="float64", device="cpu")


def relax(atoms, calc, fmax, cell=True):
    atoms = atoms.copy()
    atoms.calc = calc
    atoms.set_constraint(FixSymmetry(atoms))
    obj = FrechetCellFilter(atoms) if cell else atoms
    BFGS(obj, logfile=None).run(fmax=fmax, steps=500)
    atoms.set_constraint()
    return atoms


def to_ph(atoms):
    return PhonopyAtoms(symbols=atoms.get_chemical_symbols(), cell=atoms.cell[:],
                        scaled_positions=atoms.get_scaled_positions())


def force_constants(atoms, calc, sc, disp):
    ph = Phonopy(to_ph(atoms), supercell_matrix=np.diag(sc), primitive_matrix="P")
    ph.generate_displacements(distance=disp)
    forces = []
    for s in ph.supercells_with_displacements:
        from ase import Atoms
        a = Atoms(s.symbols, cell=s.cell, scaled_positions=s.scaled_positions, pbc=True)
        a.calc = calc
        forces.append(a.get_forces())
    ph.forces = np.array(forces)
    ph.produce_force_constants()
    ph.symmetrize_force_constants()
    return ph


def framework_bonds(atoms, pairs, factor):
    """Bonds between listed species pairs, within factor x the shortest such distance."""
    from ase.neighborlist import neighbor_list
    sym = atoms.get_chemical_symbols()
    I, J, D, S = neighbor_list("ijDS", atoms, 4.0)
    # pairs: list of (A, B) or (A, B, rmax); without rmax use factor x the shortest A-B distance
    want = {}
    for p in pairs:
        key = tuple(sorted(p[:2]))
        want[key] = float(p[2]) if len(p) > 2 else None
    sel = [k for k in range(len(I)) if tuple(sorted((sym[I[k]], sym[J[k]]))) in want]
    dmin = min(np.linalg.norm(D[k]) for k in sel)
    pmin = {}
    for k in sel:
        key = tuple(sorted((sym[I[k]], sym[J[k]])))
        pmin[key] = min(pmin.get(key, 9e9), np.linalg.norm(D[k]))
    bonds = []
    for k in sel:
        key = tuple(sorted((sym[I[k]], sym[J[k]])))
        rmax = want[key] if want[key] is not None else factor * pmin[key]
        if np.linalg.norm(D[k]) > rmax:
            continue
        i, j, s = int(I[k]), int(J[k]), tuple(int(x) for x in S[k])
        # neighbor_list lists each bond twice (i->j and j->i); keep one direction
        if i < j or (i == j and s > (0, 0, 0)):
            bonds.append((i, j, D[k].copy()))
    return bonds, dmin


def rigid_unit_bonds(atoms, bonds, centers):
    """Add edges between all framework neighbours of each centre atom (rigid polyhedra)."""
    sym = atoms.get_chemical_symbols()
    nb = {}
    for i, j, d in bonds:
        nb.setdefault(i, []).append((j, np.asarray(d)))
        nb.setdefault(j, []).append((i, -np.asarray(d)))
    extra = []
    for c, lst in nb.items():
        if sym[c] not in centers:
            continue
        for a in range(len(lst)):
            for b in range(a + 1, len(lst)):
                (i, di), (j, dj) = lst[a], lst[b]
                extra.append((i, j, dj - di))
    return bonds + extra


def compat(q_frac, atoms, bonds):
    n = len(atoms)
    rec = np.linalg.inv(atoms.cell[:]).T  # rows: reciprocal vectors without 2pi
    qc = 2 * np.pi * q_frac @ rec
    R = np.zeros((len(bonds), 3 * n), complex)
    for b, (i, j, d) in enumerate(bonds):
        e = d / np.linalg.norm(d)
        R[b, 3 * i:3 * i + 3] -= e
        R[b, 3 * j:3 * j + 3] += e * np.exp(1j * qc @ d)
    return R


def mech_basis(R, masses, tol=1e-8, frame=None):
    """Orthonormal mass-weighted basis of ker R restricted to framework atoms
    (displacement u -> e = sqrt(m) u); non-framework atoms carry zero amplitude."""
    n = len(masses)
    if frame is None:
        frame = np.arange(n)
    cols = np.concatenate([[3 * i, 3 * i + 1, 3 * i + 2] for i in frame])
    Rf = R[:, cols]
    u, s, vh = np.linalg.svd(Rf)
    smax = s.max() if s.size else 1.0
    rank = int((s > tol * smax).sum())
    Kf = vh[rank:].conj().T
    K = np.zeros((3 * n, Kf.shape[1]), complex)
    K[cols] = Kf
    if K.shape[1] == 0:
        return K
    W = np.sqrt(np.repeat(masses, 3))[:, None] * K
    Q, _ = np.linalg.qr(W)
    return Q


def analyse(ph, atoms, bonds, qs, nu_expected=None):
    masses = atoms.get_masses()
    frame = np.array(sorted({b[0] for b in bonds} | {b[1] for b in bonds}))
    out = []
    for q in qs:
        ph.dynamical_matrix.run(q)
        dm = ph.dynamical_matrix.dynamical_matrix
        w2, ev = np.linalg.eigh(dm)
        # phonopy units: eigenvalues in (THz*2pi)^2-ish internal; convert via frequency factor
        freqs = np.sqrt(np.abs(w2)) * np.sign(w2) * ph.unit_conversion_factor
        Q = mech_basis(compat(q, atoms, bonds), masses, frame=frame)
        nu = Q.shape[1]
        wt = np.sum(np.abs(Q.conj().T @ ev) ** 2, axis=0) if nu else np.zeros(len(w2))
        idx = np.argsort(-wt)[:nu]
        out.append(dict(q=list(map(float, q)), nu=nu, freqs=freqs.tolist(), weights=wt.tolist(),
                        mech_idx=sorted(int(k) for k in idx)))
    return out


def summarize(res):
    nus = sorted({r["nu"] for r in res})
    nu = max(set(r["nu"] for r in res), key=[r["nu"] for r in res].count)
    gen = [r for r in res if r["nu"] == nu]
    f2 = np.array([sorted(np.sign(np.array(r["freqs"])[r["mech_idx"]]) * np.array(r["freqs"])[r["mech_idx"]] ** 2) for r in gen])
    wts = np.array([np.array(r["weights"])[r["mech_idx"]].mean() for r in gen])
    allf2 = f2.ravel()
    branch_width = (f2.max(0) - f2.min(0))
    return dict(nu_generic=nu, nu_values=nus, n_q=len(gen),
                f2_mean=float(allf2.mean()), f2_min=float(allf2.min()), f2_max=float(allf2.max()),
                f2_q05=float(np.quantile(allf2, 0.05)), f2_q95=float(np.quantile(allf2, 0.95)),
                rel_spread_q05_q95=float((np.quantile(allf2, 0.95) - np.quantile(allf2, 0.05)) / abs(allf2.mean())),
                branch_width_over_mean=(branch_width / abs(allf2.mean())).tolist(),
                mean_mech_weight=float(wts.mean()), min_mech_weight=float(wts.min()))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--structure", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--pairs", required=True, help="framework pairs, e.g. Sc-F,Mg-F")
    p.add_argument("--model", default="medium-mpa-0")
    p.add_argument("--sc", default="3,3,3")
    p.add_argument("--disp", type=float, default=0.01)
    p.add_argument("--fmax", type=float, default=1e-3)
    p.add_argument("--norelax", action="store_true")
    p.add_argument("--remove", default="", help="species to delete after relaxation (virtual empty framework)")
    p.add_argument("--bondfactor", type=float, default=1.15)
    p.add_argument("--nq", type=int, default=300)
    p.add_argument("--seed", type=int, default=20261107)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    calc = get_calc(a.model)
    atoms = read(a.structure)
    if not a.norelax:
        atoms = relax(atoms, calc, a.fmax)
    if a.remove:
        rm = set(a.remove.split(","))
        atoms = atoms[[s not in rm for s in atoms.get_chemical_symbols()]]
    pairs = []
    for x in a.pairs.split(","):
        ab, _, r = x.partition(":")
        A, B = ab.split("-")
        pairs.append((A, B, float(r)) if r else (A, B))
    bonds, dmin = framework_bonds(atoms, pairs, a.bondfactor)
    ph = force_constants(atoms, calc, [int(x) for x in a.sc.split(",")], a.disp)
    rng = np.random.default_rng(a.seed)
    qs = rng.random((a.nq, 3)) - 0.5
    res = analyse(ph, atoms, bonds, qs)
    summ = summarize(res)
    # Gamma check for imaginary modes
    ph.dynamical_matrix.run([0, 0, 0])
    summ.update(name=a.name, model=a.model, n_atoms=len(atoms), cell=atoms.cell.cellpar().tolist(),
                n_bonds=len(bonds), bond_min=float(dmin), dof=3 * len(atoms),
                maxwell_nu=3 * len(atoms) - len(bonds),
                symbols=atoms.get_chemical_symbols())
    json.dump(dict(summary=summ, per_q=res), open(a.out, "w"))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
