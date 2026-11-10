"""Distributed-mass check in physical units: every floppy strut is split into n Euler-Bernoulli elements with the
consistent mass matrix (translation, bending rotation coupling and torsional inertia), all six DOFs of every node are
kept, and the generalized problem K(q) x = omega^2 M(q) x is solved. The cage is held fixed, so the effect of strut
mass and hub inertia is isolated; cage mobility is covered by the design map and the finite specimen.

Joint designs at the floppy hubs:
  release  each strut end has its own rotation DOFs (moment-free), hubs carry translations only;
  lock     hub rotations are held by the contact pads (ideal rotational guide, theta_hub = 0);
  rigid    welded hubs with rotary inertia J = (2/5) m0 R^2 of a sphere of the node volume.
Geometry and material: conventional cell a = 20 mm, circular floppy struts of radius r, node
volume 83.33 mm^3 (polymer 100 mg), normal contact spring k_c = 100 N/m at E = 2 GPa, scaled with E for steel.
Run: python metamaterial/sim_mass.py --rcsr DIR   (stored result: metamaterial/mass.json)
"""
import json
from multiprocessing import Pool
import numpy as np
import scipy.linalg as la
import sim_core as S

MAT = dict(polymer=dict(E=2e9, rho=1200.0), steel=dict(E=210e9, rho=7850.0))
A_CELL = 0.02
VNODE = 1e-4 / 1200.0
NU = 0.3
PAIRS = [("bcu", "nbo"), ("nia", "nia-d"), ("fcu", "flu")]


def mass_block(e, L, m, Ip_rho):
    """12x12 consistent mass (u_a, th_a, u_b, th_b) in global axes; m = rho A L, Ip_rho = rho I_p L."""
    M = np.zeros((12, 12)); ua, tha, ub, thb = slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12)
    T = np.zeros((2, 12)); T[0, ua] = e; T[1, ub] = e
    M += T.T @ (m / 6 * np.array([[2, 1], [1, 2]])) @ T
    T = np.zeros((2, 12)); T[0, tha] = e; T[1, thb] = e
    M += T.T @ (Ip_rho / 6 * np.array([[2, 1], [1, 2]])) @ T
    loc = m / 420 * np.array([[156, 22 * L, 54, -13 * L], [22 * L, 4 * L * L, 13 * L, -3 * L * L],
                              [54, 13 * L, 156, -22 * L], [-13 * L, -3 * L * L, -22 * L, 4 * L * L]])
    for t in S.perp_basis(e):
        b = np.cross(e, t); T = np.zeros((4, 12))
        T[0, ua] = t; T[1, tha] = b; T[2, ub] = t; T[3, thb] = b
        M += T.T @ loc @ T
    return M


class FE:
    def __init__(self, sc, fl, r, mat, design, nel, rotscale=None):
        self.m = S.Model(sc, fl); m = self.m
        E, rho = MAT[mat]['E'], MAT[mat]['rho']
        scale = A_CELL / np.linalg.norm(m.fr.fl['A'][0])
        self.scale = scale; self.r = r; self.E = E; self.rho = rho; self.design = design
        self.m0 = rho * VNODE; self.kc = 100.0 * E / 2e9
        Rn = (3 * VNODE / (4 * np.pi)) ** (1 / 3); self.J = 0.4 * self.m0 * Rn ** 2
        Ar = np.pi * r * r; I = np.pi * r ** 4 / 4; G = E / (2 * (1 + NU))
        self.ls = rotscale or A_CELL           # rotation DOFs are stored as theta * ls (conditioning only)
        nf = m.nf; ndof = 0
        self.hubT = [list(range(3 * h, 3 * h + 3)) for h in range(nf)]; ndof = 3 * nf
        self.hubR = []
        for h in range(nf):
            if design == 'rigid':
                self.hubR.append(list(range(ndof, ndof + 3))); ndof += 3
            else:
                self.hubR.append(None)
        self.elems = []; self.internal = []          # (dofs12, phases12, K12, M12)
        lengths = []
        for b, (i, j, s) in enumerate(m.fr.fl['edges']):
            e = m.fr.fl['unit'][b]; L = m.fr.fl['length'][b] * scale; lengths.append(L); le = L / nel
            ka, kb, kt = E * Ar / le, 12 * E * I / le ** 3, G * 2 * I / le
            Kb = S.beam_block(e, le, ka, kb, kt); Mb = mass_block(e, le, rho * Ar * le, rho * 2 * I * le)
            # node list along the strut: hub i, internal 1..nel-1, hub j (image s)
            nodes = []
            for k in range(nel + 1):
                if k == 0 or k == nel:
                    h = i if k == 0 else j; ph = 1.0 if k == 0 else None
                    T = self.hubT[h]
                    if design == 'release':
                        Rr = list(range(ndof, ndof + 3)); ndof += 3; self.internal += Rr
                    elif design == 'lock':
                        Rr = [-1, -1, -1]
                    else:
                        Rr = self.hubR[h]
                    nodes.append((T, Rr, k == nel, k == nel and design != 'release'))
                else:
                    T = list(range(ndof, ndof + 3)); Rr = list(range(ndof + 3, ndof + 6)); ndof += 6
                    self.internal += T + Rr
                    nodes.append((T, Rr, False, False))
            for k in range(nel):
                (Ta, Ra, _, _), (Tb, Rb, phT, phR) = nodes[k], nodes[k + 1]
                dofs = Ta + Ra + Tb + Rb
                img = [np.zeros(3)] * 6 + [np.array(s) if phT else np.zeros(3)] * 3 + [np.array(s) if phR else np.zeros(3)] * 3
                self.elems.append((dofs, img, Kb, Mb))
        self.ndof = ndof; self.lengths = np.array(lengths)
        self.internal = np.array(sorted(set(self.internal)), int)
        rot = np.zeros(ndof, bool)
        for h in range(nf):
            if self.hubR[h] is not None:
                rot[self.hubR[h]] = True
        for dofs, img, Kb, Mb in self.elems:
            for a in (3, 4, 5, 9, 10, 11):
                if dofs[a] >= 0:
                    rot[dofs[a]] = True
        self.rot = rot

    def matrices(self, q):
        n = self.ndof; K = np.zeros((n, n), complex); M = np.zeros((n, n), complex)
        for dofs, img, Kb, Mb in self.elems:
            ph = np.array([np.exp(2j * np.pi * np.dot(q, s)) for s in img]); d = np.array(dofs); ok = d >= 0
            P = np.outer(np.conj(ph[ok]), ph[ok])
            K[np.ix_(d[ok], d[ok])] += Kb[np.ix_(ok, ok)] * P; M[np.ix_(d[ok], d[ok])] += Mb[np.ix_(ok, ok)] * P
        for h, (i, k, s, nvec, ell) in enumerate(self.m.p['contacts']):
            T = self.hubT[i]; K[np.ix_(T, T)] += self.kc * np.outer(nvec, nvec)
        for h in range(self.m.nf):
            M[np.ix_(self.hubT[h], self.hubT[h])] += self.m0 * np.eye(3)
            if self.hubR[h] is not None:
                M[np.ix_(self.hubR[h], self.hubR[h])] += self.J * np.eye(3)
        D = np.where(self.rot, 1.0 / self.ls, 1.0)
        K = D[:, None] * K * D[None, :]; M = D[:, None] * M * D[None, :]
        return 0.5 * (K + K.conj().T), 0.5 * (M + M.conj().T)

    def solve(self, q):
        K, M = self.matrices(q)
        w, v = la.eigh(K, M)
        f = np.sqrt(np.maximum(w, 0)) / (2 * np.pi)
        hub = np.concatenate(self.hubT)
        Z = self.m.Z(q)[:, :self.m.M]
        ke = np.real(np.einsum('ij,ik,kj->j', v.conj(), M, v))
        hub_ke = self.m0 * np.sum(np.abs(v[hub]) ** 2, axis=0) / ke
        ov = np.sum(np.abs(Z.conj().T @ v[hub]) ** 2, axis=0) * self.m0 / ke
        lift = np.sort(np.argsort(-ov)[:self.m.M])
        fb = f[lift]
        strut = [k for k in range(len(f)) if hub_ke[k] < 0.5 and f[k] > 1e-3 * fb.min()]
        return fb, ov[lift], (float(f[strut[0]]) if strut else None), f, hub_ke


def analytic(fe):
    I = np.pi * fe.r ** 4 / 4; Ar = np.pi * fe.r ** 2; c = np.sqrt(fe.E * I / (fe.rho * Ar))
    L = fe.lengths.max()
    return dict(pinned=float(np.pi / (2 * L * L) * c), clamped=float(22.373 / (2 * np.pi * L * L) * c))


def job(args):
    sc, fl, r, mat, design, nel, qs = args
    fe = FE(sc, fl, r, mat, design, nel)
    F, OV, FS = [], [], []
    for q in qs:
        fb, ov, fs, f, hk = fe.solve(q); F.append(fb); OV.append(ov); FS.append(fs)
    F = np.array(F); rel, bw, mean = S.rel_branch_width(F ** 2)
    fsv = [x for x in FS if x is not None]
    kb_end = 12 * fe.E * np.pi * r ** 4 / 4 / fe.lengths ** 3
    return dict(pair=f'{sc}|{fl}', r_m=r, material=mat, design=design, nel=nel, ndof=fe.ndof, m0_kg=fe.m0, kc_N_m=fe.kc,
                J_kg_m2=fe.J, strut_lengths_m=sorted(set(np.round(fe.lengths, 9).tolist())),
                kb_N_m=sorted(set(np.round(kb_end, 6).tolist())), ka_N_m=sorted(set(np.round(fe.E * np.pi * r * r / fe.lengths, 3).tolist())),
                f_min_Hz=float(F.min()), f_max_Hz=float(F.max()), f_mean_Hz=float(F.mean()),
                branch_mean_Hz=np.sqrt(mean).tolist(), rel_width_w2=rel, rel_width_f=float(((F.max(0) - F.min(0)) / F.mean(0)).max()),
                min_overlap=float(np.min(OV)), strut_first_Hz=float(min(fsv)) if fsv else None,
                gap_ratio=float(min(fsv) / F.max()) if fsv else None, analytic_strut_Hz=analytic(fe))


def path_job(args):
    sc, fl, r, mat, design, nel, qs = args
    fe = FE(sc, fl, r, mat, design, nel); W, H = [], []
    for q in qs:
        fb, ov, fs, f, hk = fe.solve(q)
        fsel = f < 3.0 * (fs or f.max())
        W.append(f[fsel].tolist()); H.append(hk[fsel].tolist())
    return dict(pair=f'{sc}|{fl}', r_m=r, material=mat, design=design, nel=nel, f=W, hub=H)


if __name__ == '__main__':
    qsets = {}
    for sc, fl in PAIRS:
        m = S.Model(sc, fl); raw, filt, *_ = m.qsets(24, S.SEED, 0.05); qsets[(sc, fl)] = filt
    jobs = []
    for sc, fl in PAIRS:
        for design in ('release', 'lock', 'rigid'):
            for r in (0.4e-3, 0.2e-3):
                for mat in ('polymer', 'steel'):
                    jobs.append((sc, fl, r, mat, design, 6, qsets[(sc, fl)]))
            for nel in (1, 2, 4, 8):
                jobs.append((sc, fl, 0.4e-3, 'polymer', design, nel, qsets[(sc, fl)]))
    import sim_bands as B
    pq, px, pt = B.path()
    pq, px = pq[::2], px[::2]           # every second path point (cost)
    pjobs = [(sc, fl, r, 'polymer', d, 6, pq) for (sc, fl), d, r in
             [(("bcu", "nbo"), 'lock', 0.2e-3), (("bcu", "nbo"), 'rigid', 0.2e-3), (("bcu", "nbo"), 'release', 0.4e-3),
              (("nia", "nia-d"), 'lock', 0.2e-3)]]
    with Pool(14) as P:
        rows = P.map(job, jobs)
        paths = P.map(path_job, pjobs)
    json.dump(dict(params=dict(a_m=A_CELL, node_volume_m3=VNODE, nu=NU, materials=MAT, nq=24, gap_min=0.05,
                               kc_polymer_N_m=100.0), rows=rows, path=dict(x=px.tolist(), ticks=pt, runs=paths)), open(S.out('mass.json'), 'w'), indent=1)
    for r in rows:
        if r['nel'] == 6 and r['material'] == 'polymer':
            print(r['pair'], r['design'], r['r_m'], f"{r['f_mean_Hz']:.1f} Hz w2 {100*r['rel_width_w2']:.3f}% strut {r['strut_first_Hz']} gap {r['gap_ratio']}")
    print('done')
