"""Exact condition for q-independent locked bending, checked on the ten exactly flat pairs.

With hub rotations held at zero (ideal lock), the transverse beam energy of a strut is (k_b/2)|P_perp Delta_e|^2, and
on ker R_F the axial part vanishes, so Z^H K_b^(theta=0) Z = k_b Z^H (L(q) x I_3) Z with L the Bloch graph Laplacian.
Locked bending is flat at first order iff the spectrum of Z^H L Z is q-independent. For each pair this script reports
the identity residual, the mechanism-carrying vertices, their coordination and planarity, the normal purity of the
mechanism on them, the largest |n_i . n_j| over edges joining two mechanism vertices, and the range of the locked and of
the unlocked (rotations condensed) projected bending spectra over 30 wavevectors with gap >= 0.02.
Run: python metamaterial/sim_normals.py --rcsr DIR   (stored result: metamaterial/normals.json)
"""
import json
import numpy as np
import scipy.linalg as la
import sim_core as S


def laplacian(m, q):
    n = m.nf; L = np.zeros((n, n), complex)
    for i, j, s in m.fr.fl['edges']:
        ph = np.exp(2j * np.pi * np.dot(q, s))
        L[i, i] += 1; L[j, j] += 1; L[i, j] -= ph; L[j, i] -= np.conj(ph)
    return L


def bend_theta0(m, q, kb=1.0):
    """Translational block of the floppy beams with all rotations held at zero."""
    n = m.nf; K = np.zeros((3 * n, 3 * n), complex)
    for b, (i, j, s) in enumerate(m.fr.fl['edges']):
        blk = S.beam_block(m.fr.fl['unit'][b], m.fr.fl['length'][b], 0.0, kb, 0.0)
        ph = np.exp(2j * np.pi * np.dot(q, s)); I = np.r_[3 * i:3 * i + 3]; J = np.r_[3 * j:3 * j + 3]
        K[np.ix_(I, I)] += blk[0:3, 0:3]; K[np.ix_(J, J)] += blk[6:9, 6:9]
        K[np.ix_(I, J)] += blk[0:3, 6:9] * ph; K[np.ix_(J, I)] += blk[6:9, 0:3] * np.conj(ph)
    return K


def main():
    out = {}
    for sc, fl in S.PAIRS:
        m = S.Model(sc, fl); rng = np.random.default_rng(3); qs = []
        while len(qs) < 30:
            q = rng.random(3) - 0.5
            if m.gap(q) >= 0.02:
                qs.append(q)
        deg = np.zeros(m.nf, int)
        for i, j, s in m.fr.fl['edges']:
            deg[i] += 1; deg[j] += 1
        T = S.C.tensors(m.fr.fl)
        normal = []; planar = []
        for i in range(m.nf):
            w, v = la.eigh(T[i]); planar.append(bool(w[0] < 1e-9 * w[-1])); normal.append(v[:, 0])
        normal = np.array(normal)
        wv = np.zeros(m.nf); wn = np.zeros(m.nf); ident = 0.0; lock = []; free = []
        for q in qs:
            Z = m.Z(q)[:, :m.M]
            Zr = Z.reshape(m.nf, 3, -1)
            wv += np.sum(np.abs(Zr) ** 2, axis=(1, 2))
            wn += np.sum(np.abs(np.einsum('ia,iak->ik', normal, Zr)) ** 2, axis=1)
            Kb = bend_theta0(m, q); LL = np.kron(laplacian(m, q), np.eye(3))
            ident = max(ident, float(np.abs(Z.conj().T @ (Kb - LL) @ Z).max()))
            lock.append(np.sort(la.eigvalsh(Z.conj().T @ LL @ Z)))
            Kc = m.fr.assemble(q, kf_a=0.0, kf_b=1.0, ks_a=0.0, ks_b=0.0, kc=0.0)[:3 * m.nf, :3 * m.nf]
            free.append(np.sort(la.eigvalsh(Z.conj().T @ Kc @ Z)))
        mv = wv / wv.sum() > 1e-12
        purity = float(wn[mv].sum() / wv[mv].sum())
        dots = [abs(float(normal[i] @ normal[j])) for i, j, s in m.fr.fl['edges'] if mv[i] and mv[j] and planar[i] and planar[j]]
        lock = np.array(lock); free = np.array(free)
        rl, _, _ = S.rel_branch_width(lock); rf, _, _ = S.rel_branch_width(free)
        out[f'{sc}|{fl}'] = dict(M=m.M, mech_vertices=int(mv.sum()), nf=m.nf,
                                 mech_degrees=sorted(set(deg[mv].tolist())), all_degrees=sorted(set(deg.tolist())),
                                 mech_planar=int(sum(planar[i] for i in range(m.nf) if mv[i])),
                                 normal_purity=purity if all(planar[i] for i in range(m.nf) if mv[i]) else None,
                                 coupled_planar_pairs=len(dots), max_normal_dot=max(dots) if dots else None,
                                 identity_residual=ident, lock_min=float(lock.min()), lock_max=float(lock.max()),
                                 lock_rel_width=rl, free_min=float(free.min()), free_max=float(free.max()), free_rel_width=rf)
        print(sc, fl, out[f'{sc}|{fl}'], flush=True)
    json.dump(out, open(S.out('normals.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
