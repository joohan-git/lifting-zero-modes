"""Beam-frame (metamaterial) model of a floppy net caged in its tiling dual.

Every vertex is a node with 3 translations and 3 rotations. Every edge of either net is an
Euler-Bernoulli beam with axial stiffness k_a, transverse bending stiffness k_b (= 12 EI / L^3) and
torsion k_t (= GJ / L). Struts are massless; nodes carry unit mass and no rotary inertia, so the
rotations are condensed statically. Contacts between the nets are normal springs k_c (optionally
with a shear spring k_cs, and a rotational spring k_cr that ties the rotation of a floppy node to the cage
node across each contact). Units: k_a(floppy) = 1, node mass = 1, omega^2 in units of k_a / m.

The lifted band is identified by overlap with the pin-joint mechanism space of the floppy net
(translations only). Lattice-vector phase convention, as in the released code (common.composite).
"""
import sys
from pathlib import Path
import numpy as np
import scipy.linalg as la

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import composite as C  # noqa: E402
from common.records import find  # noqa: E402


def perp_basis(e):
    a = np.array([1.0, 0, 0]) if abs(e[0]) < 0.9 else np.array([0, 1.0, 0])
    t1 = np.cross(e, a); t1 /= la.norm(t1); t2 = np.cross(e, t1)
    return t1, t2


def beam_block(e, L, ka, kb, kt):
    """12x12 stiffness in global coords for dofs (u_a, th_a, u_b, th_b); kb = 12 EI / L^3."""
    K = np.zeros((12, 12))
    ua, tha, ub, thb = slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12)
    # axial
    v = np.zeros(12); v[ua] = -e; v[ub] = e
    K += ka * np.outer(v, v)
    # torsion
    v = np.zeros(12); v[tha] = -e; v[thb] = e
    K += kt * np.outer(v, v)
    # bending in two planes: transverse direction t, slope rotation axis b = e x t
    EI_L3 = kb / 12.0
    kloc = EI_L3 * np.array([[12, 6 * L, -12, 6 * L], [6 * L, 4 * L * L, -6 * L, 2 * L * L],
                             [-12, -6 * L, 12, -6 * L], [6 * L, 2 * L * L, -6 * L, 4 * L * L]])
    for t in perp_basis(e):
        b = np.cross(e, t)
        T = np.zeros((4, 12))
        T[0, ua] = t; T[1, tha] = b; T[2, ub] = t; T[3, thb] = b
        K += T.T @ kloc @ T
    return K


class Frame:
    def __init__(self, sc, fl, shift=None):
        r = find(sc, fl)
        self.p = C.pair(sc, fl, r["shift"] if shift is None else shift)
        self.fl, self.sc = self.p["fl"], self.p["sc"]
        self.nf, self.ns = self.fl["n"], self.sc["n"]
        self.n = self.nf + self.ns
        self.pr = C.prep(self.fl)

    def assemble(self, q, kf_a=1.0, kf_b=0.0, ks_a=1.0, ks_b=0.0, kc=1e-2, kcs=0.0, nu=0.3, kcr=0.0):
        """Return the condensed translational stiffness K(q) (3n x 3n, complex Hermitian)."""
        n = self.n
        K = np.zeros((6 * n, 6 * n), complex)

        def add(i, j, s, block):
            ph = np.exp(2j * np.pi * np.dot(q, s))
            I = np.r_[6 * i:6 * i + 6]; J = np.r_[6 * j:6 * j + 6]
            K[np.ix_(I, I)] += block[:6, :6]
            K[np.ix_(J, J)] += block[6:, 6:]
            K[np.ix_(I, J)] += block[:6, 6:] * ph
            K[np.ix_(J, I)] += block[6:, :6] * np.conj(ph)

        for g, off, ka, kb in [(self.fl, 0, kf_a, kf_b), (self.sc, self.nf, ks_a, ks_b)]:
            for b, (i, j, s) in enumerate(g["edges"]):
                e = g["unit"][b]; L = g["length"][b]
                kt = kb * L * L / (12.0 * (1 + nu))  # GJ/L of a circular rod with EI = kb L^3 / 12
                add(off + i, off + j, np.array(s), beam_block(e, L, ka, kb, kt))
        for (i, k, s, nvec, ell) in self.p["contacts"]:
            blk = np.zeros((12, 12))
            v = np.zeros(12); v[0:3] = -nvec; v[6:9] = nvec
            blk += kc * np.outer(v, v)
            if kcs:
                for t in perp_basis(nvec):
                    v = np.zeros(12); v[0:3] = -t; v[6:9] = t
                    blk += kcs * np.outer(v, v)
            if kcr:
                for ax in np.eye(3):
                    v = np.zeros(12); v[3:6] = -ax; v[9:12] = ax
                    blk += kcr * np.outer(v, v)
            add(i, self.nf + k, np.array(s), blk)
        # static condensation of rotations
        tr = np.concatenate([np.r_[6 * i:6 * i + 3] for i in range(n)])
        ro = np.concatenate([np.r_[6 * i + 3:6 * i + 6] for i in range(n)])
        Ktt = K[np.ix_(tr, tr)]; Ktr = K[np.ix_(tr, ro)]; Krr = K[np.ix_(ro, ro)]
        if np.abs(Krr).max() > 0:
            Keff = Ktt - Ktr @ la.pinvh(Krr, rtol=1e-10) @ Ktr.conj().T
        else:
            Keff = Ktt
        return 0.5 * (Keff + Keff.conj().T)

    def mech(self, q):
        Rf = C.compat(self.pr, q)
        Z, _ = C.null(Rf)
        Zfull = np.zeros((3 * self.n, Z.shape[1]), complex)
        Zfull[: 3 * self.nf] = Z
        return Zfull

    def lifted(self, q, **kw):
        Keff = self.assemble(q, **kw)
        w2, ev = la.eigh(Keff)
        Z = self.mech(q)
        nu = Z.shape[1]
        wt = np.sum(np.abs(Z.conj().T @ ev) ** 2, axis=0)
        idx = np.sort(np.argsort(-wt)[:nu])
        return w2[idx], wt[idx], w2

    def first_order(self, q, kc=1e-2, kf_b=0.0, nu=0.3):
        """eig[Z^H (kc S + K_bend,F) Z]: projected contact plus projected bending of the floppy net."""
        Z = self.mech(q)[: 3 * self.nf]
        Kb = self.assemble(q, kf_a=0.0, kf_b=kf_b, ks_a=0.0, ks_b=0.0, kc=0.0)[: 3 * self.nf, : 3 * self.nf]
        return la.eigvalsh(Z.conj().T @ (kc * self.p["S"] + Kb) @ Z)
