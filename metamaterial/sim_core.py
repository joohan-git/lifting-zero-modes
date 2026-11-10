"""Shared beam-frame core for the metamaterial simulations (sim_*.py).

Geometry, contacts and the beam element come from frame.py (and through it from common/), so every
simulation uses the screened placement. This module adds what the design map, the supercell tolerance ensemble and
the finite specimen need beyond frame.Frame:

- three joint designs for the floppy struts: 'rigid' (welded hubs), 'lock' (welded hubs whose rotation is tied to the
  cage node across every contact by a rotational spring k_cr) and 'release' (both strut ends free in rotation; the
  12x12 beam block is condensed over its own end rotations before assembly, which leaves the axial bar);
- lattices of N1 x N2 x N3 cells, either periodic (Bloch phase on the supercell) or finite with the cage clamped outside
  the box (cage edges, contacts and locks that leave the box end on ground);
- per-element stiffness multipliers (disorder) for contacts and floppy-strut bending.

Units as in frame.py: floppy axial stiffness k_a = 1 (times kf_a), node mass 1, omega^2 in units of k_a / m.
Rotations carry no inertia and are condensed statically.

RCSR files: pass --rcsr DIR to any sim_*.py script or set RCSR_DIR (the variable also reaches worker processes).
Results are written to outputs/metamaterial/ (out()); the stored results are in metamaterial/.
"""
import os, sys
from pathlib import Path
if '--rcsr' in sys.argv:          # consumed here so that the scripts keep their own positional arguments
    _i = sys.argv.index('--rcsr'); os.environ['RCSR_DIR'] = sys.argv[_i + 1]; del sys.argv[_i:_i + 2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
OUTDIR = Path(__file__).resolve().parents[1] / 'outputs' / 'metamaterial'


def out(name):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    return OUTDIR / name
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import frame as F
from frame import beam_block, perp_basis

C = F.C
PAIRS = [("bcu", "nbo"), ("hxg-d", "hxg"), ("nia", "nia-d"), ("she-d", "she"), ("bsn", "lvt"),
         ("qtz-x", "uom"), ("pcu", "urk"), ("hex", "bnn"), ("fcu", "flu"), ("acs", "gra")]
SEED = 20261107


def released(block):
    """Condense a 12x12 beam block over its own six end rotations (moment-free ends)."""
    t = np.r_[0:3, 6:9]; r = np.r_[3:6, 9:12]
    Ktt = block[np.ix_(t, t)]; Ktr = block[np.ix_(t, r)]; Krr = block[np.ix_(r, r)]
    Kc = Ktt - Ktr @ la.pinvh(Krr, rtol=1e-12) @ Ktr.T
    out = np.zeros((12, 12)); out[np.ix_(t, t)] = 0.5 * (Kc + Kc.T)
    return out


class Model:
    """Unit-cell elements of one composite. Node order: floppy 0..nf-1, scaffold nf..nf+ns-1."""

    def __init__(self, sc, fl):
        self.sc_name, self.fl_name = sc, fl
        self.fr = F.Frame(sc, fl)
        self.p, self.nf, self.ns, self.n = self.fr.p, self.fr.nf, self.fr.ns, self.fr.n
        self.pr = self.fr.pr
        self.A = self.fr.sc['A']
        self.pos = np.vstack([self.fr.fl['pts'] @ self.A, self.fr.sc['pts'] @ self.A])
        self._M = None

    # ---------------------------------------------------------------- mechanisms
    def Z(self, q):
        return C.null(C.compat(self.pr, q))[0]

    def gap(self, q, M=None):
        """(M+1)-th eigenvalue of R_F^H R_F: the smallest stiffness of the floppy net outside its mechanisms."""
        M = self.M if M is None else M
        R = C.compat(self.pr, q)
        w = la.eigvalsh(R.conj().T @ R)
        return float(w[M])

    @property
    def M(self):
        """Generic number of mechanisms per conventional cell (mode over 16 random wavevectors)."""
        if self._M is None:
            rng = np.random.default_rng(1)
            ds = [self.Z(rng.random(3) - 0.5).shape[1] for _ in range(16)]
            self._M = int(np.bincount(ds).argmax())
        return self._M

    def qsets(self, nq=48, seed=SEED, gmin=0.05):
        """Raw set: the first nq draws of seed (as in sweep2). Filtered set: raw draws with gap >= gmin, completed by
        further draws of the same stream. Returns raw, filtered, gaps of both and the number of rejected draws."""
        rng = np.random.default_rng(seed)
        raw = rng.random((nq, 3)) - 0.5
        graw = np.array([self.gap(q) for q in raw])
        filt = [q for q, g in zip(raw, graw) if g >= gmin]; gf = [g for g in graw if g >= gmin]
        tried = nq
        while len(filt) < nq:
            q = rng.random(3) - 0.5; g = self.gap(q); tried += 1
            if g >= gmin:
                filt.append(q); gf.append(g)
        return raw, np.array(filt), graw, np.array(gf), tried - nq

    # ---------------------------------------------------------------- elements
    def elements(self, kc=1e-2, kf_a=1.0, kf_b=0.0, ks_a=10.0, ks_b=1.0, design='rigid', kcr=100.0, nu=0.3,
                 contact_mult=None, bend_mult=None):
        """List of (i, j, s, block12) in unit-cell indexing. Multipliers index the unit-cell contacts / floppy edges
        (used by supercells through lattice(), which passes per-instance multipliers)."""
        out = []
        for b, (i, j, s) in enumerate(self.fr.fl['edges']):
            e = self.fr.fl['unit'][b]; L = self.fr.fl['length'][b]
            kb = kf_b * (1.0 if bend_mult is None else bend_mult[b])
            kt = kb * L * L / (12.0 * (1 + nu))
            blk = beam_block(e, L, kf_a, kb, kt)
            if design == 'release':
                blk = released(blk)
            out.append(('f', b, i, j, np.array(s), blk))
        for b, (i, j, s) in enumerate(self.fr.sc['edges']):
            e = self.fr.sc['unit'][b]; L = self.fr.sc['length'][b]
            kt = ks_b * L * L / (12.0 * (1 + nu))
            out.append(('s', b, self.nf + i, self.nf + j, np.array(s), beam_block(e, L, ks_a, ks_b, kt)))
        for b, (i, k, s, nvec, ell) in enumerate(self.p['contacts']):
            blk = np.zeros((12, 12)); v = np.zeros(12); v[0:3] = -nvec; v[6:9] = nvec
            blk += kc * (1.0 if contact_mult is None else contact_mult[b]) * np.outer(v, v)
            if design == 'lock' and kcr:
                for ax in np.eye(3):
                    v = np.zeros(12); v[3:6] = -ax; v[9:12] = ax
                    blk += kcr * np.outer(v, v)
            out.append(('c', b, i, self.nf + k, np.array(s), blk))
        return out

    # ---------------------------------------------------------------- Bloch unit cell
    def bloch(self, q, **kw):
        """Condensed translational K(q), 3n x 3n."""
        n = self.n; K = np.zeros((6 * n, 6 * n), complex)
        for kind, b, i, j, s, blk in self.elements(**kw):
            ph = np.exp(2j * np.pi * np.dot(q, s))
            I = np.r_[6 * i:6 * i + 6]; J = np.r_[6 * j:6 * j + 6]
            K[np.ix_(I, I)] += blk[:6, :6]; K[np.ix_(J, J)] += blk[6:, 6:]
            K[np.ix_(I, J)] += blk[:6, 6:] * ph; K[np.ix_(J, I)] += blk[6:, :6] * np.conj(ph)
        return condense_dense(K, n)

    def lifted(self, q, **kw):
        """Lifted eigenvalues (M modes with the largest overlap with the pin-joint mechanism space), sorted, plus
        their overlaps and the full spectrum."""
        K = self.bloch(q, **kw)
        w, v = la.eigh(K)
        Z = np.zeros((3 * self.n, self.M), complex); Zq = self.Z(q)
        Z[:3 * self.nf, :min(self.M, Zq.shape[1])] = Zq[:, :self.M]
        wt = np.sum(np.abs(Z.conj().T @ v) ** 2, axis=0)
        idx = np.sort(np.argsort(-wt)[:self.M])
        return w[idx], wt[idx], w, v


def condense_dense(K, n):
    tr = np.concatenate([np.r_[6 * i:6 * i + 3] for i in range(n)])
    ro = np.concatenate([np.r_[6 * i + 3:6 * i + 6] for i in range(n)])
    Ktt = K[np.ix_(tr, tr)]; Ktr = K[np.ix_(tr, ro)]; Krr = K[np.ix_(ro, ro)]
    if np.abs(Krr).max() > 0:
        Keff = Ktt - Ktr @ la.pinvh(Krr, rtol=1e-10) @ Ktr.conj().T
    else:
        Keff = Ktt
    return 0.5 * (Keff + Keff.conj().T)


# -------------------------------------------------------------------- lattices
class Lattice:
    """N1 x N2 x N3 cells of a Model. periodic=True: supercell with Bloch phase exp(2 pi i Q.image) (Q in reduced
    supercell coordinates). periodic=False: finite block; cage nodes outside the box are clamped, floppy edges that
    leave the box are removed."""

    def __init__(self, model, N=(3, 3, 3), periodic=True):
        self.m, self.N, self.periodic = model, np.array(N), periodic
        self.ncell = int(np.prod(N))
        self.cells = np.array(np.meshgrid(*[np.arange(k) for k in N], indexing='ij')).reshape(3, -1).T
        self.cid = {tuple(c): a for a, c in enumerate(self.cells)}
        n = model.n; self.nn = n * self.ncell
        self.node_cell = np.repeat(np.arange(self.ncell), n); self.node_base = np.tile(np.arange(n), self.ncell)
        self.is_floppy = self.node_base < model.nf
        self.pos = np.array([model.pos[b] + c @ model.A for c in self.cells for b in range(n)])

    def instances(self):
        """Element instances: (kind, unit index, instance number, node a, node b or -1 (ground), image)."""
        m = self.m; out = []; count = {'f': 0, 's': 0, 'c': 0}
        groups = [('f', m.fr.fl['edges'], 0, 0), ('s', m.fr.sc['edges'], m.nf, m.nf),
                  ('c', [(i, k, s) for (i, k, s, nv, el) in m.p['contacts']], 0, m.nf)]
        for kind, edges, oi, oj in groups:
            for b, (i, j, s) in enumerate(edges):
                for a, c in enumerate(self.cells):
                    t = c + np.array(s)
                    if self.periodic:
                        img = np.floor_divide(t, self.N); cc = t - img * self.N
                        nb = self.cid[tuple(cc)] * m.n + oj + j
                    else:
                        img = np.zeros(3, int)
                        if np.all((t >= 0) & (t < self.N)):
                            nb = self.cid[tuple(t)] * m.n + oj + j
                        elif kind == 'f':
                            continue
                        else:
                            nb = -1
                    out.append((kind, b, count[kind], a * m.n + oi + i, nb, img)); count[kind] += 1
        self.count = count
        return out

    def assemble(self, Q=np.zeros(3), contact_mult=None, bend_mult=None, **kw):
        """Sparse 6nn x 6nn stiffness (complex if periodic, real otherwise). contact_mult / bend_mult are per instance."""
        base = {(k, b): blk for k, b, i, j, s, blk in self.m.elements(**kw)}
        if bend_mult is not None and kw.get('design', 'rigid') != 'release':
            # the welded beam block is linear in k_b (torsion scales with k_b): split off the bending part
            kw0 = dict(kw); kw0['kf_b'] = 0.0
            axial = {(k, b): blk for k, b, i, j, s, blk in self.m.elements(**kw0) if k == 'f'}
        rows, cols, vals = [], [], []
        dt = complex if self.periodic else float
        for kind, b, inst, a, nb, img in self.instances():
            blk = base[(kind, b)]
            if kind == 'f' and bend_mult is not None and kw.get('design', 'rigid') != 'release':
                blk = axial[(kind, b)] + bend_mult[inst] * (blk - axial[(kind, b)])
            if kind == 'c' and contact_mult is not None:
                blk = blk.copy()
                nvec = self.m.p['contacts'][b][3]; v = np.zeros(12); v[0:3] = -nvec; v[6:9] = nvec
                blk += (contact_mult[inst] - 1.0) * kw.get('kc', 1e-2) * np.outer(v, v)
            ph = np.exp(2j * np.pi * np.dot(Q, img)) if self.periodic else 1.0
            I = np.arange(6 * a, 6 * a + 6)
            rows.append(np.repeat(I, 6)); cols.append(np.tile(I, 6)); vals.append(blk[:6, :6].ravel().astype(dt))
            if nb >= 0:
                J = np.arange(6 * nb, 6 * nb + 6)
                rows += [np.repeat(J, 6), np.repeat(I, 6), np.repeat(J, 6)]
                cols += [np.tile(J, 6), np.tile(J, 6), np.tile(I, 6)]
                vals += [blk[6:, 6:].ravel().astype(dt), (blk[:6, 6:] * ph).ravel().astype(dt),
                         (blk[6:, :6] * np.conj(ph)).ravel().astype(dt)]
        K = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(6 * self.nn, 6 * self.nn)).tocsc()
        return K

    def condensed(self, K):
        """Static condensation of all rotations (sparse LU); rotation DOFs without stiffness are dropped first."""
        n = self.nn
        tr = (np.arange(n)[:, None] * 6 + np.arange(3)).ravel(); ro = (np.arange(n)[:, None] * 6 + 3 + np.arange(3)).ravel()
        d = np.abs(K.diagonal())
        ro = ro[d[ro] > 1e-14 * d.max()]
        Ktt = K[tr][:, tr].toarray(); Krr = K[ro][:, ro].tocsc(); Krt = K[ro][:, tr].toarray()
        if len(ro):
            lu = spla.splu(Krr)
            X = lu.solve(Krt)
            Keff = Ktt - K[tr][:, ro] @ X
        else:
            Keff = Ktt
        Keff = np.asarray(Keff)
        return 0.5 * (Keff + Keff.conj().T)

    def mech(self, Q=np.zeros(3)):
        """Pin-joint mechanism space of the floppy net on this lattice (translations of all nodes, 3nn columns)."""
        m = self.m; fl = m.fr.fl
        rows = []
        idx = []
        for kind, b, inst, a, nb, img in self.instances():
            if kind != 'f':
                continue
            e = fl['unit'][b]; ph = np.exp(2j * np.pi * np.dot(Q, img)) if self.periodic else 1.0
            r = np.zeros(3 * self.nn, complex); r[3 * a:3 * a + 3] -= e; r[3 * nb:3 * nb + 3] += ph * e
            rows.append(r)
        R = np.array(rows)
        fidx = np.concatenate([np.arange(3 * a, 3 * a + 3) for a in np.where(self.is_floppy)[0]])
        Rf = R[:, fidx]
        Zf, rank = C.null(Rf)
        Z = np.zeros((3 * self.nn, Zf.shape[1]), complex); Z[fidx] = Zf
        return Z


def rel_branch_width(L):
    """L: nq x M lifted values (sorted per q). Largest over branches of (max - min) / mean."""
    L = np.asarray(L); bw = L.max(0) - L.min(0)
    return float((bw / L.mean(0)).max()), bw, L.mean(0)
