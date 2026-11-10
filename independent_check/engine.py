"""Second, separately written implementation of the net expansion, Bloch rigidity matrix and
composite placement. It cross-checks the floppy-side splitting sets (SM Table S4), the prospective
classifier test (Section S3.6), the zone fractions (Section S3.5) and the drive (Section S8.2).
Own cgd parser and own space-group expansion (gemmi for the operator list only).
Conventions match the main code: conventional cell, Bloch phase exp(2*pi*i q.s) with s the integer
cell shift of an edge or contact, unit bond vectors, k = k_c = m = 1.
"""
import re, itertools, numpy as np, gemmi
from scipy.linalg import svd, svdvals

TOL_FRAC = 2e-4      # fractional-coordinate matching tolerance
RANK_TOL = 1e-8      # relative singular-value tolerance for rank/kernel

def parse_cgd(path):
    out = {}; cur = None
    for line in open(path, encoding='utf8'):
        s = line.split()
        if not s or s[0].startswith('#'): continue
        key = s[0].upper()
        if key == 'CRYSTAL': cur = dict(nodes=[], edges=[])
        elif key == 'NAME': cur['name'] = s[1]
        elif key == 'GROUP': cur['group'] = s[1]
        elif key == 'CELL': cur['cell'] = [float(x) for x in s[1:]]
        elif key == 'NODE': cur['nodes'].append((int(s[2]), np.array([float(x) for x in s[3:]])))
        elif key == 'EDGE':
            v = np.array([float(x) for x in s[1:]]); cur['edges'].append(v.reshape(2, -1))
        elif key == 'END':
            if len(cur.get('cell', [])) == 6: out[cur['name']] = cur
            cur = None
    return out

def cell_matrix(cell):
    """Rows = lattice vectors a, b, c in Cartesian; cartesian = frac @ A."""
    a, b, c, al, be, ga = cell
    al, be, ga = np.deg2rad([al, be, ga])
    va = np.array([a, 0, 0])
    vb = np.array([b*np.cos(ga), b*np.sin(ga), 0])
    cx = c*np.cos(be); cy = c*(np.cos(al) - np.cos(be)*np.cos(ga))/np.sin(ga)
    cz = np.sqrt(max(c*c - cx*cx - cy*cy, 0.0))
    return np.array([va, vb, [cx, cy, cz]])

def space_ops(group):
    sg = gemmi.find_spacegroup_by_name(group)
    if sg is None: raise ValueError('unknown group ' + group)
    ops = []
    for o in sg.operations():
        r = np.array(o.rot, float)/o.DEN; t = np.array(o.tran, float)/o.DEN
        ops.append((r, t))
    return ops

def wrap(x): return x - np.floor(x + 1e-9)   # into [0,1)

def dfrac(x, y):
    d = (x - y + 0.5) % 1 - 0.5; return np.linalg.norm(d, axis=-1)

def build(rec):
    """Expand nodes and edges of a cgd record with the space-group operators. Returns dict with
    pts (V x 3 frac), edges [(i, j, s)], cn, A (physical cell, rows), V, E."""
    ops = space_ops(rec['group'])
    pts = []; cns = []
    for cn, x in rec['nodes']:
        # symmetrise the representative onto its site (removes 1e-5 rounding of the file)
        stab = [((r @ x + t) - x) - np.rint((r @ x + t) - x) for r, t in ops]
        stab = [d for d in stab if np.linalg.norm(d) < 10*TOL_FRAC]
        x = x + np.mean(stab, axis=0)
        for r, t in ops:
            p = wrap(r @ x + t)
            if not pts or dfrac(np.array(pts), p).min() > TOL_FRAC:
                pts.append(p); cns.append(cn)
    pts = np.array(pts); cn = np.array(cns); V = len(pts)
    def locate(x):
        d = dfrac(pts, x); k = int(np.argmin(d))
        if d[k] > 5*TOL_FRAC: raise ValueError(f'endpoint unmatched {d[k]:.2e}')
        return k, np.rint(x - pts[k]).astype(int)
    es = set()
    for a, b in rec['edges']:
        for r, t in ops:
            i, si = locate(r @ a + t); j, sj = locate(r @ b + t)
            s = sj - si
            if i == j and not s.any(): raise ValueError('zero edge')
            e = (i, j, tuple(int(v) for v in s))
            e2 = (j, i, tuple(int(-v) for v in s))
            es.add(min(e, e2))
    es = sorted(es)
    deg = np.zeros(V, int)
    for i, j, s in es: deg[i] += 1; deg[j] += 1
    if not np.array_equal(deg, cn):
        raise ValueError(f'degree mismatch: got {sorted(set(deg))} expected {sorted(set(cn))} (V={V},E={len(es)})')
    A = cell_matrix(rec['cell'])
    return dict(name=rec['name'], group=rec['group'], pts=pts, edges=es, cn=cn, A=A, V=V, E=len(es))

def bond_geometry(g, A=None):
    """Unit bond vectors and lengths in the cell A (default: own cell)."""
    A = g['A'] if A is None else A
    vec = np.array([(g['pts'][j] + np.array(s) - g['pts'][i]) @ A for i, j, s in g['edges']])
    L = np.linalg.norm(vec, axis=1)
    return vec/L[:, None], L

def rigidity(g, q, unit):
    """Bloch rigidity matrix R(q): E x 3V, R u = first-order bond extensions."""
    V = g['V']; R = np.zeros((len(g['edges']), 3*V), complex)
    for b, (i, j, s) in enumerate(g['edges']):
        z = np.exp(2j*np.pi*np.dot(q, s)); v = unit[b]
        R[b, 3*i:3*i+3] -= v; R[b, 3*j:3*j+3] += z*v
    return R

def kernel(R, tol=RANK_TOL):
    """Orthonormal basis of ker R (columns)."""
    _, sv, vh = svd(R, full_matrices=True)
    smax = sv[0] if len(sv) else 1.0
    r = int(np.sum(sv > tol*max(1.0, smax)))
    return vh[r:].conj().T

def n_mech(R, tol=RANK_TOL):
    sv = svdvals(R); smax = sv[0] if len(sv) else 1.0
    return R.shape[1] - int(np.sum(sv > tol*max(1.0, smax)))

# ---------------- composite placement (same rule as the main screen) ----------------
SHIFTS1 = np.array(list(itertools.product(range(-1, 2), repeat=3)))
SHIFTS2 = np.array(list(itertools.product(range(-2, 3), repeat=3)))

def place(sc, fl, grid=8, shift=None):
    """Floppy net fl placed in the physical cell of scaffold sc (floppy fractional coordinates kept).
    Origin shift t chosen on an n^3 grid maximizing min distance from floppy vertices to scaffold
    vertices and bars (first maximum in lexicographic grid order), unless shift is given."""
    A = sc['A']; Psc = sc['pts']; Pfl = fl['pts']
    Sc = (Psc[None, :, :] + SHIFTS1[:, None, :]).reshape(-1, 3) @ A
    U = np.array([Psc[i] @ A for i, j, s in sc['edges']]); W = np.array([(Psc[j] + np.array(s)) @ A for i, j, s in sc['edges']])
    SegU = (U[None, :, :] + (SHIFTS1 @ A)[:, None, :]).reshape(-1, 3)
    SegW = (W[None, :, :] + (SHIFTS1 @ A)[:, None, :]).reshape(-1, 3)
    AB = SegW - SegU; AB2 = np.einsum('ij,ij->i', AB, AB)
    def clearance(t):
        Q = wrap(Pfl + t) @ A
        dv = np.min(np.linalg.norm(Q[:, None, :] - Sc[None, :, :], axis=2))
        T = np.clip(np.einsum('pij,ij->pi', Q[:, None, :] - SegU[None, :, :], AB)/AB2, 0, 1)
        P = SegU[None, :, :] + T[:, :, None]*AB[None, :, :]
        de = np.min(np.linalg.norm(Q[:, None, :] - P, axis=2))
        return min(dv, de)
    if shift is None:
        best = None
        for tt in itertools.product(range(grid), repeat=3):
            t = np.array(tt)/grid; c = clearance(t)
            if best is None or c > best[0] + 1e-9: best = (c, t)
        cbest, t = best
    else:
        t = np.array(shift, float); cbest = clearance(t)
    # contacts: unwrapped floppy positions P'_i = Pfl_i + t, scaffold images x_k + sh, |sh|<=2
    contacts = []
    for i in range(fl['V']):
        p = Pfl[i] + t
        cand = []
        for k in range(sc['V']):
            for sh in SHIFTS2:
                dvec = (Psc[k] + sh - p) @ A; dd = np.linalg.norm(dvec)
                cand.append((dd, k, tuple(int(v) for v in sh), dvec/dd))
        dn = min(c[0] for c in cand)
        for dd, k, sh, c in cand:
            if dd < 1.05*dn: contacts.append((i, k, sh, c, dd))
    return dict(A=A, shift=t, clearance=cbest, contacts=contacts)

def contact_tensor(fl, contacts):
    S = np.zeros((3*fl['V'], 3*fl['V']))
    for i, k, sh, c, dd in contacts: S[3*i:3*i+3, 3*i:3*i+3] += np.outer(c, c)
    return S

def contact_matrix(fl, sc, contacts, q):
    """Bloch contact rigidity rows: C u = first-order contact extensions; columns [floppy | scaffold]."""
    nF, nS = 3*fl['V'], 3*sc['V']; C = np.zeros((len(contacts), nF + nS), complex)
    for r, (i, k, sh, c, dd) in enumerate(contacts):
        z = np.exp(2j*np.pi*np.dot(q, sh))
        C[r, 3*i:3*i+3] -= c; C[r, nF+3*k:nF+3*k+3] += z*c
    return C

def lifted_fixed(fl, S, unitF, q, tol=RANK_TOL):
    """First-order lifted band with the scaffold held fixed: eig of Z_F^H S Z_F."""
    Z = kernel(rigidity(fl, q, unitF), tol)
    if Z.shape[1] == 0: return np.array([])
    return np.linalg.eigvalsh(Z.conj().T @ S @ Z)

def lifted_coupled(fl, sc, contacts, unitF, unitS, q, tol=RANK_TOL):
    """First-order lifted band with scaffold mechanisms included: eig of (CZ)^H (CZ), Z = diag(Z_F, Z_S)."""
    ZF = kernel(rigidity(fl, q, unitF), tol); ZS = kernel(rigidity(sc, q, unitS), tol)
    nF, nS = 3*fl['V'], 3*sc['V']
    Z = np.zeros((nF + nS, ZF.shape[1] + ZS.shape[1]), complex)
    Z[:nF, :ZF.shape[1]] = ZF; Z[nF:, ZF.shape[1]:] = ZS
    C = contact_matrix(fl, sc, contacts, q)
    M = C @ Z
    return np.linalg.eigvalsh(M.conj().T @ M), ZF.shape[1], ZS.shape[1]

def generic_mech(g, unit, nq=12, seed=0, tol=RANK_TOL):
    rng = np.random.default_rng(seed)
    return int(np.median([n_mech(rigidity(g, rng.random(3), unit), tol) for _ in range(nq)]))

def screen_pair(cg, scaffold, floppy, nq=40, seed=0, shift=None):
    """Replicates the main screen's observables for one (scaffold, floppy) pair with forced roles."""
    sc = build(cg[scaffold]); fl = build(cg[floppy])
    unitS, _ = bond_geometry(sc)
    mS = generic_mech(sc, unitS)
    P = place(sc, fl, shift=shift)
    unitF, LF = bond_geometry(fl, P['A'])
    mF = generic_mech(fl, unitF)
    S = contact_tensor(fl, P['contacts'])
    rng = np.random.default_rng(seed); evs = []; zeros = []
    for _ in range(nq):
        w = lifted_fixed(fl, S, unitF, rng.random(3))
        if len(w) == 0: continue
        evs.append(np.sort(w)); zeros.append(int(np.sum(w < 1e-8)))
    out = dict(scaffold=scaffold, floppy=floppy, V_S=sc['V'], E_S=sc['E'], V_F=fl['V'], E_F=fl['E'],
               nu_F=3*fl['V'] - fl['E'], mech_S=mS, mech_F=mF, n_contacts=len(P['contacts']),
               clearance=float(P['clearance']), shift=P['shift'].tolist())
    if evs:
        ev = np.array(evs)
        out.update(unlifted_median=int(np.median(zeros)), eig_min=float(ev.min()), eig_max=float(ev.max()),
                   spread_rel=float(np.mean(ev.max(0) - ev.min(0))/max(np.mean(ev), 1e-12)),
                   bands=np.round(ev.mean(0), 6).tolist())
    return out, sc, fl, P, S, unitF, unitS
