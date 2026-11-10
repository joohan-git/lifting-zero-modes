"""Composite of a scaffold net and a floppy net: placement, contacts, Bloch matrices, site symmetry.

Conventional-cell fractional q, phase exp(2*pi*i*q.shift); unit masses. Rest lengths equal
current lengths, so every geometric-stiffness term vanishes.
"""
import csv
import itertools
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
import scipy.linalg as la
from scipy.spatial import cKDTree
import spglib
from . import nets as E

TOL = 2e-7
SEED = 20261106


def dump(path, obj):
    def default(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, Path): return str(x)
        if isinstance(x, (set, tuple)): return list(x)
        raise TypeError(type(x).__name__)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=default, allow_nan=False), encoding='utf8')


def csvout(path, rows):
    if not rows: return
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fields); w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in r.items()})


def CG():
    return E.catalog()


@lru_cache(None)
def graph(name):
    return E.build(CG()[name], primitive=False)


def remetric(g, A, translation=None):
    out = dict(g); p = g['pts'].copy(); edges = g['edges']
    if translation is not None:
        wrap = np.floor(p + translation + 1e-10).astype(int); p = p + translation - wrap
        edges = [(i, j, tuple(np.array(s) + wrap[j] - wrap[i])) for i, j, s in edges]
    out.update(pts=p, edges=edges, A=A)
    vec = np.array([(p[j] + s - p[i]) @ A for i, j, s in edges])
    lengths = la.norm(vec, axis=1); out.update(unit=vec / lengths[:, None], length=lengths)
    return out


def optimize_shift(sc, fl):
    """Relative origin on the 8^3 grid maximizing the clearance of floppy vertices from scaffold
    vertices and edges (first maximum, 1e-9 tie rule). A spatial index bounds the edge search."""
    A = sc['A']; d = sc['d']; sh = np.array(list(itertools.product(range(-1, 2), repeat=d)))
    images = (sc['pts'][None, :, :] + sh[:, None, :]).reshape(-1, d) @ A; vertices = cKDTree(images)
    U0 = np.array([sc['pts'][i] @ A for i, j, s in sc['edges']]); W0 = np.array([(sc['pts'][j] + s) @ A for i, j, s in sc['edges']])
    U = (U0[None, :, :] + (sh @ A)[:, None, :]).reshape(-1, d); W = (W0[None, :, :] + (sh @ A)[:, None, :]).reshape(-1, d)
    AB = W - U; den = np.einsum('ij,ij->i', AB, AB); halfmax = np.sqrt(den.max()) / 2
    centers = cKDTree((U + W) / 2); best = (-1, None)
    for tt in itertools.product(range(8), repeat=d):
        shift = np.array(tt) / 8; Q = ((fl['pts'] + shift) % 1) @ A; vd = vertices.query(Q)[0]; v = float(vd.min())
        if v <= best[0] + 1e-9: continue
        # a segment closer than v has its midpoint closer than v + half its length
        for point in Q[np.argsort(vd)]:
            ids = np.array(centers.query_ball_point(point, v + halfmax + 1e-12), int)
            if len(ids):
                z = point - U[ids]; ab = AB[ids]; tau = np.clip(np.einsum('ij,ij->i', z, ab) / den[ids], 0, 1)
                v = min(v, float(la.norm(z - tau[:, None] * ab, axis=1).min()))
            if v <= best[0] + 1e-9: break
        if v > best[0] + 1e-9: best = (v, shift)
    return best[1], best[0]


def pair(scname, flname, shift=None):
    """Both nets in the physical conventional cell of the scaffold; contacts from each floppy vertex
    to every scaffold vertex within 1.05 times its nearest contact distance."""
    sc0, fl0 = graph(scname), graph(flname)
    A = sc0['A'] * sc0['original_mean_length']; sc = remetric(sc0, A)
    clearance = None
    if shift is None: shift, clearance = optimize_shift(sc, fl0)
    shift = np.array(shift, float); fl = remetric(fl0, A, shift)
    d = fl['d']; shifts = np.array(list(itertools.product(range(-1, 2), repeat=d)))
    cs = []; S = np.zeros((d * fl['n'], d * fl['n'])); stars = [[] for _ in range(fl['n'])]
    images = (sc['pts'][:, None, :] + shifts[None, :, :]).reshape(-1, d)
    for i, p in enumerate(fl['pts']):
        v = (images - p) @ A; ell = la.norm(v, axis=1); near = ell.min()
        if near < 1e-10: raise ValueError('overlapping contact endpoints')
        for z in np.where(ell < 1.05 * near)[0]:
            k, j = divmod(int(z), len(shifts)); n = v[z] / ell[z]
            cs.append((i, k, shifts[j], n, float(ell[z]))); stars[i].append(n)
            S[d * i:d * i + d, d * i:d * i + d] += np.outer(n, n)
    return dict(sc=sc, fl=fl, shift=shift, contacts=cs, S=S, stars=stars, clearance=clearance)


def prep(g):
    d = g['d']; edges = g['edges']; b = np.repeat(np.arange(g['e']), d)
    ii = np.array([i for i, j, s in edges]); jj = np.array([j for i, j, s in edges])
    ss = np.array([s for i, j, s in edges]); cols = np.arange(d)
    return b, (ii[:, None] * d + cols).ravel(), (jj[:, None] * d + cols).ravel(), ss, g['unit'], d * g['n']


def compat(p, q):
    b, ci, cj, ss, u, n = p; R = np.zeros((len(ss), n), complex)
    R[b, ci] -= u.ravel(); R[b, cj] += ((np.exp(2j * np.pi * (ss @ q)))[:, None] * u).ravel()
    return R


def null(R):
    _, sv, v = la.svd(R, full_matrices=True, check_finite=False, lapack_driver='gesdd')
    rank = int(np.count_nonzero(sv > TOL * max(1, sv[0] if len(sv) else 1)))
    return v[rank:].conj().T, rank


def contact_matrix(p, q):
    f, s = p['fl'], p['sc']; d = f['d']; nf = d * f['n']; C = np.zeros((len(p['contacts']), nf + d * s['n']), complex)
    for b, (i, j, sh, v, ell) in enumerate(p['contacts']):
        C[b, d * i:d * i + d] = -v; C[b, nf + d * j:nf + d * j + d] = np.exp(2j * np.pi * np.dot(q, sh)) * v
    return C


def tensors(g):
    d = g['d']; out = np.zeros((g['n'], d, d))
    for b, (i, j, s) in enumerate(g['edges']):
        vv = np.outer(g['unit'][b], g['unit'][b]); out[i] += vv; out[j] += vv
    return out


def generic_stats(p, nq=40, seed=SEED):
    """First-order lifted spectrum eig(Z^H S Z) at nq random wavevectors (scaffold held fixed)."""
    pf, ps = prep(p['fl']), prep(p['sc']); rng = np.random.default_rng(seed); ev = []; mf = []; ms = []; Pavg = np.zeros_like(p['S'], complex)
    for q in rng.random((nq, 3)):
        Z, r = null(compat(pf, q)); _, rs = null(compat(ps, q)); mf.append(Z.shape[1]); ms.append(ps[-1] - rs)
        w = la.eigvalsh(Z.conj().T @ p['S'] @ Z); ev.append(w); Pavg += Z @ Z.conj().T
    same = len(set(mf)) == 1
    e = np.array(ev) if same else None
    mean = float(np.mean(np.concatenate(ev))) if any(mf) else None
    return dict(nq=nq, M_F=sorted(set(mf)), M_S=sorted(set(ms)), nu_F=pf[-1] - len(pf[3]), nu_S=ps[-1] - len(ps[3]),
                generic_self_stress_F=sorted(set(len(pf[3]) - pf[-1] + x for x in mf)),
                eig_min=float(min(x.min() for x in ev)) if any(mf) else None,
                eig_max=float(max(x.max() for x in ev)) if any(mf) else None,
                branch_means=e.mean(0).tolist() if same and any(mf) else [],
                branch_ranges=np.ptp(e, axis=0).tolist() if same and any(mf) else [],
                spread_rel=float(np.ptp(e, axis=0).mean() / max(abs(mean), 1e-12)) if same and any(mf) else None,
                centroid=mean, trace_Pavg_S=float(np.trace(Pavg / nq @ p['S']).real),
                generic_unlifted=sorted(set(int(np.sum(x < 1e-8)) for x in ev)))


# ------------------------------------------------------------------ primitive cell and full composite
def from_positions(name, A, pts, edges):
    d = len(A); g = dict(name=name, A=np.array(A), pts=np.array(pts), edges=edges, n=len(pts), e=len(edges), d=d)
    return remetric(g, np.array(A))


def primitive(p, B):
    """Reduce a composite to the primitive cell with basis rows B (conventional fractional coordinates)."""
    inv = la.inv(B); new = []; maps = []
    for g in (p['fl'], p['sc']):
        pts = []; mapping = []
        for x in g['pts'] @ inv:
            f = x % 1; f[np.isclose(f, 1, atol=1e-9)] = 0
            match = [j for j, y in enumerate(pts) if la.norm((f - y + .5) % 1 - .5) < 1e-7]
            j = match[0] if match else len(pts)
            if not match: pts.append(f)
            mapping.append((j, np.rint(x - pts[j]).astype(int)))
        edges = set()
        for i, j, s in g['edges']:
            ii, si = mapping[i]; jj, sj = mapping[j]; edges.add(E.canon(ii, jj, sj + np.rint(np.array(s) @ inv).astype(int) - si))
        new.append(from_positions(g['name'], B @ g['A'], pts, sorted(edges))); maps.append(mapping)
    contacts = {}
    for i, j, s, v, ell in p['contacts']:
        ii, si = maps[0][i]; jj, sj = maps[1][j]; shift = tuple(sj + np.rint(np.array(s) @ inv).astype(int) - si)
        contacts[(ii, jj, shift)] = (ii, jj, np.array(shift), v, ell)
    return dict(fl=new[0], sc=new[1], contacts=list(contacts.values()))


def full_matrices(p, q, kc):
    """Dynamical matrix of the full composite (k_s = k_F = 1) and the contact term C^H C."""
    R = la.block_diag(compat(prep(p['fl']), q), compat(prep(p['sc']), q)); C = contact_matrix(p, q)
    P = kc * C.conj().T @ C; D = R.conj().T @ R + P
    return D, P


# ------------------------------------------------------------------ site symmetry
def pointgroup(rot):
    return spglib.get_pointgroup(np.rint(rot).astype(int))[0].strip()


def graph_op_preserves(g, R, t):
    tree = cKDTree(g['pts'] % 1, boxsize=1); x = g['pts'] @ R.T + t; dist, ids = tree.query(x % 1)
    if dist.max() > 3e-5: return False
    images = np.rint(x - g['pts'][ids]).astype(int); edges = set(E.canon(i, j, s) for i, j, s in g['edges'])
    return all(E.canon(ids[i], ids[j], np.rint(R @ np.array(s)).astype(int) + images[j] - images[i]) in edges for i, j, s in g['edges'])


DECOMP = {
    '1': ('3A', 3), '-1': ('3Au', 3), '2': ('A+2B', 2), 'm': ("2A'+A''", 2), '2/m': ('Au+2Bu', 2),
    '222': ('B1+B2+B3', 1), 'mm2': ('A1+B1+B2', 1), 'mmm': ('B1u+B2u+B3u', 1),
    '4': ('A+E(real)', 1), '-4': ('B+E(real)', 1), '4/m': ('Au+Eu(real)', 1), '422': ('A2+E', 1), '4mm': ('A1+E', 1), '-42m': ('B2+E', 1), '4/mmm': ('A2u+Eu', 1),
    '3': ('A+E(real)', 1), '-3': ('Au+Eu(real)', 1), '32': ('A2+E', 1), '3m': ('A1+E', 1), '-3m': ('A2u+Eu', 1),
    '6': ('A+E1(real)', 1), '-6': ("A''+E'(real)", 1), '6/m': ('Au+E1u(real)', 1), '622': ('A2+E1', 1), '6mm': ('A1+E1', 1), '-6m2': ("A2''+E'", 1), '6/mmm': ('A2u+E1u', 1),
    '23': ('T', 1), 'm-3': ('Tu', 1), '432': ('T1', 1), '-43m': ('T2', 1), 'm-3m': ('T1u', 1)}


def symmetry(p):
    """Site group of every floppy vertex in the net alone and in the composite, contact tensor S_i,
    bond tensor T_i, planarity and the irreducible decomposition of the vector representation."""
    f, s = p['fl'], p['sc']; A = f['A']; inv = la.inv(A.T)
    ops, _ = E.sg_ops(CG()[f['name']], f['setting']); shift = p['shift']
    tree = cKDTree(s['pts'] % 1, boxsize=1)
    good = []
    for r, t in ops:
        tt = t + shift - r @ shift
        dist, _ = tree.query((s['pts'] @ r.T + tt) % 1)
        good.append(bool(np.max(dist) < 3e-5 and graph_op_preserves(s, r, tt)))
    T = tensors(f); sites = []
    for i, x in enumerate(f['pts']):
        single = []; composite = []; cart = []
        for k, (r, t) in enumerate(ops):
            tt = t + shift - r @ shift
            if la.norm((r @ x + tt - x + .5) % 1 - .5) < 3e-5:
                single.append(r)
                if good[k]: composite.append(r); cart.append(A.T @ r @ inv)
        hm = pointgroup(single); hc = pointgroup(composite)
        star = np.array(p['stars'][i]); Si = p['S'][3 * i:3 * i + 3, 3 * i:3 * i + 3]
        residual = max(float(la.norm(U @ Si @ U.T - Si)) for U in cart)
        starerr = max(float(cKDTree(star).query(star @ U.T)[0].max()) for U in cart)
        w, v = la.eigh(T[i]); normal = v[:, 0] if w[0] < 1e-8 and w[1] > 1e-7 else None
        normal_eigen = normal is not None and la.norm(Si @ normal - normal * (normal @ Si @ normal)) < 1e-7
        sites.append(dict(vertex=i, bond_degree=int(f['cn'][i]), site_group_single=hm, site_group_composite=hc,
                          vector_decomposition=DECOMP.get(hc, ('unknown', 99))[0], max_irrep_multiplicity=DECOMP.get(hc, ('', 99))[1],
                          contact_count=len(star), contact_directions_cartesian=np.round(star, 12).tolist(),
                          S_eigenvalues=la.eigvalsh(Si).tolist(), T_eigenvalues=w.tolist(), planar=normal is not None,
                          normal_S_eigenvector=bool(normal_eigen), normal_stiffness=float(normal @ Si @ normal) if normal is not None else None,
                          S_commutator_residual=residual, contact_star_invariance_residual=starerr))
    return sites


def bipartition(g):
    adj = [[] for _ in range(g['n'])]
    for i, j, s in g['edges']: adj[i].append(j); adj[j].append(i)
    col = np.full(g['n'], -1, int)
    for root in range(g['n']):
        if col[root] >= 0: continue
        col[root] = 0; stack = [root]
        while stack:
            i = stack.pop()
            for j in adj[i]:
                if col[j] < 0: col[j] = 1 - col[i]; stack.append(j)
                elif col[j] == col[i]: return None
    return col
