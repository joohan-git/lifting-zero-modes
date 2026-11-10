"""Primitive-cell layer of the second implementation: primitive reduction for centred groups,
placement with a spatial index (same rule as engine.place: conventional 8^3 shift grid, first maximum,
1.05 x nearest contact shell), and a geometry-only stage that runs before any lifted spectrum.
"""
import itertools, numpy as np, gemmi
from scipy.spatial import cKDTree
import engine as V

def centering(group):
    sg = gemmi.find_spacegroup_by_name(group)
    return [np.array(c, float)/24.0 for c in sg.operations().cen_ops]

def primitive_basis(group):
    """Rows of P = primitive lattice vectors in conventional fractional coordinates (P = I for primitive groups)."""
    cen = [c for c in centering(group) if c.any()]
    if not cen: return np.eye(3)
    n = len(cen) + 1
    cands = [np.eye(3)[k] for k in range(3)] + cen + [c - np.eye(3)[k] for c in cen for k in range(3)]
    best = None
    for tri in itertools.combinations(range(len(cands)), 3):
        P = np.array([cands[t] for t in tri]); d = np.linalg.det(P)
        if abs(abs(d) - 1.0/n) < 1e-9:
            # all centring vectors and conventional vectors must be integer in the new basis
            X = np.vstack([np.eye(3)] + cen) @ np.linalg.inv(P)
            if np.allclose(X, np.rint(X), atol=1e-9):
                score = (np.sum(P*P), tuple(np.round(P.ravel(), 6)))
                if best is None or score < best[0]: best = (score, P if d > 0 else P[[1, 0, 2]])
    assert best is not None, 'no primitive basis found for ' + group
    return best[1]

def reduce_primitive(g, P):
    """Reduce a conventional-cell graph (from vengine.build) to the primitive lattice with basis rows P."""
    Pi = np.linalg.inv(P)
    xp = V.wrap(g['pts'] @ Pi)
    keep = []; rep = -np.ones(g['V'], int)
    for i in range(g['V']):
        if keep:
            d = V.dfrac(xp[keep], xp[i]); k = int(np.argmin(d))
            if d[k] < 10*V.TOL_FRAC: rep[i] = keep[k]; continue
        rep[i] = i; keep.append(i)
    idx = {old: new for new, old in enumerate(keep)}
    pts = xp[keep]; cn = g['cn'][keep]
    def locate(x):
        d = V.dfrac(pts, x); k = int(np.argmin(d))
        if d[k] > 10*V.TOL_FRAC: raise ValueError(f'primitive endpoint unmatched {d[k]:.2e}')
        return k, np.rint(x - pts[k]).astype(int)
    es = set()
    for i, j, s in g['edges']:
        a = g['pts'][i] @ Pi; b = (g['pts'][j] + np.array(s)) @ Pi
        ii, si = locate(a); jj, sj = locate(b); sh = sj - si
        e = (ii, jj, tuple(int(v) for v in sh)); e2 = (jj, ii, tuple(int(-v) for v in sh))
        es.add(min(e, e2))
    es = sorted(es)
    deg = np.zeros(len(pts), int)
    for i, j, s in es: deg[i] += 1; deg[j] += 1
    if not np.array_equal(deg, cn): raise ValueError('primitive degree mismatch')
    mult = g['V']//len(pts)
    assert mult*len(pts) == g['V'] and mult*len(es) == g['E'], 'multiplicity mismatch'
    return dict(name=g['name'], group=g['group'], pts=pts, edges=es, cn=cn, A=P @ g['A'], V=len(pts), E=len(es), P=P, mult=mult)

def seg_dist(Q, U, W):
    """Distances from points Q (n,3) to segments U[k]..W[k] given per-point candidate index arrays (n,K)."""
    AB = W - U; AB2 = np.einsum('...i,...i->...', AB, AB)
    T = np.clip(np.einsum('...i,...i->...', Q[:, None, :] - U, AB)/AB2, 0, 1)
    Pp = U + T[..., None]*AB
    return np.linalg.norm(Q[:, None, :] - Pp, axis=-1)

SH2 = np.array(list(itertools.product(range(-2, 3), repeat=3)))

def place2(sc, fl, grid=8, shift=None, K=12):
    """Same placement rule as vengine.place, on primitive representations sc, fl (same P). Shift grid is the
    conventional 8^3 grid; floppy fractional coordinates (conventional) are kept in the scaffold cell."""
    A = sc['A']; P = sc['P']; Pi = np.linalg.inv(P)
    Psc = sc['pts']; Pfl = fl['pts']
    # scaffold vertex images and bar images within +-2 primitive shifts
    Sc = (Psc[None, :, :] + SH2[:, None, :]).reshape(-1, 3) @ A
    Sc_k = np.tile(np.arange(sc['V']), len(SH2)); Sc_sh = np.repeat(SH2, sc['V'], axis=0)
    U = np.array([Psc[i] @ A for i, j, s in sc['edges']]); W = np.array([(Psc[j] + np.array(s)) @ A for i, j, s in sc['edges']])
    SegU = (U[None, :, :] + (SH2 @ A)[:, None, :]).reshape(-1, 3); SegW = (W[None, :, :] + (SH2 @ A)[:, None, :]).reshape(-1, 3)
    Mid = 0.5*(SegU + SegW); Lmax2 = 0.5*np.linalg.norm(SegW - SegU, axis=1).max()
    tv = cKDTree(Sc); tm = cKDTree(Mid)
    Kq = min(K + 1, len(Mid))
    def clearance_batch(ts):
        Q = V.wrap(Pfl[None, :, :] + ts[:, None, :]) @ A       # (nt, VF, 3) primitive frac -> cart
        n_t = len(ts); Qf = Q.reshape(-1, 3)
        dv, _ = tv.query(Qf)
        dm, im = tm.query(Qf, k=Kq)
        de = seg_dist(Qf, SegU[im[:, :K]], SegW[im[:, :K]]).min(axis=1)
        bound = dm[:, K] - Lmax2 if Kq > K else np.inf
        bad = np.where(de > bound)[0]
        Kb = K
        while len(bad):   # rigorous fallback: widen the candidate set for the points whose bound is not met
            Kb = min(4*Kb, len(Mid))
            dmb, imb = tm.query(Qf[bad], k=min(Kb + 1, len(Mid)))
            deb = seg_dist(Qf[bad], SegU[imb[:, :Kb]], SegW[imb[:, :Kb]]).min(axis=1)
            de[bad] = np.minimum(de[bad], deb)
            if Kb >= len(Mid): break
            bad = bad[deb > dmb[:, Kb] - Lmax2]
        return np.minimum(dv, de).reshape(n_t, -1).min(axis=1)
    if shift is None:
        tconv = np.array(list(itertools.product(range(grid), repeat=3)))/grid
        tprim = tconv @ Pi
        c = clearance_batch(tprim)
        best = 0
        for k in range(1, len(c)):
            if c[k] > c[best] + 1e-9: best = k
        t = tprim[best]; tc = tconv[best]; cbest = float(c[best])
    else:
        tc = np.array(shift, float); t = tc @ Pi; cbest = float(clearance_batch(t[None])[0])
    # contacts: unwrapped floppy positions, scaffold images within +-2 primitive shifts (KD-tree)
    contacts = []
    Qf = (Pfl + t) @ A
    kk = min(64, len(Sc))
    dd, ii = tv.query(Qf, k=kk)
    for i in range(fl['V']):
        dn = dd[i, 0]
        sel = ii[i][dd[i] < 1.05*dn]
        if len(sel) >= kk:   # fallback when the shell exceeds the k = 64 query: all images for this vertex
            dall = np.linalg.norm(Sc - Qf[i], axis=1); sel = np.where(dall < 1.05*dall.min())[0]
        for m in sel:
            dvec = Sc[m] - Qf[i]; d = np.linalg.norm(dvec)
            contacts.append((i, int(Sc_k[m]), tuple(int(v) for v in Sc_sh[m]), dvec/d, float(d)))
    return dict(A=A, shift=t, shift_conv=tc, clearance=cbest, contacts=contacts)

def build_pair(cg, scaffold, floppy):
    """Scaffold and floppy on the common primitive lattice (same group setting -> same P), physical cell = scaffold's."""
    sc0 = V.build(cg[scaffold]); fl0 = V.build(cg[floppy])
    P = primitive_basis(sc0['group'])
    sc = reduce_primitive(sc0, P); fl = reduce_primitive(fl0, P)
    fl['A'] = sc['A']    # floppy fractional coordinates kept in the scaffold cell
    return sc, fl

def vertex_geometry(fl, unitF):
    V_ = fl['V']; T = np.zeros((V_, 3, 3)); stars = [[] for _ in range(V_)]
    for b, (i, j, s) in enumerate(fl['edges']):
        stars[i].append(unitF[b]); stars[j].append(-unitF[b])
    normals = [None]*V_; planar = np.zeros(V_, bool)
    for i in range(V_):
        B = np.array(stars[i]); T[i] = B.T @ B
        w, Uu = np.linalg.eigh(T[i])
        if w[0] < 1e-9: planar[i] = True; normals[i] = Uu[:, 0]
    return T, planar, normals

def predict_rules(fl, S, T, planar, normals, MF):
    """Rule N (local normals, M_F == n_planar) and rule B (bipartite isotropic); abstain otherwise."""
    rec = {}
    nP = int(planar.sum())
    if nP > 0 and MF == nP:
        vals = sorted(float(normals[i] @ S[3*i:3*i+3, 3*i:3*i+3] @ normals[i]) for i in range(fl['V']) if planar[i])
        par = [bool(np.linalg.norm(S[3*i:3*i+3, 3*i:3*i+3] @ normals[i] - vals_i*normals[i]) < 1e-9)
               for i, vals_i in ((i, float(normals[i] @ S[3*i:3*i+3, 3*i:3*i+3] @ normals[i])) for i in range(fl['V']) if planar[i])]
        rec['rule_N'] = dict(predict='flat', values=np.round(vals, 9).tolist(), Sn_parallel_n_all=all(par))
    elif nP > 0 and MF > nP: rec['rule_N'] = dict(predict='abstain', reason=f'M_F={MF} > n_planar={nP}')
    elif nP > 0 and MF < nP: rec['rule_N'] = dict(predict='INCONSISTENT', reason=f'M_F={MF} < n_planar={nP}')
    else: rec['rule_N'] = dict(predict='abstain', reason='no planar vertex')
    col = -np.ones(fl['V'], int); col[0] = 0; adj = {}
    for i, j, s in fl['edges']: adj.setdefault(i, []).append(j); adj.setdefault(j, []).append(i)
    stack = [0]; bip = True
    while stack:
        i = stack.pop()
        for j in adj.get(i, []):
            if col[j] < 0: col[j] = 1 - col[i]; stack.append(j)
            elif col[j] == col[i]: bip = False
    rec['bipartite'] = bool(bip and (col >= 0).all())
    if rec['bipartite']:
        ok = True; ts = {}; ss = {}
        iso = lambda M: np.allclose(M, np.trace(M)/3*np.eye(3), atol=1e-8)
        for c in (0, 1):
            idx = [i for i in range(fl['V']) if col[i] == c]
            Tc = [T[i] for i in idx]; Sc_ = [S[3*i:3*i+3, 3*i:3*i+3] for i in idx]
            if not all(iso(M) for M in Tc + Sc_): ok = False; break
            tv = [np.trace(M)/3 for M in Tc]; sv = [np.trace(M)/3 for M in Sc_]
            if np.ptp(tv) > 1e-8 or np.ptp(sv) > 1e-8: ok = False; break
            ts[c] = tv[0]; ss[c] = sv[0]
        if ok:
            rec['rule_B'] = dict(predict='flat', value=float((ts[1]*ss[0] + ts[0]*ss[1])/(ts[0] + ts[1])), t=ts, s=ss)
        else: rec['rule_B'] = dict(predict='abstain', reason='classes not isotropic/identical')
    else: rec['rule_B'] = dict(predict='abstain', reason='not bipartite')
    rec['predicted_flat'] = rec['rule_N']['predict'] == 'flat' or rec.get('rule_B', {}).get('predict') == 'flat'
    rec['n_planar'] = nP
    return rec

def prepare(cg, scaffold, floppy, shift=None):
    """Geometry-only stage: build, reduce, place, contacts, S, generic M_F. No lifted spectrum."""
    sc, fl = build_pair(cg, scaffold, floppy)
    unitS, _ = V.bond_geometry(sc)
    Pl = place2(sc, fl, shift=shift)
    unitF, _ = V.bond_geometry(fl, Pl['A'])
    S = V.contact_tensor(fl, Pl['contacts'])
    MF = V.generic_mech(fl, unitF)
    return dict(sc=sc, fl=fl, P=Pl, S=S, unitF=unitF, unitS=unitS, MF=MF)

def nq_for(fl):
    n = 3*fl['V']
    return 40 if n <= 600 else (20 if n <= 1200 else 12)

def spectra(pre, nq=None, seed=0):
    """Scaffold rigidity, fully-lifted test, fixed-scaffold lifted bands at random q."""
    sc, fl, Pl, S, unitF, unitS = pre['sc'], pre['fl'], pre['P'], pre['S'], pre['unitF'], pre['unitS']
    mS = V.generic_mech(sc, unitS)
    nq = nq or nq_for(fl)
    rng = np.random.default_rng(seed); evs = []; zeros = []; nker = []
    for _ in range(nq):
        w = V.lifted_fixed(fl, S, unitF, rng.random(3))
        nker.append(len(w))
        if len(w) == 0: continue
        evs.append(np.sort(w)); zeros.append(int(np.sum(w < 1e-8)))
    out = dict(mech_S=mS, mech_F=pre['MF'], nq=nq, n_contacts=len(Pl['contacts']), clearance=Pl['clearance'],
               shift_conv=Pl['shift_conv'].tolist(), V_S=sc['V'], E_S=sc['E'], V_F=fl['V'], E_F=fl['E'], mult=sc['mult'],
               nu_F=3*fl['V'] - fl['E'])
    if evs and len({len(e) for e in evs}) == 1:
        ev = np.array(evs); m = max(float(np.mean(ev)), 1e-300)
        out.update(unlifted_median=int(np.median(zeros)), eig_min=float(ev.min()), eig_max=float(ev.max()),
                   spread_rel=float(np.mean(ev.max(0) - ev.min(0))/m), bands=np.round(ev.mean(0), 6).tolist(),
                   S_norm2=float(np.linalg.norm(S, 2)),
                   spread_S=float(np.max(ev.max(0) - ev.min(0))/max(np.linalg.norm(S, 2), 1e-300)))
    elif evs:
        out.update(unlifted_median=int(np.median(zeros)), kernel_dims=sorted(set(nker)), spread_rel=None)
    else:
        out.update(unlifted_median=0, spread_rel=0.0, bands=[])
    return out
