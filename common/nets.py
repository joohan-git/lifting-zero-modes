"""RCSR nets: reader, space-group expansion and Bloch compatibility matrices.

The RCSR files are not part of this archive. Point the code to a directory that holds
rcsr3d.cgd and 3dall.txt (see rcsr_files.json) with --rcsr DIR or the RCSR_DIR variable.

q is a fractional reciprocal vector (Bloch phase exp(2*pi*i*q.shift)); graph lengths are
rescaled by the mean edge length.
"""
import os
for _k in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(_k, '1')
from functools import lru_cache
from pathlib import Path
import itertools
import re
import numpy as np
import scipy.linalg as la
from scipy.spatial import cKDTree
import gemmi

TOL = 2e-7
_DIR = [None]
PLANE_GROUPS = {'p1': 'P1', 'p2': 'P2', 'pm': 'P1m1', 'pg': 'P1c1', 'cm': 'C1m1',
                'p2mm': 'Pmm2', 'p2mg': 'Pmc21', 'p2gg': 'Pcc2', 'c2mm': 'Cmm2', 'p4': 'P4',
                'p4mm': 'P4mm', 'p4gm': 'P4bm', 'p3': 'P3', 'p3m1': 'P3m1', 'p31m': 'P31m',
                'p6': 'P6', 'p6mm': 'P6mm'}


def set_dir(path):
    """Directory with the user-supplied RCSR files."""
    _DIR[0] = Path(path)
    catalog.cache_clear()


def data_dir():
    d = _DIR[0] or (Path(os.environ['RCSR_DIR']) if os.environ.get('RCSR_DIR') else None)
    if d is None:
        raise SystemExit('RCSR files needed: pass --rcsr DIR or set RCSR_DIR (see README).')
    return d


def add_argument(ap):
    ap.add_argument('--rcsr', help='directory with rcsr3d.cgd and 3dall.txt (default: $RCSR_DIR)')


def apply_argument(args):
    if getattr(args, 'rcsr', None):
        set_dir(args.rcsr)


def read_cgd():
    out = {}; cur = None
    for line in (data_dir() / 'rcsr3d.cgd').read_text(encoding='utf8').splitlines():
        s = line.split()
        if not s:
            continue
        s[0] = s[0].upper()
        if s[0] == 'CRYSTAL':
            cur = {'nodes': [], 'reps': [], 'source': 'cgd'}
        elif s[0] == 'NAME':
            cur['name'] = s[1]
        elif s[0] == 'GROUP':
            cur['group'] = s[1]
        elif s[0] == 'CELL':
            cur['cell'] = list(map(float, s[1:])); cur['d'] = 2 if len(s) == 4 else 3
        elif s[0] == 'NODE':
            cur['nodes'].append((int(s[2]), np.array(list(map(float, s[3:])))))
        elif s[0] == 'EDGE':
            cur['reps'].append(np.array(list(map(float, s[1:]))).reshape(2, -1))
        elif s[0] == 'END':
            out[cur['name']] = cur; cur = None
    return out


@lru_cache(None)
def catalog():
    return read_cgd()


def read_all(d=3):
    """Metadata of 3dall.txt (tiling symbol and listed dual of each net)."""
    out = {}
    for block in (data_dir() / f'{d}dall.txt').read_text(encoding='utf8').split('start\n')[1:]:
        s = [x.strip() for x in block.strip().splitlines()]
        if len(s) < 3:
            continue
        name = s[1]; i = 3
        rec = {'name': name, 'd': d, 'nodes': [], 'centers': []}
        try:
            for key in ['symbols', 'names', 'other_names', 'keywords', 'refs']:
                n = int(s[i].split()[0]); rec[key] = s[i + 1:i + 1 + n]; i += n + 1
            rec['group'] = s[i].split()[0].replace('(', '').replace(')', ''); i += 1
            rec['cell'] = list(map(float, s[i].split())); i += 1
            n = int(s[i]); i += 1
            for _ in range(n):
                rec['nodes'].append((int(s[i].split()[1]), np.array(list(map(float, s[i + 1].split()))))); i += 6
            n = int(s[i]); i += 1
            for _ in range(n):
                rec['centers'].append(np.array(list(map(float, s[i + 1].split())))); i += 5
            rec['tiling'] = s[i + 3]; rec['dual'] = s[i + 4]
            out[name] = rec
        except Exception as e:
            rec['parse_error'] = f'{type(e).__name__}: {e}'; out[name] = rec
    return out


def canon(i, j, s):
    s = tuple(int(x) for x in s)
    return min((i, j, s), (j, i, tuple(-x for x in s)))


def sg_ops(rec, setting=None):
    d = rec['d']; g = rec['group']
    if d == 2:
        if g in ('p2', 'pm', 'pg', 'cm', 'p2mg', 'p2gg'):
            I = np.eye(2); r = -I if g == 'p2' else np.diag([-1, 1])
            ops = [(I, np.zeros(2)), (r, np.array([0, .5]) if g == 'pg' else np.zeros(2))]
            if g in ('p2mg', 'p2gg'):
                t = np.array([.5, 0]) if g == 'p2mg' else np.array([.5, .5])
                ops = [(I, np.zeros(2)), (-I, np.zeros(2)), (np.diag([-1, 1]), t), (np.diag([1, -1]), t)]
            if g == 'cm':
                ops += [(a, b + 0.5) for a, b in list(ops)]
            return ops, np.array([[.5, .5], [.5, -.5]]) if g == 'cm' else I
        sg = gemmi.find_spacegroup_by_name(PLANE_GROUPS[g])
    else:
        sg = gemmi.find_spacegroup_by_name(g + (setting or ''))
    if sg is None:
        raise ValueError('spacegroup ' + g)
    ops = []
    for o in sg.operations():
        a = np.array(o.rot, float)[:d, :d] / o.DEN; b = np.array(o.tran, float)[:d] / o.DEN
        if not any(np.allclose(a, c) and np.allclose(b, e) for c, e in ops):
            ops.append((a, b))
    if d == 2:
        B = np.array([[.5, .5], [.5, -.5]]) if g.startswith('c') else np.eye(2)
    else:
        o = sg.centred_to_primitive(); B = (np.array(o.rot, float) / o.DEN).T
    return ops, B


def build(rec, primitive=True):
    errors = []
    settings = [None] if ':' in rec['group'] or rec['d'] == 2 else [None, ':2']
    for setting in settings:
        try:
            g = _build(rec, setting, primitive)
            if g['ok']:
                return g
            errors.append(g.get('reason', 'degree mismatch'))
        except Exception as e:
            errors.append(str(e))
    raise ValueError(' | '.join(errors))


def _build(rec, setting, primitive):
    d = rec['d']; ops, B = sg_ops(rec, setting); points = []; cns = []
    tol = 0.00015
    for cn, x in rec['nodes']:
        x = x.copy()
        close = [((r @ x + t - x + .5) % 1 - .5) for r, t in ops if np.linalg.norm((r @ x + t - x + .5) % 1 - .5) < tol]
        x = x + np.mean(close, axis=0)
        for r, t in ops:
            p = (r @ x + t) % 1; p[np.isclose(p, 1, atol=1e-9)] = 0
            if not points or min(np.linalg.norm((np.array(points) - p + .5) % 1 - .5, axis=1)) > tol:
                points.append(p); cns.append(cn)
    pts = np.array(points); cn = np.array(cns); n = len(pts)
    tree = cKDTree(pts, boxsize=1)
    if d == 3:
        A = np.array(gemmi.UnitCell(*rec['cell']).orth.mat).T
    else:
        a, b, ang = rec['cell']; a0 = np.deg2rad(ang); A = np.array([[a, 0], [b * np.cos(a0), b * np.sin(a0)]])

    def locate(x):
        dist, k = tree.query(x % 1)
        if dist > tol * 3:
            raise ValueError(f'endpoint unmatched {dist:.3g}')
        return int(k), np.rint(x - pts[k]).astype(int)
    es = set(); orbits = {}
    for orb, (a, b) in enumerate(rec['reps']):
        for r, t in ops:
            i, si = locate(r @ a + t); j, sj = locate(r @ b + t)
            if i == j and np.all(si == sj):
                raise ValueError('zero edge')
            edge = canon(i, j, sj - si); es.add(edge); orbits.setdefault(edge, orb)
    es = sorted(es); deg = np.zeros(n, int)
    for i, j, s in es:
        deg[i] += 1; deg[j] += 1
    if not np.all(deg == cn):
        return {'ok': False, 'reason': f'degrees N={n} E={len(es)}; got {sorted(set(deg))}, expected {sorted(set(cn))}'}
    conventional_n = n; conventional_e = len(es)
    if primitive and abs(np.linalg.det(B)) < .999:
        inv = np.linalg.inv(B); newpts = []; newcn = []; mapping = []
        for i, p in enumerate(pts @ inv):
            f = p % 1; f[np.isclose(f, 1, atol=1e-9)] = 0
            match = [j for j, x in enumerate(newpts) if np.linalg.norm((f - x + .5) % 1 - .5) < tol]
            if match:
                j = match[0]
            else:
                j = len(newpts); newpts.append(f); newcn.append(cn[i])
            mapping.append((j, np.rint(p - newpts[j]).astype(int)))
        ee = set(); neworbits = {}
        for i, j, s in es:
            ii, si = mapping[i]; jj, sj = mapping[j]; edge = canon(ii, jj, sj + np.rint(np.array(s) @ inv).astype(int) - si)
            ee.add(edge); neworbits.setdefault(edge, orbits[(i, j, s)])
        es = sorted(ee); orbits = neworbits; pts = np.array(newpts); cn = np.array(newcn); A = B @ A; n = len(pts)
    deg = np.zeros(n, int)
    for i, j, s in es:
        deg[i] += 1; deg[j] += 1
    if not np.all(deg == cn):
        raise ValueError('primitive degree mismatch')
    vec = np.array([(pts[j] + s - pts[i]) @ A for i, j, s in es]); ell = np.linalg.norm(vec, axis=1); scale = ell.mean()
    return {'name': rec['name'], 'd': d, 'pts': pts, 'A': A / scale, 'edges': es, 'cn': cn, 'ok': True,
            'unit': vec / ell[:, None], 'length': ell / scale, 'n': n, 'e': len(es), 'setting': setting,
            'conventional_n': conventional_n, 'conventional_e': conventional_e,
            'edge_orbits': np.array([orbits[x] for x in es]), 'original_mean_length': float(scale)}


def matrices(g, q):
    """Bloch compatibility matrix R(q) (edges x d*vertices)."""
    d, n, e = g['d'], g['n'], g['e']; R = np.zeros((e, d * n), complex)
    for b, (i, j, s) in enumerate(g['edges']):
        z = np.exp(2j * np.pi * np.dot(q, s)); v = g['unit'][b]
        R[b, d * i:d * i + d] -= v; R[b, d * j:d * j + d] += z * v
    return R, None


def rank(R):
    s = la.svdvals(R, check_finite=False); return int(np.sum(s > TOL * max(1, s[0] if len(s) else 1)))


def kernel(R):
    _, s, v = la.svd(R, full_matrices=True, check_finite=False); r = int(np.sum(s > TOL * max(1, s[0] if len(s) else 1)))
    return v[r:].conj().T
