"""Shared helpers for the post hoc diagnosis of the prospective flat pairs (SM S3.7): the composite in the
common primitive cell, vertex classes, mechanism kernels, local sectors and the eligible pairs of both
prospective tests. Imported by classifier/misses.py and classifier/sector_rule.py.
"""
import json
import sys
from fractions import Fraction
from pathlib import Path
import numpy as np
import scipy.linalg as la
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from common import composite as C
import flat_classifier as F
from pipeline import reduce_pair, spectral_stats

STRATIFIED = HERE / 'stratified_test'


def build(sc, fl, shift=None):
    """Composite in the common primitive cell, with the floppy net's symmetry operations."""
    p = reduce_pair(C.pair(sc, fl, shift))
    if 'classifier_graph' not in p:
        g = dict(p['fl']); g['symmetry_ops'] = [(np.eye(3), np.zeros(3))]; p['classifier_graph'] = g
    return p


def orbits(g):
    """Vertex orbits under the symmetry operations carried by the graph (union-find)."""
    n = g['n']; pts = np.asarray(g['pts']); parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for r, t in g.get('symmetry_ops', []):
        for i in range(n):
            y = (np.asarray(r) @ pts[i] + np.asarray(t)) % 1
            d = la.norm((pts - y + .5) % 1 - .5, axis=1); j = int(np.argmin(d))
            if d[j] < 1e-6: parent[find(i)] = find(j)
    roots = {}; cls = []
    for i in range(n):
        roots.setdefault(find(i), len(roots)); cls.append(roots[find(i)])
    return np.array(cls)


def classes(p):
    """Vertex classes: orbits refined by coordination, contact count and the eigenvalues of T_i and S_i."""
    g = p['classifier_graph']; bonds, T, S, normals = F.local_data(g, p['stars'])
    orb = orbits(g); keys = {}; cls = []
    for i in range(g['n']):
        k = (int(orb[i]), len(bonds[i]), len(p['stars'][i]), tuple(np.round(la.eigvalsh(T[i]), 5)), tuple(np.round(la.eigvalsh(S[i]), 5)))
        keys.setdefault(k, len(keys)); cls.append(keys[k])
    return np.array(cls), bonds, T, S, normals


def kernel(p, q):
    """Orthonormal basis of the mechanisms ker R_F(q)."""
    return C.null(C.compat(C.prep(p['fl']), q))[0]


def sector_of(vecs, tol=1e-7):
    """Orthonormal basis (columns) of the span of a set of 3-vectors (rows)."""
    if len(vecs) == 0: return np.zeros((3, 0))
    u, s, vh = la.svd(np.asarray(vecs), full_matrices=False)
    k = int(np.sum(s > tol * max(s[0], 1e-300)))
    return vh[:k].conj().T


def exact_fraction(x, maxden=200, tol=1e-9):
    """The fraction p/q (q <= maxden) equal to x within tol, else None. Used only to label exact values."""
    f = Fraction(float(x)).limit_denominator(maxden)
    return (str(f.numerator) if f.denominator == 1 else f'{f.numerator}/{f.denominator}') if abs(float(f) - x) < tol else None


def load(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def eligible_pairs(log=print):
    """Eligible pairs (rigid scaffold, every mechanism lifted) of the two prospective tests with their outcome.

    Test 1 (300 uniform pairs): stored predictions and outcomes in classifier/. Test 2 (1000 stratified pairs):
    stored predictions in classifier/stratified_test/predictions.json; outcomes from stratified_test/outcomes.json
    when present, otherwise computed here with pipeline.spectral_stats (same 32 wavevectors; an hour or more).
    Returns rows with test, id, scaffold, floppy, shift, flat, frozen_prediction, frozen_route."""
    rows = []
    preds = load(HERE / 'predictions_prospective.json')['rows']; outs = {o['id']: o for o in load(HERE / 'outcomes_prospective.json')['rows']}
    for r in preds:
        o = outs[r['id']]
        if o['qualified']:
            rows.append(dict(test='uniform_300', id=r['id'], scaffold=r['scaffold'], floppy=r['floppy'], shift=r['shift'],
                             flat=bool(o['shape_exact_flat']), frozen_prediction=r['prediction']['prediction'], frozen_route=r['prediction'].get('route')))
    preds = load(STRATIFIED / 'predictions.json')['rows']
    stored = STRATIFIED / 'outcomes.json'
    outs = {o['id']: o for o in load(stored)['rows']} if stored.exists() else {}
    if not outs:
        log('stratified outcomes not stored; computing the spectra of all 1000 pairs')
        from stratified import apply_input_corrections      # the tep record, as in the stratified test
        apply_input_corrections({x for r in preds for x in (r['scaffold'], r['floppy'])})
    for r in preds:
        o = outs.get(r['id'])
        if o is None: o = spectral_stats(reduce_pair(C.pair(r['scaffold'], r['floppy'], r['shift'])))
        if o['qualified']:
            rows.append(dict(test='stratified_1000', id=r['id'], scaffold=r['scaffold'], floppy=r['floppy'], shift=r['shift'],
                             flat=bool(o['shape_exact_flat']), frozen_prediction=r['prediction']['prediction'], frozen_route=r['prediction'].get('route')))
    return rows
