"""Retrospective and prospective tests of the flat-band classifier (SM Section S3.6).

flat_classifier.py is the classifier as frozen before the prospective test; this script only
prepares its inputs, draws the samples, evaluates the outcomes and counts them.

  python classifier/pipeline.py sample --rcsr DIR        # redraw the 300 prospective pairs
  python classifier/pipeline.py predict --rcsr DIR [--cohort prospective] [--ids 0 1 2]
  python classifier/pipeline.py evaluate --rcsr DIR [--cohort prospective] [--ids 0 1 2]
  python classifier/pipeline.py count                    # counts from the stored files

predict reads only the geometry (floppy net, contact stars, rank of R_F); evaluate computes the
lifted spectra. Prospective predictions use the common primitive cell; outcomes of pairs whose nets
share a space group are evaluated in that cell.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import scipy.linalg as la
from scipy.stats import beta
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from common import nets as E
from common import composite as C
from common.records import screen_records
import flat_classifier as classifier

SAMPLE_SEED = 20261121
OUTCOME_Q_SEED = 20261122


def classify_pair(p):
    if 'classifier_graph' in p: return classifier.classify(p['classifier_graph'], p['stars'])
    g = dict(p['fl']); ops, _ = E.sg_ops(C.CG()[g['name']], g['setting']); shift = p['shift']
    g['symmetry_ops'] = [(r, t + shift - r @ shift) for r, t in ops]
    return classifier.classify(g, p['stars'])


def reduce_pair(p):
    """Common primitive translation cell of a same-group pair, with the symmetry operations mapped."""
    f = p['fl']; s = p['sc']; CG = C.CG()
    if CG[f['name']]['group'] != CG[s['name']]['group']: return p
    ops, _ = E.sg_ops(CG[f['name']], f['setting']); _, B = E.sg_ops(CG[s['name']], s['setting'])
    out = C.primitive(p, B); g = dict(out['fl']); inv = la.inv(B.T); shift = p['shift']; newops = []; seen = set()
    for r, t in ops:
        rp = inv @ r @ B.T; tp = (inv @ (t + shift - r @ shift)) % 1
        tp[np.isclose(tp, 1, atol=1e-10)] = 0
        key = tuple(np.round(rp, 9).ravel()) + tuple(np.round(tp, 9))
        if key not in seen: seen.add(key); newops.append((rp, tp))
    g['symmetry_ops'] = newops; out['classifier_graph'] = g
    n = 3 * g['n']; out['S'] = np.zeros((n, n)); out['stars'] = [[] for _ in range(g['n'])]
    for i, j, sh, v, l in out['contacts']:
        out['S'][3 * i:3 * i + 3, 3 * i:3 * i + 3] += np.outer(v, v); out['stars'][i].append(v)
    out.update(shift=p['shift'], clearance=p['clearance'], cell_multiplicity=f['n'] // g['n'])
    return out


def retrospective_rows():
    """Screened dual pairs and non-dual controls, then the unbiased-control cages."""
    rows = []
    for idx, r in enumerate(screen_records()):
        sc = r.get('scaffold'); fl = r.get('floppy')
        if not sc:
            a, b = C.graph(r['a']), C.graph(r['b'])
            sc, fl = (r['a'], r['b']) if 3 * a['n'] - a['e'] <= 3 * b['n'] - b['e'] else (r['b'], r['a'])
        rows.append(dict(id=idx, cohort=r['cohort'], scaffold=sc, floppy=fl, shift=r.get('shift'), screen_index=idx))
    uc = json.loads((ROOT / 'controls' / 'unbiased_control.json').read_text(encoding='utf8'))['rows']
    for idx, r in enumerate(uc):
        shift = r.get('shift')
        if shift is None:
            shift = next((x for x in rows if x['scaffold'] == r['scaffold'] and x['floppy'] == r['floppy']), {}).get('shift')
        rows.append(dict(id=len(rows), cohort='unbiased_' + r['kind'], scaffold=r['scaffold'], floppy=r['floppy'], shift=shift, control_index=idx))
    return rows


def sample():
    """Uniform draw of 300 ordered pairs of distinct nets with identical space-group setting strings,
    excluding listed duals and every pair already screened. The first draw ran over all catalogue
    entries; its five draws containing a two-dimensional net were replaced by uniform draws from the
    remaining three-dimensional population before any prediction was scored."""
    CG = C.CG(); old = retrospective_rows()
    excluded = {tuple(sorted((r['scaffold'], r['floppy']))) for r in old}
    dual = {tuple(sorted(x)) for x in json.loads((ROOT / 'screen' / 'dual_pairs.json').read_text(encoding='utf8'))['pairs']}

    def population(three_d_only):
        groups = defaultdict(list)
        for n, g in CG.items():
            if not three_d_only or g.get('d') == 3: groups[g['group']].append(n)
        return [(s, f) for group, names in sorted(groups.items()) for s in sorted(names) for f in sorted(names)
                if s != f and tuple(sorted((s, f))) not in excluded and tuple(sorted((s, f))) not in dual]
    first = population(False)
    rng = np.random.default_rng(SAMPLE_SEED); chosen = rng.choice(len(first), 300, replace=False)
    rows = [dict(id=i, cohort='prospective', scaffold=first[j][0], floppy=first[j][1], shift=None) for i, j in enumerate(chosen)]
    bad = [r for r in rows if CG[r['scaffold']].get('d') != 3 or CG[r['floppy']].get('d') != 3]
    pool = population(True); kept = {(r['scaffold'], r['floppy']) for r in rows if r not in bad}
    remaining = [x for x in pool if x not in kept]
    rng = np.random.default_rng(np.random.SeedSequence([SAMPLE_SEED, 3])); ix = rng.choice(len(remaining), len(bad), replace=False)
    for r, j in zip(bad, ix):
        rows[r['id']] = dict(id=r['id'], cohort='prospective', scaffold=remaining[j][0], floppy=remaining[j][1], shift=None)
    return dict(seed=SAMPLE_SEED, population_pairs=len(pool), old_unordered_pairs_excluded=len(excluded),
                replaced_two_dimensional=bad, prospective=rows, retrospective=old)


def predict_row(r, cohort):
    out = dict(r)
    shift = r['shift']; clearance = None
    if shift is None:
        sg = C.graph(r['scaffold']); fg = C.graph(r['floppy'])
        shift, clearance = C.optimize_shift(C.remetric(sg, sg['A'] * sg['original_mean_length']), fg)
    p = C.pair(r['scaffold'], r['floppy'], shift); p['clearance'] = clearance
    out.update(shift=p['shift'], clearance=clearance, n_F=p['fl']['n'], n_S=p['sc']['n'], n_contacts=len(p['contacts']))
    if cohort == 'prospective': p = reduce_pair(p)
    out['classification_cell_multiplicity'] = p.get('cell_multiplicity', 1)
    out['classification_n_F'] = p['fl']['n']; out['prediction'] = classify_pair(p)
    return out


def spectral_stats(p):
    """32 random wavevectors (own seed). One relative threshold for rank, positive lifting and
    branch variation; the scale of the spectral tests is ||S||_2."""
    qs = np.random.default_rng(OUTCOME_Q_SEED).random((32, 3)); pf, ps = C.prep(p['fl']), C.prep(p['sc'])
    mf = []; ms = []; vals = []; un = []; scale = max(la.eigvalsh(p['S']))
    for q in qs:
        z = classifier.kernel(C.compat(pf, q)); s = la.svdvals(C.compat(ps, q), check_finite=False)
        rank = int(np.sum(s > classifier.TOL * s[0])) if len(s) and s[0] else 0
        w = la.eigvalsh(z.conj().T @ p['S'] @ z, check_finite=False)
        mf.append(z.shape[1]); ms.append(ps[-1] - rank); vals.append(w); un.append(int(np.sum(w <= classifier.TOL * scale)))
    out = dict(q_seed=OUTCOME_Q_SEED, nq=32, M_F=sorted(set(mf)), M_S=sorted(set(ms)), unlifted=sorted(set(un)),
               S_norm=float(scale), rigid=all(x == 0 for x in ms), fully_lifted=all(x == 0 for x in un) and min(mf) > 0,
               nu_F=3 * p['fl']['n'] - p['fl']['e'], nu_S=3 * p['sc']['n'] - p['sc']['e'])
    out['qualified'] = out['rigid'] and out['fully_lifted']
    if len(set(mf)) == 1 and mf[0]:
        w = np.array(vals); ranges = np.ptp(w, axis=0); spread = float(max(ranges) / scale) if scale else 0.
        out.update(branch_means=w.mean(0), branch_ranges=ranges, shape_spread_relative=spread, shape_exact_flat=spread <= classifier.TOL,
                   shape_exact_flat_2e_7=spread <= 2e-7, eig_min=float(w.min()), eig_max=float(w.max()),
                   mean_width_over_mean=float(np.mean(ranges) / max(abs(w.mean()), 1e-300)),
                   all_zero=bool(w.max() <= classifier.TOL * scale))
    else:
        out.update(shape_exact_flat=False, shape_spread_relative=None, all_zero=False)
    return out


def evaluate_row(r):
    out = dict(id=r['id'], cohort=r['cohort'], scaffold=r['scaffold'], floppy=r['floppy'])
    if 'error' in r: raise ValueError('prediction geometry failure: ' + r['error'])
    p = reduce_pair(C.pair(r['scaffold'], r['floppy'], r['shift']))
    out.update(cell_multiplicity=p.get('cell_multiplicity', 1),
               stats_cell='common primitive' if 'cell_multiplicity' in p else 'conventional')
    out.update(spectral_stats(p))
    return out


# ------------------------------------------------------------------ counting
def wilson(k, n):
    if not n: return None
    z = 1.959963984540054; p = k / n; den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den; half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(center - half), float(center + half)]


def upper(k, n, alpha=.05):
    if not n: return None
    return float(beta.ppf(1 - alpha, k + 1, n - k)) if k < n else 1.


def cohort_stats(rows):
    completed = [r for r in rows if 'error' not in r['outcome']]
    qualified = [r for r in completed if r['outcome']['qualified']]
    flat = [r for r in qualified if r['outcome']['shape_exact_flat']]
    calls = [r for r in qualified if r['prediction'].get('prediction') == 'predicted_flat']
    fp = [r for r in calls if not r['outcome']['shape_exact_flat']]; tp = len(calls) - len(fp)
    matrix = Counter(('predicted_flat' if r['prediction'].get('prediction') == 'predicted_flat' else 'abstain') + '__' +
                     ('flat' if r['outcome']['shape_exact_flat'] else 'nonflat') for r in qualified)
    for a in ('predicted_flat', 'abstain'):
        for b in ('flat', 'nonflat'): matrix.setdefault(a + '__' + b, 0)
    return dict(total=len(rows), completed=len(completed), errors=len(rows) - len(completed), qualified=len(qualified), actual_flat=len(flat),
                predicted_flat_qualified=len(calls), true_positive=tp, false_positive=len(fp), qualified_confusion=dict(matrix),
                coverage=tp / len(flat) if flat else None, coverage_wilson95=wilson(tp, len(flat)),
                qualified_call_rate=len(calls) / len(qualified) if qualified else None, call_rate_wilson95=wilson(len(calls), len(qualified)),
                false_prediction_upper95_one_sided=upper(len(fp), len(calls)), false_prediction_upper95_two_sided=upper(len(fp), len(calls), .025),
                conventional_false_positive_rate_denominator=len(qualified) - len(flat), false_positive_rate_upper95=upper(len(fp), len(qualified) - len(flat)),
                failures=[dict(id=r['id'], scaffold=r['scaffold'], floppy=r['floppy'], route=r['prediction'].get('route')) for r in fp],
                qualified_flat_abstentions=[dict(id=r['id'], scaffold=r['scaffold'], floppy=r['floppy'], reason=r['prediction'].get('reason')) for r in flat if r not in calls])


def count(directory=HERE):
    joined = []
    for cohort in ('retrospective', 'prospective'):
        preds = json.loads((Path(directory) / f'predictions_{cohort}.json').read_text(encoding='utf8'))['rows']
        outs = {o['id']: o for o in json.loads((Path(directory) / f'outcomes_{cohort}.json').read_text(encoding='utf8'))['rows']}
        joined += [dict(**r, outcome=outs[r['id']]) for r in preds if r['id'] in outs]
    cohorts = {c: cohort_stats([r for r in joined if r['cohort'] == c]) for c in sorted({r['cohort'] for r in joined})}
    retro = [r for r in joined if r['cohort'] != 'prospective']
    cohorts['retrospective_all_rows'] = cohort_stats(retro)
    unique = {}
    for r in retro: unique.setdefault((r['scaffold'], r['floppy']), r)
    cohorts['retrospective_unique_ordered_pairs'] = cohort_stats(list(unique.values()))
    return dict(cohorts=cohorts, interval_note='Wilson 95% coverage/call-rate; exact binomial upper bounds. Pair-wise intervals do not remove shared-net dependence or catalogue-selection limitations. Retrospective rows have known outcomes and are not prospective evidence.')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=['sample', 'predict', 'evaluate', 'count'])
    ap.add_argument('--cohort', default='prospective', choices=['prospective', 'retrospective'])
    ap.add_argument('--ids', nargs='*', type=int, help='row ids (default: all)')
    ap.add_argument('--out', default=str(ROOT / 'outputs' / 'classifier'))
    E.add_argument(ap); args = ap.parse_args(); E.apply_argument(args)
    out = Path(args.out)
    if args.stage == 'count':
        res = count(); C.dump(out / 'classifier_results.json', res)
        for k, v in res['cohorts'].items():
            print(k, {x: v[x] for x in ('total', 'qualified', 'actual_flat', 'predicted_flat_qualified', 'true_positive', 'false_positive')})
        return
    if args.stage == 'sample':
        s = sample(); C.dump(out / 'sampling.json', s)
        ref = json.loads((HERE / 'sampling.json').read_text(encoding='utf8'))['prospective']
        same = [(a['scaffold'], a['floppy']) for a in s['prospective']] == [(a['scaffold'], a['floppy']) for a in ref]
        print('population', s['population_pairs'], 'replaced', len(s['replaced_two_dimensional']), 'identical to stored sample:', same)
        return
    rows = json.loads((HERE / 'sampling.json').read_text(encoding='utf8'))[args.cohort]
    if args.stage == 'evaluate':   # use the shifts recorded with the predictions
        rows = json.loads((HERE / f'predictions_{args.cohort}.json').read_text(encoding='utf8'))['rows']
    if args.ids is not None: rows = [r for r in rows if r['id'] in set(args.ids)]
    res = []
    for r in rows:
        try:
            x = predict_row(r, args.cohort) if args.stage == 'predict' else evaluate_row(r)
        except Exception as e:
            x = dict(r, error=f'{type(e).__name__}: {e}')
        res.append(x)
        if args.stage == 'predict': print(r['id'], r['scaffold'], r['floppy'], x.get('prediction', {}).get('route'), x.get('error', ''), flush=True)
        else: print(r['id'], r['scaffold'], r['floppy'], 'qualified', x.get('qualified'), 'flat', x.get('shape_exact_flat'), x.get('error', ''), flush=True)
    name = ('predictions_' if args.stage == 'predict' else 'outcomes_') + args.cohort + '.json'
    C.dump(out / name, dict(cohort=args.cohort, rows=res)); print('written', out / name)


if __name__ == '__main__':
    main()
