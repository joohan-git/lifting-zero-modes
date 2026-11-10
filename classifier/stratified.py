"""Second prospective test of the frozen classifier: 1,000 pairs stratified by crystal system (SM S3.6).

The frozen classifier (flat_classifier.py) is used unchanged. The order of the stages was fixed:
the sample was drawn and sealed first, then every prediction was made and sealed, and only then
were the lifted spectra computed. stratified_test/freezing.json records that order.

  python classifier/stratified.py sample --rcsr DIR                 # redraw the 1,000 pairs
  python classifier/stratified.py predict --rcsr DIR [--ids 0 1 2]  # geometry only
  python classifier/stratified.py evaluate --rcsr DIR [--ids 0 1 2] # lifted spectra
  python classifier/stratified.py count                             # counts from the stored files
  python classifier/stratified.py routes --rcsr DIR                 # on-site detail of the U and C calls

Sampling: within each crystal-system stratum (cubic, space groups 195-230; hexagonal, 168-194;
tetragonal, 75-142), a uniform draw of unordered pairs of distinct three-dimensional nets with the
same space-group setting string, then a random fixed orientation (scaffold, floppy net). Pairs
in any earlier cohort and listed dual pairs are excluded; nothing else is filtered.
Placement, contacts, primitive-cell reduction and spectra are those of pipeline.py.

Input correction: the RCSR record of tep gives its vertices as ATOM lines and its edges as
'EDGE atom-id x y z'. The reader in common/nets.py takes NODE lines and six-coordinate EDGE lines
only, so tep had no vertices and its prediction failed. apply_input_corrections() rebuilds such
records from the same file before any stage runs; roles, pair and coordinates are unchanged
(stratified_test/input_correction.json). In the original run the few largest cells were
evaluated wavevector by wavevector to fit a time limit; the statistic is the same as here.
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import scipy.linalg as la
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from common import nets as E
from common import composite as C
import flat_classifier as classifier
from pipeline import classify_pair, reduce_pair, spectral_stats, wilson, upper

DATA = HERE / 'stratified_test'
SEED = 20261131
STRATA = [('cubic', 195, 230, 334), ('hexagonal', 168, 194, 333), ('tetragonal', 75, 142, 333)]
ROUTES = ['L', 'U', 'C', 'mixed_L_P2']


def load(name):
    return json.loads((DATA / name).read_text(encoding='utf8'))


# ------------------------------------------------------------------ input
def apply_input_corrections(names=None):
    """Rebuild catalogue records that carry no NODE lines but ATOM / 'EDGE id x y z' lines.
    Returns the corrected names (restricted to `names` if given)."""
    CG = C.CG(); current = None; lines = []; corrected = []
    for line in (E.data_dir() / 'rcsr3d.cgd').read_text(encoding='utf8').splitlines():
        f = line.split()
        if not f: continue
        key = f[0].upper()
        if key == 'CRYSTAL': current = None; lines = []
        elif key == 'NAME': current = f[1]
        elif key == 'END':
            rec = CG.get(current)
            if rec is not None and not rec['nodes'] and (names is None or current in names):
                atoms = {int(x[1]): (int(x[2]), np.array(list(map(float, x[3:])))) for x in lines if x[0].upper() == 'ATOM'}
                edges = [x for x in lines if x[0].upper() == 'EDGE']
                if atoms and edges and all(len(x) == 5 for x in edges):
                    rec['nodes'] = list(atoms.values())
                    rec['reps'] = [np.vstack([atoms[int(x[1])][1], np.array(list(map(float, x[2:])))]) for x in edges]
                    corrected.append(current)
        else: lines.append(f)
    if corrected: C.graph.cache_clear()
    return corrected


# ------------------------------------------------------------------ stages
def sample():
    import gemmi
    stored = load('sample.json')
    seen = {tuple(sorted(x)) for x in stored['excluded_unordered_pairs']}
    dual = {tuple(sorted(x)) for x in json.loads((ROOT / 'screen' / 'dual_pairs.json').read_text(encoding='utf8'))['pairs']}
    groups = defaultdict(list)
    for name, g in sorted(C.CG().items()):
        if g.get('d') == 3: groups[g['group']].append(name)
    rng = np.random.default_rng(SEED); rows = []; strata = {}
    for label, lo, hi, number in STRATA:
        pool = []
        for setting, names in sorted(groups.items()):
            sg = gemmi.find_spacegroup_by_name(setting)
            if not lo <= sg.number <= hi: continue
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    if (a, b) not in seen and (a, b) not in dual: pool.append((a, b, setting, sg.number))
        for ix in rng.choice(len(pool), number, replace=False):
            a, b, setting, sgno = pool[ix]
            sc, fl = (a, b) if rng.integers(2) == 0 else (b, a)
            rows.append(dict(id=len(rows), stratum=label, space_group_number=sgno, setting=setting, scaffold=sc, floppy=fl))
        strata[label] = dict(space_group_range=[lo, hi], population_unordered_pairs=len(pool), n=number)
    return dict(seed=SEED, n=len(rows), strata=strata, rows=rows)


def predict_row(r):
    out = dict(r)
    sg, fg = C.graph(r['scaffold']), C.graph(r['floppy'])
    shift, clearance = C.optimize_shift(C.remetric(sg, sg['A'] * sg['original_mean_length']), fg)
    p = C.pair(r['scaffold'], r['floppy'], shift)
    out.update(shift=shift, clearance=clearance, conventional_n_F=p['fl']['n'], conventional_n_S=p['sc']['n'])
    p['clearance'] = clearance; p = reduce_pair(p)
    out.update(cell_multiplicity=p.get('cell_multiplicity', 1), n_F=p['fl']['n'], n_S=p['sc']['n'], prediction=classify_pair(p))
    return out


def evaluate_row(r):
    out = {k: r[k] for k in ('id', 'stratum', 'space_group_number', 'setting', 'scaffold', 'floppy')}
    if 'error' in r: raise ValueError('prediction geometry failure: ' + r['error'])
    p = reduce_pair(C.pair(r['scaffold'], r['floppy'], r['shift']))
    out.update(spectral_stats(p))
    pr = r['prediction']
    if pr['prediction'] == 'predicted_flat' and 'branch_means' in out and len(pr['values']) == len(out['branch_means']):
        out['prediction_value_max_error_over_S'] = float(np.max(np.abs(np.asarray(pr['values']) - out['branch_means'])) / out['S_norm'])
    return out


def clusters(values, scale):
    groups = []
    for x in sorted(values):
        if not groups or abs(x - np.mean(groups[-1])) > 1e-8 * max(scale, 1e-300): groups.append([float(x)])
        else: groups[-1].append(float(x))
    return [dict(value=float(np.mean(g)), multiplicity=len(g)) for g in groups]


def route_details(preds, outs):
    """On-site eigenvalues of S for the route-U and route-C calls (made after sealing; predictions unchanged)."""
    rows = []
    for r in preds:
        if r['prediction']['route'] not in ('U', 'C'): continue
        p = reduce_pair(C.pair(r['scaffold'], r['floppy'], r['shift'])); S = p['S']
        eig = clusters(la.eigvalsh(S), float(la.norm(S, 2)))
        for x in eig: x['count_lower_bound'] = max(0, x['multiplicity'] - p['fl']['e'])
        o = outs.get(r['id'], {})
        rows.append(dict(id=r['id'], scaffold=r['scaffold'], floppy=r['floppy'], route=r['prediction']['route'],
                         onsite=eig, globally_scalar_S=len(eig) == 1, N_L=len(S), E=p['fl']['e'], predicted_values=r['prediction']['values'],
                         M_F=o.get('M_F'), M_S=o.get('M_S'), qualified=o.get('qualified'),
                         conditional_fixed_scaffold_shape=o.get('shape_exact_flat'),
                         prediction_value_error_over_S=o.get('prediction_value_max_error_over_S')))
    return rows


# ------------------------------------------------------------------ counting
def stats(rows):
    done = [r for r in rows if r.get('outcome') and 'error' not in r['outcome']]
    qualified = [r for r in done if r['outcome']['qualified']]
    flat = [r for r in qualified if r['outcome']['shape_exact_flat']]
    calls = [r for r in qualified if r['prediction'].get('prediction') == 'predicted_flat']
    true = [r for r in calls if r['outcome']['shape_exact_flat']]; false = [r for r in calls if not r['outcome']['shape_exact_flat']]
    confusion = {}
    for pred in ('predicted_flat', 'abstain'):
        for actual in ('flat', 'nonflat'):
            confusion[pred + '__' + actual] = sum((r['prediction'].get('prediction') == 'predicted_flat') == (pred == 'predicted_flat')
                                                  and r['outcome']['shape_exact_flat'] == (actual == 'flat') for r in qualified)
    routes = []
    for route in ROUTES:
        allcalls = [r for r in rows if r['prediction'].get('route') == route]
        qc = [r for r in calls if r['prediction']['route'] == route]
        tp = sum(r['outcome']['shape_exact_flat'] for r in qc); fp = len(qc) - tp
        routes.append(dict(route=route, designated_all=len(allcalls), designated_qualified=len(qc), hit=tp, false_designation=fp,
                           outside_qualification=sum(r in done and not r['outcome']['qualified'] for r in allcalls),
                           false_designation_upper95_one_sided=upper(fp, len(qc))))
    return dict(n=len(rows), outcomes_present=sum(r.get('outcome') is not None for r in rows), completed=len(done),
                prediction_errors=sum('error' in r for r in rows),
                outcome_errors=sum(bool(r.get('outcome')) and 'error' in r['outcome'] for r in rows),
                qualified=len(qualified), rigid=sum(r['outcome']['rigid'] for r in done), flat=len(flat),
                designated_qualified=len(calls), hit=len(true), false_designation=len(false), confusion=confusion,
                coverage=len(true) / len(flat) if flat else None, coverage_wilson95=wilson(len(true), len(flat)),
                false_designation_upper95_one_sided=upper(len(false), len(calls)),
                false_designation_upper95_two_sided=upper(len(false), len(calls), .025),
                nonflat_denominator=len(qualified) - len(flat), false_positive_rate_upper95_one_sided=upper(len(false), len(qualified) - len(flat)),
                routes=routes, abstained_flat=[{k: r[k] for k in ('id', 'scaffold', 'floppy', 'stratum')} for r in flat if r not in calls])


def joined(directory=DATA):
    preds = json.loads((Path(directory) / 'predictions.json').read_text(encoding='utf8'))['rows']
    outs = {o['id']: o for o in json.loads((Path(directory) / 'outcomes.json').read_text(encoding='utf8'))['rows']}
    return [dict(r, outcome=outs.get(r['id'])) for r in preds]


def count(directory=DATA):
    rows = joined(directory); reg = json.loads((Path(directory) / 'sample.json').read_text(encoding='utf8'))
    excluded = {tuple(sorted(x)) for x in reg['excluded_unordered_pairs']}
    dual = {tuple(sorted(x)) for x in json.loads((ROOT / 'screen' / 'dual_pairs.json').read_text(encoding='utf8'))['pairs']}
    chosen = {tuple(sorted((r['scaffold'], r['floppy']))) for r in rows}
    out = dict(all=stats(rows), by_stratum={s: stats([r for r in rows if r['stratum'] == s]) for s in reg['strata']},
               interval_caution='Pair-level Wilson and Clopper-Pearson binomial intervals; shared nets are dependent. Describes this equally '
                                'allocated crystal-system sample, not the uniform first test or a universal population.',
               integrity=dict(unique_unordered_pairs=len(chosen), no_prior_overlap=not (chosen & excluded), no_listed_duals=not (chosen & dual),
                              roles_preserved=all(r['scaffold'] == reg['rows'][r['id']]['scaffold'] and r['floppy'] == reg['rows'][r['id']]['floppy'] for r in rows)))
    table = [dict(**{k: r.get(k) for k in ('id', 'stratum', 'setting', 'space_group_number', 'scaffold', 'floppy', 'shift', 'n_F', 'n_S', 'error')},
                  route=r['prediction'].get('route'), prediction=r['prediction'].get('prediction'),
                  **{k: v for k, v in (r.get('outcome') or {}).items() if k not in ('id', 'stratum', 'setting', 'space_group_number', 'scaffold', 'floppy', 'error')})
             for r in rows]
    return out, table


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=['sample', 'predict', 'evaluate', 'count', 'routes'])
    ap.add_argument('--ids', nargs='*', type=int, help='row ids (default: all)')
    ap.add_argument('--out', default=str(ROOT / 'outputs' / 'stratified_test'))
    E.add_argument(ap); args = ap.parse_args(); E.apply_argument(args); out = Path(args.out)
    if args.stage == 'count':
        res, table = count(); C.dump(out / 'results.json', res); C.csvout(out / 'classification.csv', table); C.csvout(out / 'routes.csv', res['all']['routes'])
        a = res['all']
        print({k: a[k] for k in ('n', 'qualified', 'flat', 'designated_qualified', 'hit', 'false_designation')})
        for s, v in res['by_stratum'].items(): print(s, {k: v[k] for k in ('qualified', 'flat', 'hit', 'false_designation')})
        return
    stored = load('sample.json')
    corrected = apply_input_corrections({x for r in stored['rows'] for x in (r['scaffold'], r['floppy'])})
    if corrected: print('input corrected:', corrected)
    if args.stage == 'sample':
        s = sample(); C.dump(out / 'sample.json', s)
        same = [(a['scaffold'], a['floppy']) for a in s['rows']] == [(a['scaffold'], a['floppy']) for a in stored['rows']]
        print({k: v['population_unordered_pairs'] for k, v in s['strata'].items()}, 'identical to stored sample:', same)
        return
    preds = load('predictions.json')['rows']
    if args.stage == 'routes':
        outs = {o['id']: o for o in load('outcomes.json')['rows']}
        C.dump(out / 'route_U_C_details.json', route_details(preds, outs)); print('written', out / 'route_U_C_details.json'); return
    rows = stored['rows'] if args.stage == 'predict' else preds      # evaluate uses the shifts recorded with the predictions
    if args.ids is not None: rows = [r for r in rows if r['id'] in set(args.ids)]
    res = []
    for r in rows:
        try:
            x = predict_row(r) if args.stage == 'predict' else evaluate_row(r)
        except Exception as e:
            x = dict(r, error=f'{type(e).__name__}: {e}')
        res.append(x)
        if args.stage == 'predict': print(r['id'], r['scaffold'], r['floppy'], x.get('prediction', {}).get('route'), x.get('error', ''), flush=True)
        else: print(r['id'], r['scaffold'], r['floppy'], 'qualified', x.get('qualified'), 'flat', x.get('shape_exact_flat'), x.get('error', ''), flush=True)
    name = 'predictions.json' if args.stage == 'predict' else 'outcomes.json'
    C.dump(out / name, dict(rows=res)); print('written', out / name)


if __name__ == '__main__':
    main()
