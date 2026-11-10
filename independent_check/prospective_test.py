"""Prospective classifier test repeated with the second implementation (SM Section S3.6).

predict: geometry-only rules on the 300 prospective pairs of classifier/sampling.json, in the common
primitive cell: rule N (planar vertices whose normals exhaust the generic mechanisms) and rule B
(bipartite net with isotropic bond and contact tensors in each class); abstain otherwise. No lifted
spectrum is computed. evaluate: scaffold rigidity, lifting and fixed-scaffold spectra with the origin
found in the predict stage. compare: counts and pair-by-pair agreement with the primary calculation.

  python independent_check/prospective_test.py predict --rcsr DIR [--ids 187 263]
  python independent_check/prospective_test.py evaluate --rcsr DIR [--ids 187 263]
  python independent_check/prospective_test.py compare       # from the stored files
"""
import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
import engine as V
import engine_primitive as V2
from common.composite import dump


def predict_row(cg, p):
    rec = dict(id=p['id'], scaffold=p['scaffold'], floppy=p['floppy'])
    pre = V2.prepare(cg, p['scaffold'], p['floppy'])
    fl = pre['fl']; T, planar, normals = V2.vertex_geometry(fl, pre['unitF'])
    rules = V2.predict_rules(fl, pre['S'], T, planar, normals, pre['MF'])
    rec.update(group=fl['group'], mult=fl['mult'], V_F=fl['V'], E_F=fl['E'], V_S=pre['sc']['V'], E_S=pre['sc']['E'], M_F=pre['MF'],
               shift_conv=pre['P']['shift_conv'].tolist(), clearance=pre['P']['clearance'], n_contacts=len(pre['P']['contacts']))
    rec.update(rules)
    return rec


def evaluate_row(cg, p):
    rec = dict(id=p['id'], scaffold=p['scaffold'], floppy=p['floppy'])
    if 'error' in p: raise RuntimeError('prediction stage error: ' + p['error'])
    pre = V2.prepare(cg, p['scaffold'], p['floppy'], shift=p['shift_conv'])
    assert len(pre['P']['contacts']) == p['n_contacts']
    out = V2.spectra(pre); rec.update(out)
    mf = out.get('mech_F', 0); sp = out.get('spread_rel')
    rec['rigid'] = out['mech_S'] == 0
    rec['all_lifted'] = out.get('unlifted_median') == 0 and sp is not None
    # qualified: rigid scaffold, every mechanism lifted, and at least one mechanism to lift
    rec['qualified'] = bool(rec['rigid'] and rec['all_lifted'] and mf > 0)
    rec['shape_flat'] = bool(mf > 0 and sp is not None and (sp < 1e-6 or out.get('eig_max', 1) < 1e-8))
    rec['exact_flat'] = bool(rec['qualified'] and rec['shape_flat'])
    rec['predicted_flat'] = bool(p['predicted_flat']); rec['n_planar'] = p.get('n_planar'); rec['rule_N'] = p.get('rule_N'); rec['rule_B'] = p.get('rule_B')
    if rec['predicted_flat'] and rec.get('bands'):
        vals = p['rule_N'].get('values') if p['rule_N']['predict'] == 'flat' else [p['rule_B']['value']] * len(rec['bands'])
        rec['value_match'] = bool(rec['exact_flat'] and np.allclose(sorted(rec['bands']), sorted(vals), atol=1e-5))
    return rec


def compare(rows, root=ROOT):
    """Counts of the second implementation and agreement with the primary outcomes and predictions."""
    prim = {o['id']: o for o in json.loads((root / 'classifier' / 'outcomes_prospective.json').read_text(encoding='utf8'))['rows']}
    ppred = {r['id']: r for r in json.loads((root / 'classifier' / 'predictions_prospective.json').read_text(encoding='utf8'))['rows']}
    key = lambda r: f"{r['scaffold']}|{r['floppy']}"
    q = [r for r in rows if r['qualified']]; flat = [r for r in q if r['exact_flat']]; calls = [r for r in q if r['predicted_flat']]
    pq = {i for i, o in prim.items() if o.get('qualified')}; pf = {i for i in pq if prim[i].get('shape_exact_flat')}
    pc = {i for i in pq if ppred[i]['prediction'].get('prediction') == 'predicted_flat'}
    return dict(n=len(rows), errors=[r['id'] for r in rows if 'error' in r], rigid=sum(bool(r.get('rigid')) for r in rows),
                qualified=len(q), actual_flat=len(flat), flat_pairs=[key(r) for r in flat],
                predicted_flat_qualified=len(calls), true_positive=sum(r['exact_flat'] for r in calls),
                false_positive=[key(r) for r in calls if not r['exact_flat']],
                flat_abstained=[f"{key(r)}: {r['rule_N'].get('reason')}" for r in flat if not r['predicted_flat']],
                values_match_band=all(r.get('value_match') for r in calls if r['exact_flat']),
                same_qualified_set_as_primary={r['id'] for r in q} == pq,
                same_flat_set_as_primary={r['id'] for r in flat} == pf,
                same_designations_as_primary={r['id'] for r in calls} == pc,
                disagree_rigid=[r['id'] for r in rows if bool(r.get('rigid')) != bool(prim[r['id']].get('rigid'))])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=['predict', 'evaluate', 'compare'])
    ap.add_argument('--ids', nargs='*', type=int, help='pair ids (default: all 300)')
    ap.add_argument('--rcsr', help='directory with rcsr3d.cgd (default: $RCSR_DIR)')
    ap.add_argument('--out', default=str(ROOT / 'outputs' / 'independent_check'))
    args = ap.parse_args()
    if args.stage == 'compare':
        rows = json.loads((HERE / 'prospective_outcomes.json').read_text(encoding='utf8'))['rows']
        print(json.dumps(compare(rows), indent=1)); return
    d = args.rcsr or os.environ.get('RCSR_DIR')
    if not d: raise SystemExit('RCSR files needed: pass --rcsr DIR or set RCSR_DIR (see README).')
    cg = V.parse_cgd(str(Path(d) / 'rcsr3d.cgd'))
    if args.stage == 'predict':
        todo = json.loads((ROOT / 'classifier' / 'sampling.json').read_text(encoding='utf8'))['prospective']
    else:
        todo = json.loads((HERE / 'prospective_predictions.json').read_text(encoding='utf8'))['rows']
    if args.ids is not None: todo = [r for r in todo if r['id'] in set(args.ids)]
    rows = []
    for p in todo:
        try:
            rec = predict_row(cg, p) if args.stage == 'predict' else evaluate_row(cg, p)
        except Exception as e:
            rec = dict(id=p['id'], scaffold=p['scaffold'], floppy=p['floppy'], error=f'{type(e).__name__}: {e}', predicted_flat=False, qualified=False, exact_flat=False)
        rows.append(rec)
        print(p['id'], p['scaffold'], p['floppy'], 'flat call' if rec.get('predicted_flat') else '-',
              'qualified' if rec.get('qualified') else '', 'FLAT' if rec.get('exact_flat') else '', rec.get('error', ''), flush=True)
    name = 'prospective_predictions.json' if args.stage == 'predict' else 'prospective_outcomes.json'
    dump(Path(args.out) / name, dict(rows=rows))


if __name__ == '__main__':
    main()
