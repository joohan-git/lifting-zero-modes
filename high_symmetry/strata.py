"""Summaries of the scans: one row per pair and scanned set (line, plane, zone face or edge), one
witness wavevector per anomaly, and the catalogue of sets on which the scaffold has mechanisms at
every scanned point away from the zone centre (input of SM Table S4).

  python high_symmetry/strata.py [--scans DIR]
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import scipy.linalg as la
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.composite import dump, csvout


def main():
    here = Path(__file__).parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scans', default=str(here / 'scans'))
    ap.add_argument('--generic', default=str(here / 'generic'))
    ap.add_argument('--out', default=str(ROOT / 'outputs' / 'high_symmetry'))
    args = ap.parse_args()
    strata = []; exceptions = []
    from common.records import screen_records, pair_id
    order = [Path(args.scans) / f"{pair_id(r['scaffold'], r['floppy'])}.json" for r in screen_records() if r['cohort'] == 'dual' and 'scaffold' in r]
    for sp in [p for p in order if p.exists()]:
        scan = json.loads(sp.read_text(encoding='utf8')); sr = scan['rows']
        g = json.loads((Path(args.generic) / sp.name).read_text(encoding='utf8')); m = g['M_F'][0]
        name = scan['scaffold'] + '|' + scan['floppy']
        nonGamma = lambda x: la.norm(np.array(x['q']) - np.rint(x['q'])) > 1e-7
        by = defaultdict(list)
        for x in sr:
            for label in x['strata']: by[label].append(x)
        for label, rows in by.items():
            nz = [x for x in rows if nonGamma(x)]
            strata.append(dict(pair=name, stratum=label, nq=len(rows), nonGamma_nq=len(nz),
                               scaffold_zero_count=sum(x['M_S'] > 0 for x in nz), scaffold_M_max=max(x['M_S'] for x in rows),
                               floppy_evaluated=sum('M_F' in x for x in rows), floppy_M_max=max((x['M_F'] for x in rows if 'M_F' in x), default=None),
                               fixed_unlifted_max=max((x.get('fixed_zero', 0) for x in rows), default=0),
                               missing_generic_flat_multiplicity_max=max((x.get('generic_flat_multiplicity_missing', 0) for x in rows), default=0)))
            witnesses = {}
            for what, test in [('scaffold_zero', lambda x: x['M_S'] > 0),
                               ('flat_branch_break', lambda x: x.get('generic_flat_multiplicity_missing', 0) > 0),
                               ('extra_floppy', lambda x: x.get('M_F', m) > m), ('unlifted', lambda x: x.get('fixed_zero', 0) > 0)]:
                w = next((x for x in nz if test(x)), None)
                if w: witnesses[what] = w
            if witnesses: exceptions.append(dict(pair=name, stratum=label, witnesses=witnesses))
    out = Path(args.out)
    csvout(out / 'high_symmetry_strata.csv', strata); dump(out / 'high_symmetry_witnesses.json', exceptions)
    csvout(out / 'zero_set_catalog.csv', [r for r in strata if r['nonGamma_nq'] > 20 and r['scaffold_zero_count'] == r['nonGamma_nq']
                                         and r['stratum'].startswith(('line_', 'plane_', 'BZ_face_', 'BZ_edge_'))])
    print('pairs', len({r['pair'] for r in strata}), 'strata', len(strata), 'witness sets', len(exceptions), '->', out)


if __name__ == '__main__':
    main()
