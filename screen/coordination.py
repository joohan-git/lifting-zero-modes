"""Mean coordination of dual scaffolds and the Euler relation (main text; SM Section S3.4).

For each screened dual pair: vertices V and edges E of the floppy net, vertices C and edges F of the
scaffold, the mean scaffold degree 2F/C (Maxwell bound 6 for rigidity in three dimensions), the
mean face count of the floppy net's natural tiling from 3dall.txt, and the residual V - E + F - C.

  python screen/coordination.py --rcsr DIR
  python screen/coordination.py count       # counts from the stored file
"""
import argparse
import json
import re
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from common import nets as E
from common import composite as C
from common.records import screen_records


def tiling_faces(symbol):
    primary = symbol.split('(')[0].strip(); terms = []
    for coefficient, body in re.findall(r'(\d*)\s*\[([^\]]+)\]', primary):
        count = 0
        for term in body.split('.'):
            m = re.fullmatch(r'\d+(?:\^(\d+))?', term)
            if m is None: raise ValueError('unsupported tiling face symbol component ' + term)
            count += int(m[1] or 1)
        terms.append((int(coefficient or 1), count))
    if not terms: raise ValueError('no natural-tiling face expression')
    return sum(w * f for w, f in terms) / sum(w for w, f in terms), terms


def rows():
    meta = E.read_all(); out = []
    for r in screen_records():
        if r['cohort'] != 'dual' or 'scaffold' not in r: continue
        gs, gf = C.graph(r['scaffold']), C.graph(r['floppy']); V, Ee, Cc, F = gf['n'], gf['e'], gs['n'], gs['e']; degree = 2 * F / Cc
        faces, classes = tiling_faces(meta[r['floppy']]['tiling']); match = abs(faces - degree) < 1e-8; euler = V - Ee + F - Cc
        out.append(dict(scaffold=r['scaffold'], floppy=r['floppy'], V=V, E=Ee, C_dual_vertices=Cc, F_dual_edges=F,
                        tile_mean_faces=faces, dual_mean_degree=degree, tile_mean_matches_dual_degree=match, nu_scaffold=3 * Cc - F,
                        generic_scaffold_M=r['gen_mech_scaffold'], rigid=r['gen_mech_scaffold'] == 0, common_cell_euler_residual=euler,
                        euler_consistent=bool(match and euler == 0)))
    return out


def summary(rr):
    return dict(total=len(rr), dual_degree_at_least_6=sum(r['dual_mean_degree'] >= 6 for r in rr),
                rigid_with_dual_degree_at_least_6=sum(r['rigid'] and r['dual_mean_degree'] >= 6 for r in rr),
                rigid_with_dual_degree_below_6=sum(r['rigid'] and r['dual_mean_degree'] < 6 for r in rr),
                euler_inconsistent=[r['scaffold'] + '|' + r['floppy'] for r in rr if not r['euler_consistent']],
                degree_at_least_6_not_rigid=[r['scaffold'] + '|' + r['floppy'] for r in rr if r['dual_mean_degree'] >= 6 and not r['rigid']])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', nargs='?', default='run', choices=['run', 'count'])
    ap.add_argument('--out', default=str(ROOT / 'outputs' / 'screen'))
    E.add_argument(ap); args = ap.parse_args(); E.apply_argument(args)
    rr = json.loads((HERE / 'coordination.json').read_text(encoding='utf8'))['rows'] if args.stage == 'count' else rows()
    s = summary(rr); print(json.dumps(s, indent=1))
    if args.stage == 'run': C.dump(Path(args.out) / 'coordination.json', dict(summary=s, rows=rr))


if __name__ == '__main__':
    main()
