"""Generic first-order spectra of the screened dual pairs (reference values for the scans).

Rebuilds each screened composite with its stored origin and evaluates eig(Z^H S Z) at 40 random
wavevectors, with mechanism counts, Maxwell index and states of self-stress of the floppy net.

  python high_symmetry/generic_spectra.py --rcsr DIR [--pairs bcu:nbo]
"""
import argparse
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import nets as E
from common import composite as C
from common.records import screen_records, pair_id


def compute(r, nq=40):
    p = C.pair(r['scaffold'], r['floppy'], r['shift']); st = C.generic_stats(p, nq)
    lengths = np.concatenate([p['fl']['length'], p['sc']['length'], [x[4] for x in p['contacts']]])
    st.update(scaffold=r['scaffold'], floppy=r['floppy'], rest_length_min=float(lengths.min()),
              rest_length_max=float(lengths.max()), rest_length_rule='l0 = current bond/contact length',
              force_residual=0.0, geometric_stiffness_norm=0.0)
    return st


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pairs', nargs='*', help='scaffold:floppy (default: all computed dual pairs)')
    ap.add_argument('--out', default=str(ROOT / 'outputs' / 'high_symmetry' / 'generic'))
    E.add_argument(ap); args = ap.parse_args(); E.apply_argument(args)
    want = {tuple(x.split(':')) for x in args.pairs} if args.pairs else None
    for r in screen_records():
        if r['cohort'] != 'dual' or 'scaffold' not in r: continue
        if want and (r['scaffold'], r['floppy']) not in want: continue
        st = compute(r)
        C.dump(Path(args.out) / f"{pair_id(r['scaffold'], r['floppy'])}.json", st)
        print(r['scaffold'], r['floppy'], st['M_F'], st['nu_F'], st['branch_means'][:4], st['spread_rel'], flush=True)


if __name__ == '__main__':
    main()
