"""First-order screen of dual pairs and of non-dual controls (SM Sections S3.1-S3.3).

For each pair: generic mechanism counts of both nets, clearance-optimized placement, the 1.05
nearest-contact rule, and the lifted spectrum eig(Z^H S Z) on ker R_F at 40 random wavevectors
with the scaffold held fixed.

  python screen/first_order_screen.py dual --rcsr DIR [--pairs bcu:nbo fcu:flu]
  python screen/first_order_screen.py nondual --rcsr DIR [--limit 5]
  python screen/first_order_screen.py summary          # counts from the stored results

dual: the net with fewer generic mechanisms is the scaffold. nondual: roles as listed in
nondual_pairs.json (scaffold first). Results go to outputs/screen/ unless --out is given.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import nets as E
from common import composite as C
from common.composite import dump


def gen_mech(g, rng, nq=12):
    return int(np.median([g['d'] * g['n'] - E.rank(E.matrices(g, q)[0]) for q in rng.random((nq, 3))]))


def run(a, b, forced, seed=0):
    rng = np.random.default_rng(seed)
    ga, gb = C.graph(a), C.graph(b)
    ma, mb = gen_mech(ga, rng), gen_mech(gb, rng)
    if forced or ma < mb:
        sc, fl, msc, mfl = a, b, ma, mb
    else:
        sc, fl, msc, mfl = b, a, mb, ma
    if mfl == 0:
        return dict(skip='floppy subnet has no generic mechanisms')
    p = C.pair(sc, fl)
    f = p['fl']; nfl = f['n']; pf = C.prep(f)
    evs = []; zeros = []; local = []
    for q in rng.random((40, 3)):
        Z, _ = C.null(C.compat(pf, q))
        if Z.shape[1] == 0: continue
        w = np.linalg.eigvalsh(Z.conj().T @ p['S'] @ Z); evs.append(np.sort(w)); zeros.append(int(np.sum(w < 1e-8)))
        W = np.abs(Z) ** 2; vw = W.reshape(nfl, 3, -1).sum(1); local.append(float(np.mean(vw.max(0))))
    if not evs:
        return dict(skip='no floppy kernel at sample q')
    ev = np.array(evs); spread = float(np.mean(ev.max(0) - ev.min(0)) / max(np.mean(ev), 1e-12))
    return dict(scaffold=sc, floppy=fl, gen_mech_scaffold=msc, gen_mech_floppy=mfl,
                n_contacts=len(p['contacts']), contacts_per_floppy_vertex=len(p['contacts']) / nfl,
                min_clearance=round(float(p['clearance']), 4), shift=p['shift'].tolist(),
                unlifted_mech_median=int(np.median(zeros)),
                lifted_eig_range=[round(float(ev.min()), 4), round(float(ev.max()), 4)],
                band_spread_rel=round(spread, 4), mech_vertex_locality=round(float(np.mean(local)), 3))


def counts(rows):
    comp = [r for r in rows if 'scaffold' in r]
    rigid = [r for r in comp if r['gen_mech_scaffold'] == 0]
    lifted = [r for r in rigid if r['unlifted_mech_median'] == 0]
    good = [r for r in lifted if r['band_spread_rel'] < 0.05]
    exact = [r for r in lifted if r['band_spread_rel'] == 0]
    key = lambda r: r['scaffold'] + '|' + r['floppy']
    return dict(pairs=len(rows), computed=len(comp), rigid=len(rigid), lifted=len(lifted),
                near_flat=[key(r) for r in good], exact=[key(r) for r in exact],
                skips=[[r['a'], r['b'], r['skip']] for r in rows if 'skip' in r])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['dual', 'nondual', 'summary'])
    ap.add_argument('--pairs', nargs='*', help='a:b pairs to run (default: all of the list)')
    ap.add_argument('--limit', type=int, help='run only the first N pairs of the list')
    ap.add_argument('--out', default=str(ROOT / 'outputs' / 'screen'))
    E.add_argument(ap); args = ap.parse_args(); E.apply_argument(args)
    import json
    here = Path(__file__).parent
    if args.mode == 'summary':
        for name in ('dual_screen.json', 'nondual_screen.json'):
            c = counts(json.loads((here / name).read_text(encoding='utf8'))['rows'])
            print(name, {k: (len(v) if isinstance(v, list) else v) for k, v in c.items()})
        return
    if args.mode == 'dual':
        todo = json.loads((here / 'dual_pairs.json').read_text(encoding='utf8'))['same_group']
    else:
        todo = json.loads((here / 'nondual_pairs.json').read_text(encoding='utf8'))['pairs']
    if args.pairs:
        todo = [p.split(':') for p in args.pairs]
    if args.limit:
        todo = todo[:args.limit]
    rows = []
    for a, b in todo:
        try:
            r = run(a, b, forced=args.mode == 'nondual')
        except Exception as e:
            r = dict(error=f'{type(e).__name__}: {e}')
        r = dict(a=a, b=b, **r); rows.append(r); print(a, b, r, flush=True)
    out = Path(args.out) / f'{args.mode}_screen.json'
    dump(out, dict(rows=rows, summary=counts(rows)))
    print('written', out)


if __name__ == '__main__':
    main()
