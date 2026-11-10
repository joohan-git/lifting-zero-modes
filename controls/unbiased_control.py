"""Unbiased control (main text; SM Section S3.4 and Table S6).

32 floppy nets drawn uniformly (seed 20261111) from the distinct nets with nu > 0 that appear in
either role of a same-group dual pair; each is caged by a listed dual and by three uniformly drawn
nets of the same space group. Rigidity from 12 random wavevectors; first-order spectrum as in the
screen for rigid cages. Counts and a cluster bootstrap over floppy nets.

  python controls/unbiased_control.py sample --rcsr DIR      # redraw the sample
  python controls/unbiased_control.py run --rcsr DIR [--rows 0 1 2 3]
  python controls/unbiased_control.py count                  # counts from the stored file
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from common import nets as E
from common import composite as C
from common.records import screen_records

SEED = C.SEED + 5          # 20261111


def sample():
    pairs = json.loads((ROOT / 'screen' / 'dual_pairs.json').read_text(encoding='utf8'))['same_group']; pop = {}; errors = []
    CG = C.CG()
    for a, b in pairs:
        for f, s in ((a, b), (b, a)):
            try:
                g = C.graph(f)
                if 3 * g['n'] - g['e'] > 0: pop.setdefault(f, []).append(s)
            except Exception as e:
                errors.append([f, str(e)])
    rng = np.random.default_rng(SEED); chosen = rng.choice(sorted(pop), min(32, len(pop)), replace=False)
    rows = []
    for f in chosen:
        duals = pop[f]; s = str(rng.choice(duals)); rows.append(dict(floppy=str(f), scaffold=s, kind='dual'))
        pool = sorted(n for n, v in CG.items() if v['group'] == CG[f]['group'] and n != f and n not in duals)
        for s in rng.choice(pool, min(3, len(pool)), replace=False): rows.append(dict(floppy=str(f), scaffold=str(s), kind='nondual'))
    return dict(seed=SEED, population=pop, population_nets=len(pop), sample_floppy_nets=len(chosen), eligibility_errors=errors, rows=rows,
                sampling='Uniform without replacement over distinct nu>0 nets with a listed same-group dual. One uniform dual and three uniform same-group nondual names per selected net. No flatness or rigidity selection; no size cap.')


def run_row(r, screened):
    out = dict(r)
    s, f = C.graph(r['scaffold']), C.graph(r['floppy']); ps = C.prep(s); rng = np.random.default_rng(C.SEED + 50)
    ms = [ps[-1] - E.rank(C.compat(ps, q)) for q in rng.random((12, 3))]
    out.update(scaffold_M=sorted(set(ms)), rigid=all(x == 0 for x in ms), nu_F=3 * f['n'] - f['e'], nu_S=3 * s['n'] - s['e'])
    if out['rigid']:
        prev = next((x for x in screened if x.get('scaffold') == r['scaffold'] and x.get('floppy') == r['floppy']), {})
        p = C.pair(r['scaffold'], r['floppy'], prev.get('shift'))
        st = C.generic_stats(p, 40, C.SEED + 51); out.update(stats=st, shift=p['shift'], clearance=p['clearance'])
        out['fully_lifted'] = st['generic_unlifted'] == [0]
        out['exact_flat'] = out['fully_lifted'] and st['spread_rel'] < 1e-7
        out['near_flat'] = out['fully_lifted'] and st['spread_rel'] < .05
    else:
        out.update(fully_lifted=False, exact_flat=False, near_flat=False)
    return out


def summarize(rows):
    hs = {}
    for kind in ('dual', 'nondual'):
        rr = [r for r in rows if r['kind'] == kind]; ok = [r for r in rr if 'error' not in r]
        nr = sum(r['rigid'] for r in ok); nl = sum(r['fully_lifted'] for r in ok); nf = sum(r['exact_flat'] for r in ok)
        hs[kind] = dict(selected=len(rr), computed=len(ok), errors=len(rr) - len(ok), rigid=nr, fully_lifted=nl, exact_flat=nf,
                        near_flat=sum(r['near_flat'] for r in ok), rigidity_rate=nr / len(ok) if ok else None,
                        flat_rate_all=nf / len(ok) if ok else None, flat_rate_rigid_lifted=nf / nl if nl else None)
    boot = {}; nets = sorted(set(r['floppy'] for r in rows)); brng = np.random.default_rng(C.SEED + 55)
    for metric in ('rigid', 'fully_lifted', 'exact_flat'):
        d = np.array([np.mean([r[metric] for r in rows if r['floppy'] == f and r['kind'] == 'dual' and 'error' not in r]) for f in nets])
        n = np.array([np.mean([r[metric] for r in rows if r['floppy'] == f and r['kind'] == 'nondual' and 'error' not in r]) for f in nets])
        smp = brng.integers(0, len(nets), (5000, len(nets))); delta = (d[smp] - n[smp]).mean(1)
        boot[metric] = dict(dual_minus_nondual=float((d - n).mean()), cluster_bootstrap_95CI=np.quantile(delta, [.025, .975]).tolist(),
                            resamples=5000, cluster='floppy net; keep all three nondual controls with each dual')
    return hs, boot


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=['sample', 'run', 'count'])
    ap.add_argument('--rows', nargs='*', type=int, help='row indices to run (default: all 128)')
    ap.add_argument('--out', default=str(ROOT / 'outputs' / 'controls'))
    E.add_argument(ap); args = ap.parse_args(); E.apply_argument(args)
    stored = json.loads((HERE / 'unbiased_control.json').read_text(encoding='utf8'))
    if args.stage == 'count':
        hs, boot = summarize(stored['rows'])
        for k in hs: print(k, {x: hs[k][x] for x in ('computed', 'rigid', 'fully_lifted', 'exact_flat')})
        b = boot['rigid']; lo, hi = b['cluster_bootstrap_95CI']
        print(f"rigidity difference {100 * b['dual_minus_nondual']:.2f} points, 95% bootstrap interval {100 * lo:.2f} to {100 * hi:.2f}")
        return
    if args.stage == 'sample':
        s = sample(); C.dump(Path(args.out) / 'unbiased_sample.json', s)
        same = [(r['floppy'], r['scaffold'], r['kind']) for r in s['rows']] == [(r['floppy'], r['scaffold'], r['kind']) for r in stored['rows']]
        print('floppy nets', s['sample_floppy_nets'], 'of', s['population_nets'], '; identical to stored sample:', same)
        return
    screened = screen_records(); rows = []
    for idx, r in enumerate(stored['rows']):
        if args.rows is not None and idx not in args.rows: continue
        base = {k: r[k] for k in ('floppy', 'scaffold', 'kind')}
        try:
            x = run_row(base, screened)
        except Exception as e:
            x = dict(base, error=f'{type(e).__name__}: {e}')
        rows.append(dict(index=idx, **x)); print(idx, r['kind'], r['scaffold'], r['floppy'], x.get('rigid'), x.get('exact_flat'), flush=True)
    out = dict(rows=rows)
    if args.rows is None: out['summary'], out['paired_uncertainty'] = summarize(rows)
    C.dump(Path(args.out) / 'unbiased_control.json', out)


if __name__ == '__main__':
    main()
