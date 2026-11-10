"""Design map of the beam-frame metamaterial: flatness of the lifted band versus contact stiffness k_c
and strut bending k_b, for the ten exactly flat dual pairs (both nets with the same struts).

Per pair and parameter point: mean and q05-q95 relative spread of omega^2/k_c over the lifted band,
the same for the first-order prediction eig[Z^H (k_c S + K_b) Z], and the mean mechanism weight.

  python metamaterial/sweep_contact_bending.py --rcsr DIR [--pairs bcu:nbo] [--nq 48]

Stored result: metamaterial/sweep_contact_bending.json (48 wavevectors, seed 20261107).
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import frame as F
from common import nets as E

PAIRS = [("bcu", "nbo"), ("hxg-d", "hxg"), ("nia", "nia-d"), ("she-d", "she"), ("bsn", "lvt"),
         ("qtz-x", "uom"), ("pcu", "urk"), ("hex", "bnn"), ("fcu", "flu"), ("acs", "gra")]
KC = [1e-3, 1e-2, 3e-2, 1e-1]                          # contact / axial strut stiffness
RB = [0.0, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]    # bending / contact
SEED = 20261107


def spread(a):
    a = np.asarray(a).ravel(); m = a.mean()
    q05, q95 = np.quantile(a, [0.05, 0.95])
    return float(m), float((q95 - q05) / m), float(q05), float(q95)


def run_pair(sc, fl, qs, kcs=KC, rbs=RB):
    fr = F.Frame(sc, fl); rows = []
    for kc in kcs:
        for rb in rbs:
            kb = rb * kc; L, FO, W = [], [], []
            for q in qs:
                w2, wt, _ = fr.lifted(q, kc=kc, kf_b=kb, ks_b=kb)
                L.append(w2 / kc); W.append(wt.mean()); FO.append(fr.first_order(q, kc=kc, kf_b=kb) / kc)
            m, rs, a, b = spread(L); mf, rf, _, _ = spread(FO)
            rows.append(dict(kc=kc, kb_over_kc=rb, mean=m, rel_spread=rs, q05=a, q95=b, fo_mean=mf, fo_rel_spread=rf, weight=float(np.mean(W))))
    return dict(nf=fr.nf, ns=fr.ns, contacts=len(fr.p["contacts"]), rows=rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pairs', nargs='*', help='scaffold:floppy (default: the ten exactly flat pairs)')
    ap.add_argument('--nq', type=int, default=48)
    ap.add_argument('--out', default=str(HERE.parent / 'outputs' / 'metamaterial' / 'sweep_contact_bending.json'))
    E.add_argument(ap); args = ap.parse_args(); E.apply_argument(args)
    pairs = [tuple(x.split(':')) for x in args.pairs] if args.pairs else PAIRS
    qs = np.random.default_rng(SEED).random((args.nq, 3)) - 0.5
    out = {}
    for sc, fl in pairs:
        out[f"{sc}|{fl}"] = run_pair(sc, fl, qs); print(sc, fl, flush=True)
    p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(out, indent=1), encoding='utf8')
    print('written', p)


if __name__ == "__main__":
    main()
