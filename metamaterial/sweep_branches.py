"""Branch-resolved design map of the beam-frame metamaterial, ten exactly flat dual pairs.

Thick cage struts (axial 10, bending 1, in units of the floppy axial stiffness); floppy-strut bending
k_b = (k_b/k_c) k_c; contacts with (k_cr = 100) and without (k_cr = 0) rotation locking of the floppy
nodes. Per point: mean and width of each lifted branch of omega^2/k_c over the wavevectors, and the
largest branch width over branch mean.

  python metamaterial/sweep_branches.py --rcsr DIR [--pairs bcu:nbo] [--nq 48]

Stored result: metamaterial/sweep_branches.json (48 wavevectors, seed 20261107).
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
KC = [1e-3, 1e-2, 3e-2, 1e-1]
RB = [0.0, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
LOCK = [0.0, 100.0]
KSA, KSB = 10.0, 1.0
SEED = 20261107


def point(fr, qs, kcr, kc, rb):
    kb = rb * kc
    L = np.array([fr.lifted(q, kc=kc, kf_b=kb, ks_a=KSA, ks_b=KSB, kcr=kcr)[0] / kc for q in qs])
    bw = L.max(0) - L.min(0)
    return dict(kcr=kcr, kc=kc, kb_over_kc=rb, branch_mean=L.mean(0).tolist(), branch_width=bw.tolist(),
                max_rel_branch_width=float((bw / L.mean(0)).max()), mean=float(L.mean()))


def run_pair(sc, fl, qs, locks=LOCK, kcs=KC, rbs=RB):
    fr = F.Frame(sc, fl)
    rows = [point(fr, qs, kcr, kc, rb) for kcr in locks for kc in kcs for rb in rbs]
    return dict(nf=fr.nf, ns=fr.ns, rows=rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pairs', nargs='*', help='scaffold:floppy (default: the ten exactly flat pairs)')
    ap.add_argument('--nq', type=int, default=48)
    ap.add_argument('--out', default=str(HERE.parent / 'outputs' / 'metamaterial' / 'sweep_branches.json'))
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
