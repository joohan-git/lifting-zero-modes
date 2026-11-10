"""Branch-resolved design map of the beam frame for seven near-flat dual pairs (max over lifted branches of
branch width / band mean), thick cage struts (k_s,axial = 10, k_s,bend = 1 in units of the floppy axial stiffness),
contacts with and without rotation locking of the floppy nodes; 24 wavevectors (seed 20261107).

  python metamaterial/sweep_near_flat.py --rcsr DIR      (stored result: metamaterial/sweep_near_flat.json)
"""
import json, os, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
if '--rcsr' in sys.argv:
    os.environ['RCSR_DIR'] = sys.argv[sys.argv.index('--rcsr') + 1]
import frame as F
OUT = HERE.parent / 'outputs' / 'metamaterial' / 'sweep_near_flat.json'

PAIRS = [("sve","cai"),("bet","mep"),("bsn","lig"),("thp","ctn-x"),("mgz-x","mgz-x-d"),("mgc-x","mtn"),("nas","nas-d")]
KC = [1e-3, 1e-2]
RB = [0.0, 1e-2, 1e-1, 1.0]
LOCK = [0.0, 100.0]
NQ = 24
KSA, KSB = 10.0, 1.0


def main():
    rng = np.random.default_rng(20261107)
    qs = rng.random((NQ, 3)) - 0.5
    out = {}
    for sc, fl in PAIRS:
        fr = F.Frame(sc, fl); rows = []
        for kcr in LOCK:
            for kc in KC:
                for rb in RB:
                    kb = rb * kc
                    L = np.array([fr.lifted(q, kc=kc, kf_b=kb, ks_a=KSA, ks_b=KSB, kcr=kcr)[0] / kc for q in qs])
                    bw = L.max(0) - L.min(0)
                    rows.append(dict(kcr=kcr, kc=kc, kb_over_kc=rb, branch_mean=L.mean(0).tolist(),
                                     branch_width=bw.tolist(), max_rel_branch_width=float((bw / L.mean(0)).max()),
                                     mean=float(L.mean())))
        out[f"{sc}|{fl}"] = dict(nf=fr.nf, ns=fr.ns, rows=rows)
        print(sc, fl, flush=True)
        OUT.parent.mkdir(parents=True, exist_ok=True); json.dump(out, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
