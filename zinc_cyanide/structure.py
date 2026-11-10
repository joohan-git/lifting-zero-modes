"""Zn(CN)2 structure used in the article: ordered head-to-tail model, P-43m, a = 5.902 A, Zn-C/N 1.98 A
along <111> (ZnCN2.cif), and the same structure in the face-centred cell of four formula units that
holds both interpenetrated dia nets (ZnCN2_fcc.cif). Requires ase.

  python zinc_cyanide/structure.py [--out DIR]
"""
import argparse
from pathlib import Path
import numpy as np
from ase.build import make_supercell
from ase.io import write
from ase.spacegroup import crystal


def zinc_cyanide():
    a = 5.902
    x = 1.98 / (a * np.sqrt(3))
    s = crystal(["Zn", "Zn", "C", "N"], basis=[(0, 0, 0), (0.5, 0.5, 0.5), (x, x, x), (0.5 - x, 0.5 - x, 0.5 - x)],
                spacegroup=215, cellpar=[a, a, a, 90, 90, 90])
    return s, make_supercell(s, [[0, 1, 1], [1, 0, 1], [1, 1, 0]])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "outputs" / "zinc_cyanide"))
    out = Path(ap.parse_args().out); out.mkdir(parents=True, exist_ok=True)
    s, f = zinc_cyanide()
    write(out / "ZnCN2.cif", s); write(out / "ZnCN2_fcc.cif", f)
    print("written", out)
