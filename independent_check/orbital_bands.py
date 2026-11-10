"""Separate implementation of the p-orbital crystal-field results (SM Section S9), written without
orbital/orbital.py: the bonds are found from the site positions, and the wavevectors are random.

hcb (nearest-neighbour distance 1, Bravais vectors (3/2, +-sqrt(3)/2)) with p_z on-site energy 4;
fields diag(3/2, 3/2, 0) (route U) and diag(2.1, 0.9, 0) (broken), lambda = 0.03, 400 random q.
flu (conventional cell, a = 1, bonds of length sqrt(3)/4): field 4/3 on the 8 A sites and 2 on the 4 B sites
(route C), plus 0.4 diag(1, -1, 0) on every site for the broken case; 200 random q, of which the points with a
fourfold zero band are used. E_minus is the closed-form finite-field energy of route C.
Random numbers: numpy default_rng(7), 400 x 2 draws then 200 x 3 draws.

  python independent_check/orbital_bands.py [--out FILE]
"""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def build(sites, cell, bonds_cut):
    """sites: fractional positions; returns q -> R(q) and the bond list."""
    A = np.array(cell, float)
    n = len(sites); bonds = []
    for i, j in itertools.product(range(n), repeat=2):
        for s in itertools.product([-1, 0, 1], repeat=len(cell)):
            d = (np.array(sites[j]) + np.array(s) - np.array(sites[i])) @ A
            L = np.linalg.norm(d)
            if abs(L - bonds_cut) < 1e-6 and (i < j or (i == j and tuple(s) > (0,) * len(s))):
                bonds.append((i, j, np.array(s), d / L))
    dim = 3

    def R(q):
        M = np.zeros((len(bonds), n * dim), complex)
        for b, (i, j, s, u) in enumerate(bonds):
            u3 = np.zeros(3); u3[:len(u)] = u
            M[b, dim * j:dim * j + 3] += u3 * np.exp(2j * np.pi * q @ s)
            M[b, dim * i:dim * i + 3] -= u3
        return M
    return R, bonds


def run():
    rng = np.random.default_rng(7)
    out = {}
    a1, a2 = np.array([1.5, np.sqrt(3) / 2]), np.array([1.5, -np.sqrt(3) / 2])
    Bfrac = np.linalg.solve(np.array([a1, a2]).T, np.array([1.0, 0.0]))
    Rh, bh = build([[0, 0], list(Bfrac)], [a1, a2], 1.0)
    assert len(bh) == 3
    Pz = np.diag([0, 0, 1, 0, 0, 1.0])
    SU = np.diag([1.5, 1.5, 0, 1.5, 1.5, 0])
    Sb = np.diag([2.1, 0.9, 0, 2.1, 0.9, 0])
    lam = 0.03
    low_U, first_b = [], []
    for q in rng.random((400, 2)):
        H0 = Rh(q).conj().T @ Rh(q) + 4 * Pz
        w, V = np.linalg.eigh(H0)
        Z = V[:, :1]                               # generic q: one in-plane zero mode
        assert w[0] < 1e-10 < w[1]
        first_b.append(np.linalg.eigvalsh(Z.conj().T @ Sb @ Z)[0])
        low_U.append(np.linalg.eigvalsh(H0 + lam * SU)[0])
    out['hcb_U_lowest_over_lambda_range'] = [min(low_U) / lam, max(low_U) / lam]
    out['hcb_broken_first_order_range'] = [min(first_b), max(first_b)]

    A_sites = [list(p) for p in itertools.product([0.25, 0.75], repeat=3)]
    B_sites = [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]]
    Rf, bf = build(A_sites + B_sites, np.eye(3), np.sqrt(3) / 4)
    assert len(bf) == 32, len(bf)
    nA = 8
    S = np.diag([4 / 3] * (3 * nA) + [2.0] * 12)
    Sbroken = np.diag(list(np.tile([0.4, -0.4, 0], 12)))
    tA, tB, sA, sB = 4 / 3, 8 / 3, 4 / 3, 2.0
    Em = (tA + tB + lam * (sA + sB) - np.sqrt((tA - tB + lam * (sA - sB)) ** 2 + 4 * tA * tB)) / 2
    first, low, firstb = [], [], []
    for q in rng.random((200, 3)):
        H0 = Rf(q).conj().T @ Rf(q)
        w, V = np.linalg.eigh(H0)
        k = int(np.sum(w < 1e-9))
        if k != 4:
            continue
        Z = V[:, :4]
        first.append(np.linalg.eigvalsh(Z.conj().T @ S @ Z))
        firstb.append(np.linalg.eigvalsh(Z.conj().T @ (S + Sbroken) @ Z))
        low.append(np.linalg.eigvalsh(H0 + lam * S)[:4])
    first, low, firstb = map(np.array, (first, low, firstb))
    out['flu_n_generic'] = len(first)
    out['flu_first_order_range'] = [first.min(), first.max()]
    out['flu_exact_lowest4_minus_Eminus_max'] = float(np.abs(low - Em).max())
    out['flu_Eminus_at_0.03'] = Em
    out['flu_broken_branch_widths'] = list(np.ptp(firstb, axis=0))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=str(ROOT / 'outputs' / 'independent_check' / 'orbital_bands.json'))
    out = Path(ap.parse_args().out); out.parent.mkdir(parents=True, exist_ok=True)
    res = run()
    text = json.dumps(res, indent=1, default=float)
    out.write_text(text, encoding='utf-8'); print(text)


if __name__ == '__main__':
    main()
