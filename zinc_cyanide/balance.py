"""Balance law at the level of rigid units in Zn(CN)2 (SM Section S5.3).

The two tetrahedra of one cyanide bridge, Zn(C)4 and Zn(N)4, are taken as the two classes. From the contact
tensors S_i of contact_rigid_units.json and the atomic masses alone (no fitted band value): the translational
contact of each unit, tr(sum_i S_i)/(3 m_unit); the kinetic shares t_A/(t_A+t_B) with t = 4/(3 m_unit); and the
predicted on-site translation doublet, the share-weighted mean of the two translational contacts. It is compared
with the two lowest branch means of the computed on-site contact operator. Requires ase (atomic masses).

  python zinc_cyanide/balance.py [--out FILE]      (stored result: zinc_cyanide/balance.json)
"""
import argparse
import json
from pathlib import Path
import numpy as np
from ase.data import atomic_masses, atomic_numbers

HERE = Path(__file__).resolve().parent
GROUPS = [(0, [2, 3, 4, 5], 'C'), (1, [6, 7, 8, 9], 'N')]      # Zn centre and its four ends, per tetrahedron


def mass(symbol):
    return float(atomic_masses[atomic_numbers[symbol]])


def balance(data):
    tensors = data['contact_tensors']; units = []
    for centre, ends, symbol in GROUPS:
        unit_mass = mass('Zn') + 4 * mass(symbol)
        total = sum((np.asarray(tensors[str(i)]['S']) for i in [centre] + ends), start=np.zeros((3, 3)))
        ev = np.array([tensors[str(i)]['eig'] for i in ends])
        assert np.max(np.abs(ev[:, 1] - ev[:, 2])) < 1e-8          # tangential plane degenerate at the threefold ends
        units.append(dict(endpoint=symbol, unit_mass=unit_mass, t_cross=4 / (3 * unit_mass),
                          contact_translation=float(np.trace(total) / (3 * unit_mass)),
                          contact_rotation=float(ev[:, 1:].mean() / mass(symbol))))
    fa = units[1]['t_cross'] / (units[0]['t_cross'] + units[1]['t_cross'])
    doublet = fa * units[0]['contact_translation'] + (1 - fa) * units[1]['contact_translation']
    return dict(units=units, kinetic_shares=[fa, 1 - fa], translation_doublet=doublet,
                recorded_doublet=data['contact_real_onsite']['branch_mean'][:2])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', default=str(HERE / 'contact_rigid_units.json'))
    ap.add_argument('--out', default=str(HERE.parent / 'outputs' / 'zinc_cyanide' / 'balance.json'))
    a = ap.parse_args()
    res = balance(json.loads(Path(a.input).read_text(encoding='utf8')))
    p = Path(a.out); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(res, indent=1), encoding='utf8')
    print(json.dumps(res, indent=1))
