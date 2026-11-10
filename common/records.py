"""Screened pairs in a fixed order: 76 same-group dual pairs, then 250 non-dual controls.

Each record carries the roles and the relative origin found by the screen, so that later
calculations rebuild exactly the screened composite.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def pair_id(scaffold, floppy):
    """File stem used for per-pair result files."""
    return f'{scaffold}__{floppy}'


def screen_records(root=ROOT):
    out = []
    for cohort, name in (('dual', 'dual_screen.json'), ('nondual', 'nondual_screen.json')):
        for r in json.loads((Path(root) / 'screen' / name).read_text(encoding='utf8'))['rows']:
            v = dict(r); v.update(cohort=cohort, key=r['a'] + '|' + r['b'])
            out.append(v)
    return out


def find(scaffold, floppy, root=ROOT):
    return next(r for r in screen_records(root) if r.get('scaffold') == scaffold and r.get('floppy') == floppy)
