"""Branch widths of the single-net rigid-unit band of Zn(CN)2 (intrinsic_band), compared with the on-site contact
projection, from contact_rigid_units.json and contact_pin_joints.json (SM Section S5.3 and main text). Units THz^2.

  python zinc_cyanide/zn_branches.py [--out DIR]       (stored result: zinc_cyanide/zn_branches.json)
"""
import argparse
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent


def branches():
    out = {}
    for key, f in (('rigid_units', 'contact_rigid_units.json'), ('pin_joints', 'contact_pin_joints.json')):
        d = json.loads((HERE / f).read_text(encoding='utf8')); b = d['intrinsic_band']; w, mu = b['branch_width'], b['branch_mean']
        c = d['contact_real_onsite']; rel = [x / m for x, m in zip(w, mu)]
        out[key] = dict(source=f, model=d['model'], n_branches=len(w),
                        units='THz^2 (dynamical-matrix eigenvalues x conversion^2, as in contact_operator.py)',
                        intrinsic_band_branch_width=w, intrinsic_band_branch_mean=mu, relative_width_per_branch=rel,
                        width_min=min(w), width_max=max(w), relative_width_min=min(rel), relative_width_max=max(rel),
                        relative_width_lowest_branch=rel[0], band_total_range=[b['min'], b['max']], band_mean=b['mean'],
                        contact_onsite_branch_width_max=max(c['branch_width']), contact_onsite_branch_mean=c['branch_mean'],
                        ratio_min_intrinsic_width_to_max_onsite_width=min(w) / max(c['branch_width']))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=str(HERE.parent / 'outputs' / 'zinc_cyanide'))
    out = branches(); p = Path(ap.parse_args().out) / 'zn_branches.json'; p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding='utf8')
    for k, v in out.items():
        print(k, v['n_branches'], 'width', round(v['width_min'], 3), round(v['width_max'], 3), 'onsite max', v['contact_onsite_branch_width_max'])
