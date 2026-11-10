"""LaTeX tables and macros of the Supplemental Material, generated from the result files of this archive.
No number is copied by hand.

  python tables/make_tables.py [--root DIR] [--out DIR]
  python tables/make_full_tables.py --rcsr DIR [--root DIR] [--out DIR]

--root: archive root holding the result files (default: the folder above tables/); --out: where the .tex files
are written (default: ROOT/outputs/tables). Run make_tables.py first; make_full_tables.py reads its macro files.
tables/reference/ holds the files as printed in the article, for comparison.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]      # set by configure()
OUT = ROOT / 'outputs' / 'tables'


def configure(root=None, out=None):
    global ROOT, OUT
    if root: ROOT = Path(root).resolve()
    OUT = Path(out).resolve() if out else ROOT / 'outputs' / 'tables'
    OUT.mkdir(parents=True, exist_ok=True)


def J(rel):
    return json.loads((ROOT / rel).read_text(encoding='utf8'))


def CSV(rel):
    import csv
    return list(csv.DictReader(open(ROOT / rel, encoding='utf-8-sig')))
EOL = ' \\\\'          # LaTeX row end
HL = '\\hline'


def net(n):
    return '\\net{' + n + '}'


def screen_table():
    rows = []
    for d in J('screen/dual_screen.json')['rows']:
        if 'scaffold' in d:
            ok = d['gen_mech_scaffold'] == 0 and d['unlifted_mech_median'] == 0
            sp = d['band_spread_rel']
            cls = ('F' if sp == 0 else 'N' if sp < 0.05 else '') if ok else ''
            lo, hi = (max(0.0, x) for x in d['lifted_eig_range'])
            # range below the stored precision (4 decimals) with a nonzero spread: the mean band value is near zero
            note = '$^a$' if hi < 1e-4 and sp > 0 else ''
            assert not note or d['unlifted_mech_median'] == d['gen_mech_floppy'] - 1, (d['scaffold'], d['floppy'])
            rows.append((d['scaffold'], d['floppy'], d['gen_mech_scaffold'], d['gen_mech_floppy'],
                         d['unlifted_mech_median'], f'{lo:.3f}--{hi:.3f}', f'{sp:.3f}' + note, cls))
        else:
            rows.append((d['a'], d['b'], '--', 0, '--', '--', '--', 'skip'))
    order = {'F': 0, 'N': 1, '': 2, 'skip': 3}
    rows.sort(key=lambda r: (order[r[7]], r[0]))
    head = 'Scaffold & Floppy & $M_S$ & $M_F$ & $U$ & band & spread & class' + EOL
    cap = ('\\caption{RCSR dual pairs whose two nets share a space group. $M_S$, $M_F$: mechanisms of the '
           'scaffold and of the floppy net per conventional cell of the scaffold at a generic wavevector. $U$: median over wavevectors of the '
           'mechanisms left unlifted by the contacts. band: range of the first-order $\\omega^2/k_c$ over all '
           'branches, unlifted ones included, clipped at 0. spread: at each of 40 random wavevectors of the '
           'conventional cell of the scaffold, with the scaffold fixed, the first-order eigenvalues are sorted into branches, '
           'unlifted ones included. The mean over branches of their range is divided by the mean of all values, and the '
           'result is stored to four decimals. Tables~\\ref{tab:full_flat} and \\ref{tab:full_retro} use the same definition. '
           'class F: exactly flat; N: spread below 0.05; blank: neither; skip: floppy net without generic mechanisms. '
           'The spread is listed for completeness but is not meaningful when the mean band value is near zero. '
           '$^a$In ' + ', '.join(f'{net(r[0])}$\\otimes${net(r[1])}' for r in rows if r[6].endswith('$^a$')) + ' '
           'all but one mechanism stay unlifted and every first-order value lies below $10^{-4}\\kc$, so the band '
           'rounds to zero while the spread divides a small width by a mean near zero.}'
           '\\label{tab:screen}' + EOL)
    out = ['\\begin{longtable}{llrrrlrl}', cap,
           HL + ' ' + head + HL + '\\endfirsthead',
           HL + ' ' + head + HL + '\\endhead']
    for r in rows:
        out.append(f'{net(r[0])} & {net(r[1])} & {r[2]} & {r[3]} & {r[4]} & {r[5]} & {r[6]} & {r[7]}' + EOL)
    out += [HL, '\\end{longtable}']
    (OUT / 'tab_screen.tex').write_text('\n'.join(out), encoding='utf8')
    return rows


def scaling_table():
    q = J('second_order/nbo_scaling.json')
    out = ['\\begin{table}[h]\\centering',
           '\\caption{Deviation of the \\net{nbo} band from $2\\kc$. The median reports $|\\omega^2-2\\kc|/\\kc$ for '
           'wavevectors more than 0.25 reduced units from $\\Gamma$. The column dev.$/\\kc$ divides it by $\\kc$. '
           'The last column gives the fraction of assigned modes within 1\\% of $2\\kc$, including the zone centre. '
           'Each sample contains 1,500 random wavevectors and six assigned modes per wavevector.}\\label{tab:scaling}',
           '\\begin{tabular}{lcccc}', HL,
           'Scaffold & $k_c$ & median dev. & dev.$/k_c$ & within 1\\%' + EOL, HL]
    for lab, name in [('bcu', '\\net{bcu}'), ('bcu+axial', '\\net{bcu} with axial bonds')]:
        for k, v in sorted(q[lab].items(), key=lambda x: float(x[0])):
            m = v['median_dev_far_from_Gamma']
            out.append(f'{name} & {float(k):g} & {m:.5f} & {m / float(k):.3f} & {v["frac_within_1pct"]:.3f}' + EOL)
    out += [HL, '\\end{tabular}', '\\end{table}']
    (OUT / 'tab_scaling.tex').write_text('\n'.join(out), encoding='utf8')


def control_table():
    dc = J('controls/dia_dia.json')
    hb = J('planar_pair/hcb_band_quantiles.json')
    out = ['\\begin{table}[h]\\centering',
           '\\caption{Lifted bands of the homo-interpenetrated self-dual pair \\net{dia}$\\otimes$\\net{dia} '
           'and of the planar dual pair \\net{hxl}$\\otimes$\\net{hcb}: 5\\%, 50\\% and 95\\% quantiles of '
           '$\\omega^2/k_c$, and unlifted mechanisms. For \\net{dia}$\\otimes$\\net{dia}, which has several lifted '
           'branches, the quantiles pool all lifted modes over the sampled wavevectors. For \\net{hxl}$\\otimes$\\net{hcb}, '
           'these are single-band quantiles over 400 random wavevectors. At each wavevector we select the eigenvalue '
           'with the largest overlap with the \\net{hcb} mechanism, as for the markers of Fig.~\\ref{fig:planar}b.}\\label{tab:control}',
           '\\begin{tabular}{lccc}', HL,
           'Pair & $k_c$ & quantiles of $\\omega^2/k_c$ & unlifted' + EOL, HL]
    for k, v in dc.items():
        a, b, c = v['q05_q50_q95']
        out.append(f'\\net{{dia}}$\\otimes$\\net{{dia}} & {float(k):g} & {a:.3f}, {b:.3f}, {c:.3f} & {v["unlifted_median"]}' + EOL)
    for k in ['0.003', '0.01', '0.03']:
        a, b, c = hb[k]['q05_q50_q95']
        out.append(f'\\net{{hxl}}$\\otimes$\\net{{hcb}} & {float(k):g} & {a:.3f}, {b:.4f}, {c:.4f} & 0' + EOL)
    out += [HL, '\\end{tabular}', '\\end{table}']
    (OUT / 'tab_control.tex').write_text('\n'.join(out), encoding='utf8')


def nondual_table():
    ds = J('screen/dual_screen.json')['summary']; s = J('screen/nondual_screen.json')['summary']
    dual = {'n': ds['computed'], 'rigid': ds['rigid'], 'lifted': ds['lifted'],
            'good': len(ds['near_flat']), 'exact': len(ds['exact'])}
    nd = {'n': s['computed'], 'rigid': s['rigid'], 'lifted': s['lifted'], 'good': len(s['near_flat']), 'exact': len(s['exact'])}
    out = ['\\begin{table}[h]\\centering',
           '\\caption{Dual versus non-dual scaffolds. Non-dual controls pair each floppy net of the ten '
           'exactly flat dual pairs with 25 other nets of the same space group. Rigid: no scaffold mechanisms '
           'at a generic wavevector. Lifted: rigid and no unlifted mechanism. Near flat: lifted with spread below '
           '0.05. Exact: lifted with zero spread.}\\label{tab:nondual}',
           '\\begin{tabular}{lccccc}', HL,
           'Scaffold set & pairs & rigid & lifted & near flat & exact' + EOL, HL]
    for name, d in [('dual (pairs with a floppy net)', dual), ('non-dual controls', nd)]:
        out.append(f"{name} & {d['n']} & {d['rigid']} & {d['lifted']} & {d['good']} & {d['exact']}" + EOL)
    out += [HL, '\\end{tabular}', '\\end{table}']
    (OUT / 'tab_nondual.tex').write_text('\n'.join(out), encoding='utf8')


def bending_table():
    d = J('bending/bending.json')
    ks = ['0.01', '0.1', '1.0']
    out = ['\\begin{table}[h]\\centering',
           '\\caption{Relative band spread of the ten exactly flat pairs with joint bending added at first '
           'order, for angular springs and for beam-like straightening of collinear edges, at bending '
           'stiffness $k_b/k_c=0.01$, $0.1$ and $1$. Zero spread at $k_b=0$ for every pair.}\\label{tab:bending}',
           '\\begin{tabular}{lcccccc}', HL,
           'Pair & \\multicolumn{3}{c}{angular} & \\multicolumn{3}{c}{beam}' + EOL,
           ' & ' + ' & '.join(ks + ks) + EOL, HL]
    for k, v in d.items():
        sc, fl = k.split('|')
        cells = [f"{v['angle_' + x]:.2f}" for x in ks] + [f"{v['beam_' + x]:.2f}" for x in ks]
        out.append(f'{net(sc)}$\\otimes${net(fl)} & ' + ' & '.join(cells) + EOL)
    out += [HL, '\\end{tabular}', '\\end{table}']
    (OUT / 'tab_bending.tex').write_text('\n'.join(out), encoding='utf8')


def zone_rows():
    """Zone fractions of the eight splitting pairs and, in the last twelve rows, of fcu|flu and she-d|she, run later
    with finite_stiffness/zone_fractions.py on the same 8,192 Sobol wavevectors."""
    rows = CSV('finite_stiffness/zone_fraction_summary.csv')
    a, b = rows[:-12], rows[-12:]
    assert {(r['scaffold'], r['floppy']) for r in b} == {('fcu', 'flu'), ('she-d', 'she')}
    assert not {(r['scaffold'], r['floppy']) for r in a} & {(r['scaffold'], r['floppy']) for r in b}
    return a + b


def zero_volume_table():
    rows = zone_rows()
    pairs, data = [], {}
    for r in rows:
        k = (r['scaffold'], r['floppy'])
        if k not in pairs:
            pairs.append(k)
        data[(k, float(r['kc']), float(r['threshold']))] = float(r['volume_fraction'])
    kcs = sorted({float(r['kc']) for r in rows})
    # caption sentence on she-d (x) she at k_c = 0.01: most of the zone outside 1%, little outside 5%
    assert data[(('she-d', 'she'), .01, .01)] > .9 and data[(('she-d', 'she'), .01, .05)] < .05
    out = ['\\begin{table}[h]\\centering',
           '\\caption{Zone fractions (\\%) where any assigned lifted branch deviates from its first-order squared frequency by more '
           'than 1\\%; parentheses give the 5\\% threshold. Full-composite spectra use 8,192 wavevectors in eight scrambled '
           'Sobol blocks, with $\\ks=1$. '
           '\\net{fcu}$\\otimes$\\net{flu} and \\net{she-d}$\\otimes$\\net{she}, which split on no measure-zero set, '
           'were computed with the same script and wavevectors. Where the 1\\% share is large but the 5\\% share small, '
           'as for \\net{she-d}$\\otimes$\\net{she} at $\\kc=0.01$, the lifted branches deviate by between 1 and 5\\% '
           'over most of the zone.}\\label{tab:zerovol}',
           '\\begin{tabular}{l' + 'c' * len(kcs) + '}', HL,
           'Pair & ' + ' & '.join(f'$\\kc={k:g}$' for k in kcs) + EOL, HL]
    for k in pairs:
        cells = [f"{100*data[(k, kc, 0.01)]:.1f} ({100*data[(k, kc, 0.05)]:.1f})" for kc in kcs]
        out.append(f'{net(k[0])}$\\otimes${net(k[1])} & ' + ' & '.join(cells) + EOL)
    out += [HL, '\\end{tabular}', '\\end{table}']
    (OUT / 'tab_zerovol.tex').write_text('\n'.join(out), encoding='utf8')


def unbiased_table():
    h = J('controls/unbiased_control.json')
    s = h['summary']
    out = ['\\begin{table}[h]\\centering',
           '\\caption{Unbiased control: 32 floppy nets drawn uniformly from those with $\\nu>0$ and a same-group dual, each '
           'caged by its dual and by three nets of the same space group drawn uniformly. Lifted: rigid and every mechanism '
           'lifted. Exactly flat: lifted with a first-order spread of zero.}\\label{tab:h5}',
           '\\begin{tabular}{lcccc}', HL, 'Cages & pairs & rigid & lifted & exactly flat' + EOL, HL]
    for name, key in [('dual', 'dual'), ('same space group, non-dual', 'nondual')]:
        d = s[key]
        out.append(f"{name} & {d['computed']} & {d['rigid']} & {d['fully_lifted']} & {d['exact_flat']}" + EOL)
    out += [HL, '\\end{tabular}', '\\end{table}']
    (OUT / 'tab_h5.tex').write_text('\n'.join(out), encoding='utf8')




def _hkl(v):
    return '[' + ''.join(f'\\bar{{{-int(x)}}}' if int(x) < 0 else str(int(x)) for x in v) + ']'


def _plane(v, c):
    terms = []
    for x, a in zip(v, 'xyz'):
        x = int(x)
        if x:
            terms.append(('-' if x < 0 else '+') + ('' if abs(x) == 1 else str(abs(x))) + f'q_{a}')
    e = ''.join(terms).lstrip('+')
    return (f'${e}={c:g}$').replace('=0.5$', '=\\tfrac12$')


def _stratum(s):
    if s.startswith('line_'):
        return 'line', '$' + _hkl(s[5:].split(',')) + '$'
    if s.startswith('plane_'):
        v, c = s[6:].split('=')
        return 'plane', _plane(v.split(','), float(c))
    if s.startswith('BZ_edge_BZ_face_'):
        return 'zone-face edge set', '$\\partial' + _hkl(s[16:].split(',')).replace('[', '(').replace(']', ')') + '$'
    if s.startswith('BZ_face_'):
        return 'zone face', '$' + _hkl(s[8:].split(',')).replace('[', '(').replace(']', ')') + '$'
    return 'other', s


def _frac(x):
    from fractions import Fraction
    f = Fraction(x).limit_denominator(256)
    if f.denominator == 1:
        return str(f.numerator)
    return ('-' if f < 0 else '') + f'\\tfrac{{{abs(f.numerator)}}}{{{f.denominator}}}'


def _spec(vals):
    out, grp = [], []
    for v in sorted(max(0.0, x) for x in vals):
        if grp and abs(v - grp[-1][0]) < 1e-3:
            grp[-1][1] += 1
        else:
            grp.append([v, 1])
    for v, n in grp:
        t = f'{v:.3f}'.rstrip('0').rstrip('.') if abs(v - round(v)) > 1e-9 else f'{round(v):d}'
        out.append(t + (f'^{{\\times{n}}}' if n > 1 else ''))
    return ', '.join('$' + o + '$' for o in out)          # separate math groups so the cell can wrap


def split_table():
    """Where the generic flat band of each of the ten exactly flat pairs splits. A set is listed when the
    scaffold has mechanisms at every scanned non-Gamma point of it (high_symmetry/zero_set_catalog.csv) and the
    generic flat value loses multiplicity at >= 90% of those points (high_symmetry/scans/); pairs without such
    a line or plane list their isolated split points. Floppy-side splitting with a rigid scaffold is read
    from the scan of the second implementation (high_symmetry/floppy_side_scan.json)."""
    from collections import defaultdict
    cat = defaultdict(list)
    for r in CSV('high_symmetry/zero_set_catalog.csv'):
        cat[r['pair']].append(r['stratum'])
    fv = J('high_symmetry/floppy_side_scan.json')
    order = ['bcu|nbo', 'hxg-d|hxg', 'nia|nia-d', 'she-d|she', 'bsn|lvt', 'qtz-x|uom', 'pcu|urk',
             'hex|bnn', 'fcu|flu', 'acs|gra']
    head = ' Pair & Side & Split set & Example $\\mathbf q$ & $M_S$, $M_F$ & Spectrum' + EOL
    out = ['\\begingroup\\small', '\\begin{longtable}{llllcl}',
           '\\caption{Measure-zero sets on which the generic flat band of the ten exactly flat pairs splits. '
           'The scans cover 13 low-index lines, low-index planes and the faces and edges of the primitive zone, '
           'a face $(hkl)$ being labelled by its outward normal and $\\partial(hkl)$ denoting the set of its edges, '
           'with the scaffold free (joint first-order spectrum). Wavevectors are reduced coordinates of the '
           'conventional reciprocal cell. Side: scaffold, the scaffold has mechanisms on the set; floppy, the '
           'scaffold is rigid and the floppy net gains a mechanism. $M_S$, $M_F$: mechanisms of the scaffold and '
           'of the floppy net per conventional cell of the scaffold at the example wavevector; $M_F$ above its generic value means that the floppy net also gains mechanisms there. Spectrum: first-order $\\omega^2/\\kc$ there, with '
           'multiplicities; zeros are mechanisms left unlifted. A dash means that no split was found on the '
           'scanned sets. Extra floppy mechanisms do not always split the original flat band. '
           'When the scaffold remains rigid and the value and multiplicity of the band persist, the corresponding sets are omitted. '
           'For \\net{bcu}$\\otimes$\\net{nbo}, this occurs at 1,133 of 2,909 scanned points. On its $\\langle100\\rangle$ lines, '
           'four extra mechanisms per conventional cell remain at zero frequency. There are also 14 such points each for '
           '\\net{hxg-d}$\\otimes$\\net{hxg} and \\net{she-d}$\\otimes$\\net{she}.}\\label{tab:split}' + EOL,
           HL + head + HL + '\\endfirsthead',
           HL + head + HL + '\\endhead']
    summary = {}
    for p in order:
        sc, fl = p.split('|')
        rows = J(f'high_symmetry/scans/{sc}__{fl}.json')['rows']
        rows = [r for r in rows if not all(abs(x - round(x)) < 1e-9 for x in r['q'])]

        def split(r):
            return (r.get('generic_flat_multiplicity_missing') or 0) > 0
        full = []
        for s in cat.get(p, []):
            on = [r for r in rows if s in r['strata']]
            if on and sum(map(split, on)) >= 0.9 * len(on):
                full.append(s)
        lines = []
        if full:
            kinds = defaultdict(list)
            for s in full:
                k, t = _stratum(s)
                kinds[k].append(t)
            rank = ['line', 'plane', 'zone face', 'zone-face edge set']
            desc = '; '.join(f"{len(kinds[k])} {k}{'s' if len(kinds[k]) > 1 else ''}: " + ', '.join(kinds[k])
                             for k in rank if k in kinds)
            on = [r for r in rows if full[0] in r['strata'] and split(r)]
            ex = min(on, key=lambda r: sum(abs(abs(x) - 0.25) for x in r['q'] if abs(x) > 1e-9))
            lines.append(('scaffold', desc, ex['q'], ex['M_S'], ex['M_F'], ex['joint_spectrum']))
        else:
            pts = [r for r in rows if split(r)]
            if pts:
                ex = pts[0]
                lines.append(('scaffold', f'{len(pts)} isolated zone-boundary points, on $q_z=0$ and on the '
                              f'faces $q_z=\\pm\\tfrac12$', ex['q'], ex['M_S'], ex['M_F'], ex['joint_spectrum']))
        f = fv.get(p)
        if f and f['n_break_with_rigid_scaffold'] > 0:
            w = f['witnesses'][0]
            m = min(abs(x) for x in w['q'] if abs(x) > 1e-12)
            d = [round(x / m) for x in w['q']]
            lines.append(('floppy', f"line ${_hkl(d)}$ and parts of planes; the band splits at "
                                    f"{f['n_break_with_rigid_scaffold']} of the {f['n_q_MF_gt_P']} scanned "
                                    f"points with extra floppy mechanisms",
                          w['q'], w['M_S'], w['M_F'], w['spectrum']))
        summary[p] = [l[0] for l in lines]
        name = f'{net(sc)}$\\otimes${net(fl)}'
        if not lines:
            out.append(f'{name} & -- & -- & -- & -- & --' + EOL)
        for j, (side, desc, q, ms, mf, sp) in enumerate(lines):
            qs = '$(' + ',\\,'.join(_frac(x) for x in q) + ')$'
            # REVTeX breaks p{} columns in longtable, so long cells are top-aligned parboxes in l columns
            out.append(f"{name if j == 0 else ''} & {side} & \\parbox[t]{{5.4cm}}{{\\raggedright {desc}}} & {qs} & "
                       f"{ms}, {mf} & \\parbox[t]{{2.9cm}}{{\\raggedright {_spec(sp)}}}" + EOL)
    out += [HL, '\\end{longtable}', '\\endgroup']
    (OUT / 'tab_split.tex').write_text('\n'.join(out), encoding='utf8')
    return summary


def gain_macros():
    """Counts of the drive paragraph and abstract, from drive/gain/*.json and drive/gain_special_q.json.
    Window: |g/omega_ref / (h/4) - 1| < 4% at Omega = 2 omega_ref; positive: g/omega_ref > 1e-5."""
    out = ['% generated by make_tables.py from the result files; do not edit']
    name = {'nbo': 'Nbo', 'flu': 'Flu', 'hcb': 'Hcb', 'dia': 'Dia'}
    pct = []
    for m in ['nbo', 'flu', 'hcb', 'dia']:
        d = J(f'drive/gain/{m}.json')
        i = d['Omega_over_reference'].index(2.0)
        g = [row[i] for row in d['gain_over_reference']]
        ref = d['h'] / 4
        win = sum(abs(x / ref - 1) < 0.04 for x in g)
        pos = sum(x > 1e-5 for x in g)
        out += [f'\\newcommand{{\\gainWin{name[m]}}}{{{win}}}', f'\\newcommand{{\\gainPos{name[m]}}}{{{pos}}}']
        if m != 'dia':
            pct.append(100 * win / len(g))
        nq = len(g)
    out += [f'\\newcommand{{\\gainNq}}{{{nq}}}',
            f'\\newcommand{{\\gainPctMin}}{{{min(pct):.0f}}}', f'\\newcommand{{\\gainPctMax}}{{{max(pct):.0f}}}']
    sp = J('drive/gain_special_q.json')
    gf = next(e for e in sp if e['model'] == 'flu' and e['label'] == 'Gamma')
    ref = J('drive/gain/flu.json')['h'] / 4
    out += [f"\\newcommand{{\\gainGammaFlu}}{{{gf['gain_at_ratio_2']:.4f}}}",
            f"\\newcommand{{\\gainGammaFluPct}}{{{100 * gf['gain_at_ratio_2'] / ref:.0f}}}"]
    for m in ['nbo', 'hcb']:
        e = next(e for e in sp if e['model'] == m and e['label'] == 'Gamma')
        assert e['gain_at_ratio_2'] < 1e-6 and e['adaptive_gain_at_ratio_2'] < 1e-6, m
    (OUT / 'gain_values.tex').write_text('\n'.join(out) + '\n', encoding='utf8')
    return out


def sm_macros():
    """Further numbers quoted in the text: floppy-side split value of bsn(x)lvt, gain of the other
    flat pairs, finite-stiffness zone fractions, classifier error bound and the planar second-order coefficient."""
    out = ['% generated by make_tables.py from the result files; do not edit']
    # 1. bsn (x) lvt: the value a lifted band moves to where the floppy net gains a ninth mechanism
    w = J('high_symmetry/high_symmetry_witnesses.json')
    vals = sorted({round(x, 6) for e in w if e['pair'] == 'bsn|lvt' for v in e['witnesses'].values()
                   for x in (v.get('fixed_spectrum') or []) if x > 1.7})
    # the sets also hold other values (1.719, 1.895) on scaffold zero sets; the floppy-side value is the
    # one that the separate fixed-scaffold scan finds on the [112] line with a rigid scaffold
    ref = max(J('high_symmetry/floppy_side_scan.json')['bsn|lvt']['witnesses'][0]['spectrum'])
    v = min(vals, key=lambda x: abs(x - ref))
    assert abs(v - ref) < 1e-4, (v, ref)
    out.append(f'\\newcommand{{\\lvtSplit}}{{{v:.3f}}}')
    # 2. gain of the other flat pairs at k_c = 1e-3 (Sobol 256 wavevectors, same drive)
    rows = [r for r in CSV('finite_stiffness/gain_other_pairs_summary.csv')
            if abs(float(r['kc']) - 1e-3) < 1e-12 and r['scaffold'] != 'bcu']
    fr = [float(r['gain_q_fraction']) for r in rows]
    mg = [float(r['mean_gain_fraction_of_Mathieu']) for r in rows]
    low = min(rows, key=lambda r: float(r['mean_gain_fraction_of_Mathieu']))
    out += [f'\\newcommand{{\\otherPairs}}{{{len({(r["scaffold"], r["floppy"]) for r in rows})}}}',
            f'\\newcommand{{\\otherNq}}{{{rows[0]["nq"]}}}',
            f'\\newcommand{{\\otherFracMin}}{{{100 * min(fr):.0f}}}', f'\\newcommand{{\\otherFracMax}}{{{100 * max(fr):.0f}}}',
            f'\\newcommand{{\\otherMeanMin}}{{{min(mg):.3f}}}', f'\\newcommand{{\\otherMeanMax}}{{{max(mg):.3f}}}',
            f'\\newcommand{{\\otherLow}}{{\\net{{{low["scaffold"]}}}$\\otimes$\\net{{{low["floppy"]}}}}}']
    # 3. zone fraction outside 1% at k_c = 1e-3 (Table S5)
    zv = {(r['scaffold'], r['floppy']): float(r['volume_fraction'])
          for r in zone_rows()
          if abs(float(r['kc']) - 1e-3) < 1e-12 and abs(float(r['threshold']) - 0.01) < 1e-12}
    out += [f"\\newcommand{{\\zvBcu}}{{{100 * zv[('bcu', 'nbo')]:.1f}}}",
            f"\\newcommand{{\\zvHex}}{{{100 * zv[('hex', 'bnn')]:.1f}}}"]
    # same fractions for the zone-fraction column of Table I, \zvT{scaffold}{floppy}, now for all ten flat pairs;
    # a pair missing from the scan would print as a dash
    out.append('\\newcommand{\\zvT}[2]{\\ifcsname zv@#1@#2\\endcsname\\csname zv@#1@#2\\endcsname\\else--\\fi}')
    for (sc, fl), x in zv.items():
        out.append(f'\\expandafter\\def\\csname zv@{sc}@{fl}\\endcsname{{{100 * x:.1f}}}')
    # interpenetrated dia: 5-95% range of the lifted band over the three k_c of Table S3
    dc = J('controls/dia_dia.json')
    out += [f"\\newcommand{{\\diaLo}}{{{min(v['q05_q50_q95'][0] for v in dc.values()):.2f}}}",
            f"\\newcommand{{\\diaHi}}{{{max(v['q05_q50_q95'][2] for v in dc.values()):.1f}}}"]
    # 4. one-sided 95% bound on the error rate among the prospective flat calls
    cr = J('classifier/classifier_results.json')['cohorts']['prospective']
    out.append(f"\\newcommand{{\\falseCallBound}}{{{100 * cr['false_prediction_upper95_one_sided']:.0f}}}")
    # 5. planar pair: g~ along Gamma-M in the limit k_c -> 0 and the 1/q^2 prefactor
    g = J('planar_pair/gtilde_values.json')
    for key, nm in [('1', 'One'), ('0.6', 'Six'), ('0.3', 'Three')]:
        a, b = g['g_at_points'][key]
        assert abs(a - b) / b < 0.01, (key, a, b)               # converged in k_c
        out.append(f'\\newcommand{{\\gt{nm}}}{{{b:.1f}}}')
    out += [f"\\newcommand{{\\gtC}}{{{g['C_inverse_square']:.1f}}}",
            f"\\newcommand{{\\gtSlope}}{{{-g['loglog_slope']:.1f}}}",
            f"\\newcommand{{\\gtFitLo}}{{{g['fit_range'][0]:g}}}", f"\\newcommand{{\\gtFitHi}}{{{g['fit_range'][1]:g}}}",
            f"\\newcommand{{\\gtFitRes}}{{{100 * g['fit_max_rel_residual']:.0f}}}"]
    hb = J('planar_pair/hcb_band_quantiles.json')
    f1 = J('planar_pair/planar_band.json')
    out += [f"\\newcommand{{\\hcbMedian}}{{{hb['0.003']['q05_q50_q95'][1]:.4f}}}",
            f"\\newcommand{{\\hcbFigMedian}}{{{f1['zone_median_w2_over_kc']:.4f}}}"]
    (OUT / 'sm_values.tex').write_text('\n'.join(out) + '\n', encoding='utf8')
    return out


def orbital_table():
    """p-orbital crystal-field demonstration (SM Section S9): first-order shift per unit field strength,
    its spread over the sampled wavevectors, and the full-diagonalization bandwidth at lambda = 0.03."""
    s = J('orbital/summary.json')
    lam = s['lambda_']

    def rng(a):
        lo, hi = a
        return f'{lo:.3f}' if hi - lo < 1e-9 else f'{lo:.3f}--{hi:.3f}'

    def sci(x):
        if x < 1e-12:
            return '$<10^{-12}$'
        if x >= 1e-3:
            return f'{x:.3f}'
        m, e = f'{x:.1e}'.split('e')
        return f'${m}\\times10^{{{int(e)}}}$'

    def fw(key):
        v = s[key]
        return max(v['full_branch_widths']) if 'full_branch_widths' in v else v['full_width']

    def pw(key):
        v = s[key]
        return max(v['first_branch_widths']) if 'first_branch_widths' in v else v['first_width']
    rows = [
        ('\\net{hcb}', 'common scalar, $\\one$ on every site', 'scalar', 'hcb_scalar'),
        ('\\net{hcb}', 'triangular dual star, $\\mathrm{diag}(\\tfrac32,\\tfrac32,0)$', 'U', 'hcb_U'),
        ('\\net{hcb}', 'anisotropic, $\\mathrm{diag}(2.1,0.9,0)$', 'none', 'hcb_U_broken'),
        ('\\net{flu}', 'fcc dual stars, $\\tfrac43\\one$ and $2\\one$', 'C', 'flu_C'),
        ('\\net{flu}', 'plus $0.4\\,\\mathrm{diag}(1,-1,0)$ on every site', 'none', 'flu_C_broken'),
        ('Lieb', 'common scalar on both majority sites', 'scalar', 'Lieb_scalar'),
        ('Lieb', 'unequal scalars 1.4 and 0.6', 'none', 'Lieb_unequal'),
    ]
    out = ['\\begin{table}[h]\\centering',
           '\\caption{Electronic crystal-field test of the rule. The \\net{hcb} in-plane and \\net{flu} rows use spinless $p$ '
           'orbitals with $\\sigma$ hopping and the on-site energies that make the Bloch matrix $R^\\dagger R$. The Lieb rows '
           'use the zero band of the Lieb lattice, and the \\net{hcb} $p_z$ row uses the two decoupled $p_z$ bands placed at '
           'energy 4 (Section~S9.2). The field is $\\lambda\\bar S$, with $\\bar S$ the dimensionless contact tensor. Route: the condition '
           'of Section~S2 that applies (none: no condition holds). Shift: first-order energy shift per unit $\\lambda$ '
           'over the sampled wavevectors. Width: spread of the first-order shift, and bandwidth of the full '
           f'diagonalization at $\\lambda={lam:g}$ (largest over branches). Energies in units of the $\\sigma$ hopping.}}'
           '\\label{tab:orbital}',
           '\\begin{tabular}{lllccc}', HL,
           'Net & Field & Route & Shift & First-order width & Full width' + EOL, HL]
    for net_, field, route, key in rows:
        v = s[key]
        out.append(f"{net_} & {field} & {route} & {rng(v['first_range'])} & {sci(pw(key))} & {sci(fw(key))}" + EOL)
    L = s['L']
    out.append(f"\\net{{hcb}} $p_z$ & tilted, $p_z$ not an eigenvector & L & {L['first_eigenvalues'][0]:.3f} & "
               f"$<10^{{-12}}$ & {sci(L['full_width_at_lambda_0p03'])}" + EOL)
    out += [HL, '\\end{tabular}', '\\end{table}']
    (OUT / 'tab_orbital.tex').write_text('\n'.join(out), encoding='utf8')
    # macros quoted in the text
    v = J('independent_check/orbital_bands.json')
    # the separate implementation (independent_check/orbital_bands.py) must agree with the values tabulated here
    assert abs(v['hcb_U_lowest_over_lambda_range'][0] - 1.5) < 1e-9 and abs(v['hcb_U_lowest_over_lambda_range'][1] - 1.5) < 1e-9
    assert all(abs(x - s['flu_C']['first_range'][0]) < 1e-9 for x in v['flu_first_order_range'])
    assert v['flu_exact_lowest4_minus_Eminus_max'] < 1e-12
    assert abs(v['hcb_broken_first_order_range'][0] - s['hcb_U_broken']['first_range'][0]) < 1e-3
    lo, hi = s['hcb_U_broken']['first_range']
    sl = s['scaling_slopes']
    h5 = J('controls/unbiased_control.json')['summary']
    mac = ['% generated by make_tables.py from the result files; do not edit',
           f'\\newcommand{{\\orbLambda}}{{{lam:g}}}',
           f'\\newcommand{{\\orbAnisoLo}}{{{lo:.1f}}}', f'\\newcommand{{\\orbAnisoHi}}{{{hi:.1f}}}',
           f"\\newcommand{{\\orbFluWidthBroken}}{{{max(s['flu_C_broken']['first_branch_widths']):.2f}}}",
           f"\\newcommand{{\\orbSlopeU}}{{{sl['U_broken_max_error']:.2f}}}",
           f"\\newcommand{{\\orbSlopeL}}{{{sl['L_mix_max_branch_width']:.2f}}}",
           f"\\newcommand{{\\orbLvalue}}{{{L['first_eigenvalues'][0]:.1f}}}",
           # the tilted field of the hcb p_z test (one site block of honey_S_L), quoted in SM S9.2
           '\\newcommand{\\orbSL}{\\begin{pmatrix}' + '\\\\'.join(
               '&'.join(f'{x:g}' for x in row[:3]) for row in J('orbital/model.json')['honey_S_L'][:3]) + '\\end{pmatrix}}',
           f"\\newcommand{{\\rigidDual}}{{{h5['dual']['rigid']}}}", f"\\newcommand{{\\nDual}}{{{h5['dual']['computed']}}}",
           f"\\newcommand{{\\rigidOther}}{{{h5['nondual']['rigid']}}}", f"\\newcommand{{\\nOther}}{{{h5['nondual']['computed']}}}",
           f"\\newcommand{{\\pctDual}}{{{100 * h5['dual']['rigidity_rate']:.0f}}}",
           f"\\newcommand{{\\pctOther}}{{{100 * h5['nondual']['rigidity_rate']:.0f}}}"]
    # fixed-cage exactness (SM S2.6): sampled bandwidths of the exact routes and second-order widths of the
    # non-eigenvector route-L pairs (fixed_cage/fixed_cage.py)
    import math
    bw = CSV('fixed_cage/bandwidths.csv')
    wmax = max(float(r['bandwidth']) for r in bw)
    e = math.floor(math.log10(wmax)); m = math.ceil(wmax / 10 ** e)
    fc = J('fixed_cage/results.json')
    tb = fc['sigma_TB']
    assert abs(tb['expected'] - 4 * math.sqrt(2) / 3) < 1e-12 and max(tb['positive_bandwidth'], tb['negative_bandwidth']) < 1e-13
    mac += [f'\\newcommand{{\\exactWidth}}{{{m}\\times10^{{{e}}}}}',
            f"\\newcommand{{\\exactNq}}{{{int(bw[0]['sample_count']):,}}}",
            # used in math mode ($\kc/k_F=\exactKcs$), so the conjunction is set as text with its spaces
            f"\\newcommand{{\\exactKcs}}{{{', '.join(f'{float(k):g}' for k in sorted({r['kc'] for r in bw}, key=float)[:-1])}"
            f"\\text{{ and }}{float(sorted({r['kc'] for r in bw}, key=float)[-1]):g}}}",
            f"\\newcommand{{\\lvtCtwo}}{{{fc['lvt']['coefficient_domains']['0.1']['maximum_branch_width_coefficient']:.2f}}}",
            f"\\newcommand{{\\uomCtwo}}{{{fc['uom']['coefficient_domains']['0.1']['maximum_branch_width_coefficient']:.2f}}}"]
    (OUT / 'np_values.tex').write_text('\n'.join(mac) + '\n', encoding='utf8')
    return mac


def zn_macros():
    """Zn(CN)2 numbers of SM S5.3, from zinc_cyanide/contact_rigid_units.json (rigid-unit mechanism space)."""
    d = J('zinc_cyanide/contact_rigid_units.json')
    c, b, o = d['contact_real_onsite'], d['intrinsic_band'], d['contact_operator']
    assert max(c['branch_width']) < 1e-12 and min(o['branch_width']) > 0.1      # on-site branches flat, full operator dispersive
    out = [f"\\newcommand{{\\znContactMean}}{{{c['mean']:.2f}}}", f"\\newcommand{{\\znContactRel}}{{{c['rel_spread']:.1f}}}",
           f"\\newcommand{{\\znContactRange}}{{{c['abs_spread']:.2f}}}",
           f"\\newcommand{{\\znIntrinsicMean}}{{{b['mean']:.0f}}}", f"\\newcommand{{\\znRatio}}{{{b['mean'] / c['mean']:.0f}}}",
           f"\\newcommand{{\\znOperatorMean}}{{{o['mean']:.2f}}}", f"\\newcommand{{\\znOperatorRel}}{{{o['rel_spread']:.1f}}}"]
    (OUT / 'zn_values.tex').write_text('\n'.join(out) + '\n', encoding='utf8')
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', help='archive root with the result files (default: the folder above tables/)')
    ap.add_argument('--out', help='output directory for the .tex files (default: ROOT/outputs/tables)')
    a = ap.parse_args(argv); configure(a.root, a.out)
    print('\n'.join(zn_macros()))
    print('\n'.join(gain_macros()))
    print('\n'.join(orbital_table()))
    print('\n'.join(sm_macros()))
    zero_volume_table()
    unbiased_table()
    rows = screen_table()
    scaling_table()
    control_table()
    nondual_table()
    bending_table()
    print('split sides:', split_table())
    print('written: tab_screen.tex (%d rows), tab_scaling.tex, tab_control.tex' % len(rows))
    print('written to', OUT)


if __name__ == '__main__':
    main()
