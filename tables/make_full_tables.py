"""Full data-disclosure tables for the Supplemental Material (pair-by-pair and row-by-row).
Every number is read from a result file; nothing is copied by hand. The checks at the end assert that the
table totals reproduce the counts quoted in the abstract, main text and supplement (macros in *_values.tex).
RCSR coordinates are not redistributed: only net names, symmetry symbols and derived statistics are printed.

  python tables/make_full_tables.py --rcsr DIR [--root DIR] [--out DIR]
writes full_*.tex, prot_values.tex and si_data_tables.tex into the output directory of make_tables.py (run it first).
The RCSR file is read only for the space-group symbols of the six pairs without generic mechanisms."""
import argparse
import hashlib
import json
import sys
import math
import re
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import make_tables as T
from make_tables import net, EOL, HL, J, CSV
from common import nets as E

EXPECTED_SHA = 'cdb601e1c1c00d92b18a33aa3576f4981b5535a88978aeb077a8e5e44c8711d7'
REPORT = []          # findings printed at the end (mismatches with the manuscript text)


def _s(v):
    """A stored value as it reads from a CSV file of the result tables."""
    if v is None: return ''
    if isinstance(v, (list, dict)): return json.dumps(v)
    return str(v)


def classification_table():
    """Predictions joined with outcomes for the retrospective rows and the first prospective test."""
    rows = []
    for cohort in ('retrospective', 'prospective'):
        outs = {o['id']: o for o in J(f'classifier/outcomes_{cohort}.json')['rows']}
        for r in J(f'classifier/predictions_{cohort}.json')['rows']:
            pr = r['prediction']; o = outs[r['id']]
            x = dict(cohort=r['cohort'], id=str(r['id']), scaffold=r['scaffold'], floppy=r['floppy'], route=_s(pr.get('route')),
                     prediction=pr['prediction'], values=json.dumps(pr.get('values', [])))
            x.update({k: _s(v) for k, v in o.items() if k not in x})
            rows.append(x)
    return rows


def macros(*files):
    m = {}
    for f in files:
        for line in (T.OUT / f).read_text(encoding='utf8').splitlines():
            g = re.match(r'\\newcommand\{\\(\w+)\}\{(.*)\}$', line)
            if g:
                m[g.group(1)] = g.group(2)
    return m


MAC = {}          # filled in main() from the macro files written by make_tables.py


def pair(sc, fl):
    return f'{net(sc)}$\\otimes${net(fl)}'


def val(x, den=64, dec=3):
    """Exact fraction only when the number is that fraction to 1e-9; otherwise a decimal."""
    x = 0.0 if abs(x) < 1e-12 else x
    f = Fraction(x).limit_denominator(den)
    if abs(x - float(f)) < 1e-9:
        if f.denominator == 1:
            return f'{f.numerator}'
        s = '-' if f < 0 else ''
        return f'{s}\\tfrac{{{abs(f.numerator)}}}{{{f.denominator}}}'
    if abs(x) < 0.01:
        m, e = f'{x:.2e}'.split('e')
        return f'{m}\\times10^{{{int(e)}}}'
    return f'{x:.{dec}f}'


def distinct_dec(vals, den=64, dec=3):
    """Smallest number of decimals (at least dec) at which distinct values also print differently,
    so that close but different values (e.g. 1.93066 and 1.93073) are not shown as one number twice."""
    while dec < 8 and len({val(v, den, dec) for v in vals}) < len(vals):
        dec += 1
    return dec


def grouped(vals, tol=1e-6, den=64, dec=3):
    """Values with multiplicities as separate math groups, so that a cell can wrap."""
    grp = []
    for v in sorted(vals):
        if grp and abs(v - grp[-1][0]) < tol:
            grp[-1][1] += 1
        else:
            grp.append([v, 1])
    dec = distinct_dec([v for v, n in grp], den, dec)
    return ', '.join('$' + val(v, den, dec) + (f'\\,(\\times{n})' if n > 1 else '') + '$' for v, n in grp)


def sci(x):
    if x == 0:
        return '0'
    m, e = f'{x:.1e}'.split('e')
    return f'${m}\\times10^{{{int(e)}}}$'


def _screw_map():
    """Unsubscripted short Hermann-Mauguin symbol -> spglib symbol with screw subscripts ('P6222' -> 'P6_222')."""
    import spglib
    m = {}
    for h in range(1, 531):
        t = spglib.get_spacegroup_type(h)
        s = t.international_short if hasattr(t, 'international_short') else t['international_short']
        assert m.setdefault(s.replace('_', ''), s) == s
    return m


_SCREW = _screw_map()


def sg(s):
    """Hermann-Mauguin symbol in math mode: '-6m2' -> \\bar6m2, 'P63/mmc' -> P6_3/mmc, 'P6222' -> P6_222.
    Space-group screw axes come from the spglib symbol table, so the subscript always sits on the screw axis."""
    if s[:1] in 'PIFCABR':          # space group; point groups (e.g. 'm-3m', '4/mmm') pass through
        base, sep, setting = s.partition(':')
        base = _SCREW[base.replace('_', '')] + sep + setting
        s = re.sub(r'_(\d)', r'_{\1}', base)
    s = re.sub(r'-(\d)', r'\\bar{\1}', s)
    return '$' + s + '$'


def longtable(name, spec, head, caption, label, rows, size='footnotesize'):
    out = [f'\\begingroup\\{size}', f'\\begin{{longtable}}{{{spec}}}',
           f'\\caption{{{caption}}}\\label{{{label}}}' + EOL,
           HL + ' ' + head + EOL + ' ' + HL + '\\endfirsthead',
           HL + ' ' + head + EOL + ' ' + HL + '\\endhead',
           HL + '\\endfoot']
    out += [r + EOL for r in rows]
    out += ['\\end{longtable}', '\\endgroup']
    text = '\n'.join(out)
    assert '?' not in caption and '\u2014' not in text, name
    (T.OUT / f'{name}.tex').write_text(text + '\n', encoding='utf8')
    return len(rows)


def per_primitive(vals, m, tol=1e-6):
    """Designated values of the classifier are counted in the conventional cell; divide each multiplicity by the number m
    of primitive cells in it, so that they refer to the common primitive cell of the observed spectra."""
    grp = []
    for v in sorted(vals):
        if grp and abs(v - grp[-1][0]) < tol:
            grp[-1][1] += 1
        else:
            grp.append([v, 1])
    assert all(n % m == 0 for _, n in grp), (grp, m)
    return [v for v, n in grp for _ in range(n // m)]


def pt(s, w):
    """Top-aligned wrapped cell for l columns (REVTeX breaks p{} columns inside longtable)."""
    return f'\\parbox[t]{{{w}}}{{\\strut\\raggedright {s}\\strut}}'


# ------------------------------------------------------------------ RCSR space groups (symbols only)
def rcsr_groups():
    g, name = {}, None
    for line in (E.data_dir() / 'rcsr3d.cgd').read_text(encoding='utf8', errors='replace').splitlines():
        t = line.split()
        if not t:
            continue
        k = t[0].upper()
        if k == 'NAME':
            name = t[1]
        elif k == 'GROUP' and name:
            g[name] = t[1]
    return g


# ------------------------------------------------------------------ 1. atlas of the 76 dual pairs
def atlas_table():
    ds = J('screen/dual_screen.json')
    coord = {(r['scaffold'], r['floppy']): r for r in J('screen/coordination.json')['rows']}
    csum = J('screen/coordination.json')['summary']
    maxw = {(r['scaffold'], r['floppy']): r for r in CSV('screen/maxwell_counting.csv')}
    site = {(r['scaffold'], r['floppy']): r for r in J('classifier/site_symmetry.json')['rows'] if r.get('cohort') == 'dual'}
    cl = {(r['scaffold'], r['floppy']): r for r in classification_table() if r['cohort'] == 'dual'}
    grp = rcsr_groups()
    rows, stages = [], Counter()
    mismatch = []
    order = {'exact': 0, 'near flat': 1, 'dispersive': 2, 'unlifted': 3, 'scaffold floppy': 4, 'no mechanisms': 5}
    data = []
    for d in ds['rows']:
        if 'scaffold' not in d:
            a, b = d['a'], d['b']
            ga, gb = grp[a], grp[b]
            assert ga == gb, (a, b)
            data.append(dict(sc=a, fl=b, stage='no mechanisms', sg=ga, skip=True))
            continue
        sc, fl = d['scaffold'], d['floppy']
        ga, gb = grp[sc], grp[fl]
        assert ga == gb, (sc, fl)
        s = site[(sc, fl)]
        assert s['space_group_floppy'] == ga, (sc, fl, s['space_group_floppy'], ga)
        rigid = d['gen_mech_scaffold'] == 0
        lifted = rigid and d['unlifted_mech_median'] == 0
        sp = d['band_spread_rel']
        stage = ('scaffold floppy' if not rigid else 'unlifted' if not lifted else
                 'exact' if sp == 0 else 'near flat' if sp < 0.05 else 'dispersive')
        c, m, q = coord[(sc, fl)], maxw[(sc, fl)], cl[(sc, fl)]
        # the classifier test computes the spectra again; it must agree on rigidity and complete lifting
        if (q['rigid'] == 'True') != rigid or (q['qualified'] == 'True') != lifted:
            mismatch.append((sc, fl))
        assert int(m['nu']) == d['gen_mech_floppy'] - int(m['self_stress']), (sc, fl)
        route = ''
        if q['prediction'] == 'predicted_flat':
            route = {'mixed_L_P2': 'L + C'}.get(q['route'], q['route']) + ('' if q['qualified'] == 'True' else '$^*$')
        gi = sorted({x['site_group_composite'] for x in s['sites']})
        data.append(dict(sc=sc, fl=fl, stage=stage, sg=ga, skip=False, gi=gi, nc=s['n_contacts'],
                         V=int(m['V_F']), E=int(m['E_F']), nu=int(m['nu']), nss=int(m['self_stress']),
                         z=c['dual_mean_degree'], eu=c['euler_consistent'], res=c['common_cell_euler_residual'],
                         fm=c['tile_mean_matches_dual_degree'],
                         route=route, rigid=rigid))
    assert not mismatch, mismatch
    data.sort(key=lambda r: (order[r['stage']], r['sc'], r['fl']))
    for r in data:
        stages[r['stage']] += 1
        if r['skip']:
            rows.append(f"{pair(r['sc'], r['fl'])} & {sg(r['sg'])} & -- & -- & -- & -- & -- & -- & -- & -- & -- & {r['stage']} & ")
            continue
        rows.append(f"{pair(r['sc'], r['fl'])} & {sg(r['sg'])} & {', '.join(sg(x) for x in r['gi'])} & {r['nc']} & "
                    f"{r['V']} & {r['E']} & {r['nu']} & {r['nss']} & {r['z']:.2f} & ${r['res']}$ & {'yes' if r['fm'] else 'no'} & "
                    f"{r['stage']} & {r['route']}")
    comp = [r for r in data if not r['skip']]
    n_rigid = sum(r['rigid'] for r in comp)
    n_lift = sum(r['stage'] in ('exact', 'near flat', 'dispersive') for r in comp)
    assert (len(data), len(comp), n_rigid, n_lift) == (76, 70, 49, 30)
    assert stages['exact'] == 10 and stages['exact'] + stages['near flat'] == 17
    # main text: seven route L, one U, one C, one L + C among the ten exact pairs; no eligible dispersive pair designated
    assert Counter(r['route'] for r in data if r['stage'] == 'exact') == Counter({'L': 7, 'U': 1, 'C': 1, 'L + C': 1})
    assert not any(r['route'] and not r['route'].endswith('$^*$') for r in data if r['stage'] != 'exact' and not r['skip'])
    # coordination bound (main text: 49 of 50 rigid at mean degree >= 6, none of the 20 below)
    hi = [r for r in comp if r['z'] >= 6 - 1e-9]
    assert len(hi) == 50 and sum(r['rigid'] for r in hi) == 49 and len(comp) - len(hi) == 20
    assert not any(r['rigid'] for r in comp if r['z'] < 6 - 1e-9)
    assert [(r['sc'], r['fl']) for r in hi if not r['rigid']] == [('xbt', 'tof')]
    assert csum['dual_degree_at_least_6'] == 50 and csum['rigid_with_dual_degree_at_least_6'] == 49
    inc = sorted(f"{r['sc']}|{r['fl']}" for r in comp if not r['eu'])
    resid = sorted(f"{r['sc']}|{r['fl']}" for r in comp if r['res'] != 0)
    # the combined flag of coordination.json is the conjunction of the two printed columns
    assert all(r['eu'] == (r['res'] == 0 and r['fm']) for r in comp)
    face_only = sorted(f"{r['sc']}|{r['fl']}" for r in comp if r['res'] == 0 and not r['fm'])
    assert face_only == ['acs|gra', 'bcu-x-a|xag', 'fse|xux', 'jsm|uoc', 'sta|ats', 'xaf|xag', 'xbp|xbz'], face_only
    REPORT.append(f'Euler: euler_consistent=False for {len(inc)} pairs {inc}; nonzero common-cell residual for '
                  f'{len(resid)} pairs {resid}. Supplement S3.4 names four (ild|epv, pcu|urk, qzd|qtz-h, qzd|uog).')
    head = ('Pair & Group & $G_i$ & $n_c$ & $V_F$ & $E_F$ & $\\nu$ & $N_{\\rm ss}$ & $\\bar z_S$ & $\\chi$ & $\\bar f$ & Stage & Route')
    cap = ('Structural data and screening outcome of all 76 same-group dual pairs, listed as scaffold$\\otimes$floppy net. '
           'Group: common space group. $G_i$: site groups of the floppy vertices in the composite. $n_c$: contacts per '
           'conventional cell of the scaffold. $V_F$, $E_F$: floppy vertices and edges in that cell; $\\nu=dV_F-E_F$; '
           '$N_{\\rm ss}$: generic states of self-stress, so that $M_F=\\nu+N_{\\rm ss}$. $\\bar z_S$: mean coordination of '
           'the scaffold. The next two columns are the two checks of Section~S3.4. $\\chi=V_F-E_F+F-C$, with $C$ '
           'scaffold vertices and $F$ scaffold edges in the same cell, is the Euler residual, which vanishes when the '
           'scaffold is the dual of a tiling of the floppy net in that cell. $\\bar f$: whether the mean face count of the '
           'catalogue natural tiling of the floppy net equals $\\bar z_S$, as it must when the scaffold is the dual of '
           'that particular tiling. Stage: last step of the funnel passed (no mechanisms: floppy net without generic mechanisms, '
           'roles not assigned; scaffold floppy: scaffold has generic mechanisms; unlifted: rigid scaffold, some mechanisms '
           'unlifted; dispersive, near flat, exact: all mechanisms lifted, with spread at least 0.05, below 0.05 and zero). '
           'Route: route of the frozen classifier when it designates the pair flat; $^*$, designation outside the '
           'eligible set (cage not rigid or not fully lifting), which is not scored. The mechanism counts and lifted '
           'bands of the same pairs are in Table~\\ref{tab:screen}.')
    n = longtable('full_atlas', 'lllrrrrrrrlll', head, cap, 'tab:full_atlas', rows, size='scriptsize')
    return n, stages


# ------------------------------------------------------------------ 2. the 17 flat and near-flat pairs
def flat_table():
    a = CSV('high_symmetry/flat_pairs.csv')
    ds = {(r['scaffold'], r['floppy']): r for r in J('screen/dual_screen.json')['rows'] if 'scaffold' in r}
    pred = {(r['scaffold'], r['floppy']): r for r in classification_table() if r['cohort'] == 'dual'}
    rows, n_exact, dmax = [], 0, 0.0
    nqs = {J('high_symmetry/generic/' + r['pair'].replace('|', '__') + '.json')['nq'] for r in a}
    assert len(nqs) == 1
    nq = nqs.pop()
    for cls in ['exact', 'near']:
        for r in a:
            if r['original_class'] != cls:
                continue
            sc, fl = r['pair'].split('|')
            cc = json.loads(r['contact_counts'])
            se = json.loads(r['S_eigenvalues_unique'])
            stars = '; '.join(f'{c}: (' + ', '.join('$' + val(x) + '$' for x in e) + ')' for c, e in zip(cc, se))
            centres = json.loads(r['band_centers_over_kc'])
            widths = json.loads(r['band_widths_over_kc'])
            gi = ', '.join(sg(x) for x in json.loads(r['site_groups_composite']))
            mean = sum(centres) / len(centres)
            # the stored spread is the mean branch range over the mean value; the worst branch uses the same denominator
            assert abs(float(r['spread_rel']) - sum(widths) / len(widths) / mean) < 1e-9, (sc, fl)
            worst = max(widths) / mean
            if cls == 'exact':
                n_exact += 1
                om = json.loads(r['omega2_over_kc'])
                dev = max(min(abs(c - o) for o in om) for c in centres)
                p = json.loads(pred[(sc, fl)]['values'])
                dp = max(min(abs(c - x) for x in p) for c in centres)
                assert dev < 1e-9 and dp < 1e-9, (sc, fl, dev, dp)
                assert abs(float(r['spread_rel'])) < 1e-10 and ds[(sc, fl)]['band_spread_rel'] == 0
                spec = grouped(centres)
                rows.append(f"{pair(sc, fl)} & {pt('exact, ' + sci(float(r['spread_rel'])), '1.25cm')} & {gi} & {pt(stars, '3.0cm')} & {pt(spec, '2.6cm')} & "
                            f"{sci(max(widths))} & {sci(worst)} & {sci(max(dev, dp))}")
            else:
                spec = f"${float(r['eig_min']):.3f}$--${float(r['eig_max']):.3f}$ ({len(centres)} branches)"
                sp = float(r['spread_rel'])
                sp2 = ds[(sc, fl)]['band_spread_rel']
                assert sp < 0.05 and 0 < sp2 < 0.05, (sc, fl)
                dmax = max(dmax, abs(sp - sp2))
                rows.append(f"{pair(sc, fl)} & near, {sp:.3f} & {gi} & {pt(stars, '3.0cm')} & {pt(spec, '2.6cm')} & "
                            f"{max(widths):.3f} & {worst:.3f} & --")
    assert n_exact == 10 and len(rows) == 17
    head = ('Pair & Class & $G_i$ & ' + pt('Contact stars: $n_c$: eig $S_i/\\kc$', '3.0cm') + ' & ' +
            pt('First-order $\\omega^2/\\kc$', '2.6cm') + ' & Width & Worst & $\\Delta$')
    cap = ('First-order lifted bands of the ten exactly flat and seven near-flat dual pairs. Class: exact or near, with '
           'the relative spread $s$ of the second implementation. The spread is defined as in Table~\\ref{tab:screen}: the first-order eigenvalues are sorted at '
           'each wavevector into branches, the range of each branch over the wavevectors is taken, and the mean of these '
           'ranges is divided by the mean of all branch values. Contact stars: for each class of floppy vertex, the '
           'number of contacts $n_c$ and the eigenvalues of $S_i$ in units of $\\kc$, one entry per distinct contact count. First-order $\\omega^2/\\kc$: branch '
           'means over the generic wavevectors of the conventional cell of the scaffold, with multiplicities; for the near-flat pairs, the '
           'range over all branches. Width: largest range of a single branch over the sampled wavevectors, in units of '
           '$\\kc$. Worst: the largest absolute branch range divided by the mean over all branch values. '
           'This uses the same denominator as $s$ and is not the width divided by the mean of its own branch. '
           'The condition $s<0.05$ classifies an average spread, not every branch. $\\Delta$: '
           'largest difference between a numerical branch mean and the closed-form value of Table~\\ref{M-tab:atlas} or '
           'the value designated by the frozen classifier. Spreads and widths come from the second implementation, at an '
           f'independent set of {nq} random wavevectors of the conventional cell of the scaffold; for the near-flat pairs the spreads of Table~\\ref{{tab:screen}} '
           f'differ from them by at most {dmax:.3f}. Fractions are printed only where the number equals the '
           'fraction to $10^{-9}$; the values $1.645$ and $1.471$ are embedding values.')
    n = longtable('full_flat', 'lllllllr', head, cap, 'tab:full_flat', rows, size='scriptsize')
    return n


# ------------------------------------------------------------------ 3. fixed-cage exactness at every order
def exact_order_table():
    bw = CSV('fixed_cage/bandwidths.csv')
    r2 = J('fixed_cage/results.json')
    name = {'nbo': ('bcu', 'nbo', 'L'), 'bnn': ('hex', 'bnn', 'U'), 'flu': ('fcu', 'flu', 'C')}
    rows = []
    kcs = sorted({r['kc'] for r in bw}, key=float)
    wmax = 0.0
    for m, (sc, fl, rt) in name.items():
        for br in sorted({r['branch'] for r in bw if r['net'] == m}):
            sel = {r['kc']: r for r in bw if r['net'] == m and r['branch'] == br}
            cells = ' & '.join(f"{sci(float(sel[k]['bandwidth']))}" for k in kcs)
            wmax = max([wmax] + [float(sel[k]['bandwidth']) for k in kcs])
            lab = '' if m != 'flu' else (' lower root' if br == '0' else ' upper root')
            rows.append(f"{pair(sc, fl)}{lab} & {rt} & {sel[kcs[0]]['sample_count']} & {cells}")
    e = math.floor(math.log10(wmax)); mm = math.ceil(wmax / 10 ** e)
    assert MAC['exactWidth'] == f'{mm}\\times10^{{{e}}}', (MAC['exactWidth'], wmax)
    rows2 = []
    doms = sorted(set(r2['lvt']['coefficient_domains']) | set(r2['uom']['coefficient_domains']), key=float)
    for m, (sc, fl) in [('lvt', ('bsn', 'lvt')), ('uom', ('qtz-x', 'uom'))]:
        cd = r2[m]['coefficient_domains']
        def g3(x):
            """three significant digits, large values as m x 10^n rather than e-notation"""
            if x < 1e4:
                return f'{x:.3g}'
            m, e = f'{x:.2e}'.split('e')
            return f'${float(m):g}\\times10^{{{int(e)}}}$'
        cells = ' & '.join((f"{g3(cd[k]['maximum_branch_width_coefficient'])} ({cd[k]['nq']})" if k in cd else '--') for k in doms)
        rows2.append(f"{pair(sc, fl)} & L & {r2[m]['normal_S_mixing_norm']:.3f} & {cells}")
    assert f"{r2['lvt']['coefficient_domains']['0.1']['maximum_branch_width_coefficient']:.2f}" == MAC['lvtCtwo'] == '2.80'
    assert f"{r2['uom']['coefficient_domains']['0.1']['maximum_branch_width_coefficient']:.2f}" == MAC['uomCtwo'] == '4.38'
    out = ['\\begin{table}[h]\\centering\\footnotesize',
           '\\caption{Lifted bands with the scaffold held fixed, at every order in $\\kc$. Top: sampled squared-frequency '
           'width $\\Delta\\omega^2$, in units of $k_F$, of the exact bands of the three pairs covered by Lemma~\\ref{lem:exact}. Columns give '
           '$\\kc/k_F=' + ', '.join(f'{float(k):g}' for k in kcs) +
           '$ and the number $n_q$ of generic wavevectors. Bottom: two sampled route-L pairs with non-eigenvector normals. '
           'Mixing is $\\|(\\one-\\mathbf n_i\\mathbf n_i^{\\mathsf T})S_i\\mathbf n_i\\|/\\kc$, the off-normal part of '
           '$S_i\\mathbf n_i$. Entries give the largest second-order '
           'branch-width coefficient, in units of $\\kc^2/k_F$, for each stated gap cutoff. Gaps are in units of the floppy '
           'bond stiffness $k_F$. Parentheses give the number of retained wavevectors. The third non-eigenvector pair, '
           '\\net{pcu}$\\otimes$\\net{urk}, was computed separately (Section~S2.6). '
           'The other four flat dual pairs are covered by Lemma~\\ref{lem:exact}(i) or Lemma~\\ref{lem:balance_exact}, and the post hoc calculation of Section~S3.7 samples \\net{acs}$\\otimes$\\net{gra}.}\\label{tab:full_exact}',
           '\\begin{tabular}{llr' + 'c' * len(kcs) + '}', HL,
           'Pair & Route & $n_q$ & ' + ' & '.join(f'$\\kc/k_F={float(k):g}$' for k in kcs) + EOL, HL]
    out += [r + EOL for r in rows]
    out += [HL, '\\end{tabular}\\\\[4pt]', '\\begin{tabular}{llc' + 'c' * len(doms) + '}', HL,
            'Pair & Route & mixing & ' + ' & '.join(f'gap $\\ge{float(k):g}$' for k in doms) + EOL, HL]
    out += [r + EOL for r in rows2]
    out += [HL, '\\end{tabular}', '\\end{table}']
    (T.OUT / 'full_exact.tex').write_text('\n'.join(out) + '\n', encoding='utf8')
    return len(rows) + len(rows2)


# ------------------------------------------------------------------ 4. counting rule
def counting_table():
    A = J('counting/counting.json')
    # rank-resolved kernel dimension per eigenvalue of S, same pairs and clusters (counting/rank_count.py)
    RK = J('counting/rank_count.json')['rows']
    rows, by = [], {}
    for r in A['rows']:
        # names from the 'pair' field split, then forced to scaffold (x) floppy order
        p = r['pair'].split(' (')[0]
        sc, fl = [x.strip() for x in p.split('\u2297')]
        by[(sc, fl)] = r
        rk = RK[r['pair']]
        assert [(o['multiplicity'], round(o['value'], 6)) for o in r['onsite_eigenvalues']] == \
            [(x['n'], round(x['value'], 6)) for x in rk], r['pair']
        # the rank-resolved count equals the number of flat bands whose value is an eigenvalue of S
        assert sum(x['dim'] for x in rk) == sum(o['multiplicity'] for o in r['actual_flat_bands'] if o['is_S_eigenvalue']), r['pair']
        dec = distinct_dec([o['value'] for o in r['onsite_eigenvalues']])
        ons = '; '.join(f"${val(o['value'], dec=dec)}$: {o['multiplicity']}$\\to${o['count_lower_bound']}" for o in r['onsite_eigenvalues'])
        dims = '; '.join(str(x['dim']) for x in rk)
        dec = distinct_dec([o['value'] for o in r['actual_flat_bands']])
        act = '; '.join(f"${val(o['value'], dec=dec)}\\,(\\times{o['multiplicity']})$" + ('' if o['is_S_eigenvalue'] else '$^*$')
                        for o in r['actual_flat_bands'])
        coh = {'dual': 'dual', 'test_1': 'test 1', '2D_reference': 'planar'}[r['cohort']]
        route = {'mixed_L_P2': 'L + C', 'scalar_S': 'scalar', None: 'abstain'}.get(r['route'], r['route'])
        rows.append(f"{pair(sc, fl)} & {coh} & {route} & {r['N_L']} & {r['N_Ltilde']} & {pt(ons, '4.6cm')} & "
                    f"{pt(dims, '1.1cm')} & {pt(act, '2.6cm')} & {r['count_total']} & {sum(x['dim'] for x in rk)} & "
                    f"({r['classification']}) & {r['conventional_multiplicity']}")
    assert ('hxl', 'hcb') in by and ('hcb', 'hxl') not in by
    dual = [r for r in A['rows'] if r['cohort'] == 'dual']
    assert len(dual) == 10 and all(r['count_total'] == 0 for r in dual)
    assert by[('hxl', 'hcb')]['count_total'] == 1 and by[('hxl', 'hcb')]['classification'] == 'i'
    # every actual flat band is counted: sum of multiplicities equals the lifted mechanism count
    for r in A['rows']:
        assert sum(o['multiplicity'] for o in r['actual_flat_bands']) == r['M_F'], r['pair']
    nonS = sum(r['has_flat_value_not_in_S'] for r in dual)
    REPORT.append(f'counting: {nonS} of 10 dual flat pairs have a flat value that is not an S eigenvalue; '
                  f"{sum(r['has_flat_value_not_in_S'] for r in A['rows'] if r['cohort'] == 'test_1')} of 5 test-1 flat pairs")
    mv = [o['value'] for o in by[('mob', 'scp')]['onsite_eigenvalues'] if abs(o['value'] - 4 / 3) < 1e-3]
    assert len(mv) == 2
    mob = [val(x, dec=distinct_dec(mv)) for x in mv]
    head = ('Pair & Set & Route & $dV$ & $E$ & eig $S/\\kc$: $n_a\\to b_a$ & $d_a$ & Flat $\\omega^2/\\kc$ & '
            '$\\sum b_a$ & $\\sum d_a$ & Class & $m$')
    cap = ('Counting rule against the flat bands found. The rule of Ref.~\\cite{calugaru_general_2022} (App.~I.4 therein) '
           'guarantees $b_a=\\max(0,n_a-E)$ flat bands at each eigenvalue $a$ of the on-site tensor $S$ of multiplicity '
           '$n_a$, with $E$ the number of floppy edges; $dV$ is the number of displacement components. The rank-resolved '
           'count of Section~S2.7 replaces $E$ by the rank of $R_F$ on the eigenspace: $d_a=\\dim(\\mathcal E_a\\cap\\mathcal K)='
           'n_a-\\operatorname{rank}(R_FB_a)$, listed in the order of the eigenvalues and evaluated at 12 generic '
           'wavevectors, at which it does not vary. In every row $\\sum d_a$ equals the number of flat bands whose value '
           'is an eigenvalue of $S$, so the bands left over are exactly those marked $^*$. Counts are per '
           'common primitive cell; $m$ is the number of primitive cells in the conventional cell of the scaffold. Flat: '
           'flat first-order values with multiplicities, at 64 generic wavevectors; $^*$, the value is not an eigenvalue '
           f'of $S$. Close but distinct eigenvalues, such as ${mob[0]}$ and ${mob[1]}$ in \\net{{mob}}$\\otimes$\\net{{scp}}, are '
           'printed with as many digits as separate them. Set: dual, the ten '
           'flat dual pairs; test 1, the five flat pairs of the first prospective test; planar, \\net{hxl}$\\otimes$\\net{hcb}. '
           'Route: route of the frozen classifier (abstain: no designation). Class is defined by $b_a$. In (i), the bound accounts for every flat band. In (ii), '
           'it is positive but accounts for only some; no pair has this class. In (iii), the bound is zero and every flat '
           'value is an eigenvalue of $S$. In (iv), the bound is zero and at least one flat value lies outside that spectrum.')
    return longtable('full_counting', 'lllrrlllrrll', head, cap, 'tab:counting', rows, size='scriptsize')


# ------------------------------------------------------------------ 5. classifier: retrospective
def _spec_cell(r):
    means = json.loads(r['branch_means'])
    if r['shape_exact_flat'] == 'True':
        return 'flat: ' + grouped(means)
    # the same two spreads as Table tab:full_flat: mean and largest branch range over the mean branch value
    ranges = json.loads(r['branch_ranges'])
    m = abs(sum(means) / len(means))
    s, w = sum(ranges) / len(ranges) / m, max(ranges) / m
    assert abs(s - float(r['mean_width_over_mean'])) < 1e-9 * max(1, s), (r['scaffold'], r['floppy'])
    return f"disp.: ${float(r['eig_min']):.3f}$--${float(r['eig_max']):.3f}$, $s={s:.3f}$, worst {w:.3f}"


ROUTE = {'mixed_L_P2': 'L + C', '': '--'}


def retro_table():
    t = classification_table()
    sets = {'dual': 'D', 'nondual': 'N', 'unbiased_dual': 'Ud', 'unbiased_nondual': 'Un'}
    q = [r for r in t if r['cohort'] in sets and r['qualified'] == 'True']
    g = defaultdict(list)
    for r in q:
        g[(r['scaffold'], r['floppy'])].append(r)
    rows, flat, hit, false = [], 0, 0, 0
    for k in sorted(g, key=lambda k: (g[k][0]['shape_exact_flat'] != 'True', k)):
        rs = g[k]
        assert len({(r['shape_exact_flat'], r['prediction'], r['route']) for r in rs}) == 1, k
        r = rs[0]
        isf = r['shape_exact_flat'] == 'True'
        des = r['prediction'] == 'predicted_flat'
        flat += isf; hit += isf and des; false += des and not isf
        if des:
            prim = per_primitive(json.loads(r['values']), int(r['cell_multiplicity']))
            if isf:
                assert len(prim) == len(json.loads(r['branch_means'])), k
            pv = grouped(prim)
        else:
            pv = '--'
        rows.append(f"{pair(*k)} & {', '.join(sets[x['cohort']] for x in rs)} & {ROUTE.get(r['route'], r['route'])} & "
                    f"{pt(pv, '2.4cm')} & {pt(_spec_cell(r), '5.0cm')} & {'yes' if des == isf else 'no'}")
    res = J('classifier/classifier_results.json')['cohorts']['retrospective_unique_ordered_pairs']
    assert (len(rows), flat, hit, false) == (res['qualified'], res['actual_flat'], res['true_positive'], 0) == (58, 22, 22, 0)
    head = 'Pair & Sets & Route & Designated $\\omega^2/\\kc$ & First-order outcome & Correct'
    cap = ('Retrospective classification of every eligible pair (rigid cage, all mechanisms lifted), counting each '
           'ordered pair once. Sets: D, dual pairs (Table~\\ref{tab:full_atlas}); N, non-dual controls of Section~S3.3; '
           'Ud and Un, dual and non-dual cages of the unbiased control of Section~S3.4. Route: frozen-classifier route '
           '(--, abstention). Outcome: flat values with multiplicities, or the dispersive range of $\\omega^2/\\kc$, at 32 '
           'generic wavevectors of the common primitive cell. A pair is flat when the largest sorted-branch range is at '
           'most $10^{-8}\\|S\\|_2$. For a dispersive pair, $s$ is the spread of Tables~\\ref{tab:screen} and '
           '\\ref{tab:full_flat}, the mean branch range over the mean branch value, and worst is the largest branch range '
           'over the same mean. Because this cell is primitive and the sample differs, $s$ can differ from the '
           'conventional-cell value of Table~\\ref{tab:screen} for the same pair. Both the designated and the observed multiplicities '
           'are per common primitive cell. Correct: the designation agrees with '
           f'the outcome, an abstention counting as correct for a dispersive pair. Of the {len(rows)} pairs, {flat} are flat and '
           f'{hit} of them are designated; {false} of the {len(rows) - flat} dispersive pairs are designated.')
    return longtable('full_retro', 'llllll', head, cap, 'tab:full_retro', rows)


# ------------------------------------------------------------------ 6. classifier: prospective tests 1 and 2
def prospective_table():
    frozen = J('classifier/freezing.json')
    sha = frozen['frozen_classifier']['SHA-256']
    assert sha == EXPECTED_SHA
    assert frozen['test_1']['classifier_matches_frozen'] and frozen['test_2']['classifier_matches_frozen']
    assert hashlib.sha256((T.ROOT / 'classifier' / 'flat_classifier.py').read_bytes()).hexdigest() == sha
    t1 = [r for r in classification_table() if r['cohort'] == 'prospective']
    t2 = CSV('classifier/stratified_test/classification.csv')
    pv2 = {str(r['id']): r['prediction'] for r in J('classifier/stratified_test/predictions.json')['rows']}
    for r in t2:
        p = pv2[r['id']]
        assert p['prediction'] == r['prediction'] and (p.get('route') or '') == r['route'], r['id']
        r['values'] = json.dumps(p.get('values', []))
    uc = J('classifier/stratified_test/route_U_C_details.json')
    rows, tot = [], {}

    def outcome(r):
        isf = r['shape_exact_flat'] == 'True'
        des = r['prediction'] == 'predicted_flat'
        return isf, des, ('flat, designated' if isf and des else 'flat, abstained' if isf else
                          'false designation' if des else 'dispersive, abstained')
    for tag, tab, strat in [('1', t1, None), ('2', t2, 'stratum')]:
        e = [r for r in tab if r['qualified'] == 'True']
        e.sort(key=lambda r: (outcome(r)[2] != 'flat, designated', outcome(r)[2] != 'flat, abstained', int(r['id'])))
        n = Counter()
        for r in e:
            isf, des, o = outcome(r)
            n['elig'] += 1; n['flat'] += isf; n['call'] += des and isf; n['false'] += des and not isf
            pv = grouped(json.loads(r['values'])) if des else '--'
            st = r[strat] if strat else 'uniform'
            rows.append(f"{tag} & {r['id']} & {st} & {pair(r['scaffold'], r['floppy'])} & {ROUTE.get(r['route'], r['route'])} & "
                        f"{pt(pv, '2.2cm')} & {o} & {pt(_spec_cell(r), '4.6cm')}")
        tot[tag] = n
    # test 2: designations outside the eligible set by routes U and C (the scaffold is not rigid)
    t2i = {r['id']: r for r in t2}
    nuc = 0
    for u in uc:
        r = t2i[str(u['id'])]
        assert r['qualified'] == 'False' and r['prediction'] == 'predicted_flat' and r['route'] == u['route']
        assert r['rigid'] == 'False' and json.loads(r['M_S'])[0] == u['M_S'][0] > 0
        nuc += 1
        rows.append(f"2 & {r['id']} & {r['stratum']} & {pair(r['scaffold'], r['floppy'])} & {u['route']} & "
                    f"{pt(grouped(u['predicted_values']), '2.2cm')} & outside, $M_S={u['M_S'][0]}$ & "
                    f"{pt(_spec_cell(r) + ' (fixed scaffold)', '4.6cm')}")
    assert nuc == 3 and {u['route'] for u in uc} == {'U', 'C'}
    others = [r for r in t2 if r['prediction'] == 'predicted_flat' and r['qualified'] == 'False' and r['route'] in ('U', 'C')]
    assert len(others) == 3
    # totals against the result files and the text
    r1 = J('classifier/classifier_results.json')['cohorts']['prospective']
    rb = J('classifier/stratified_test/results.json')['all']
    assert (tot['1']['elig'], tot['1']['flat'], tot['1']['call'], tot['1']['false']) == \
        (r1['qualified'], r1['actual_flat'], r1['true_positive'], r1['false_positive']) == (20, 5, 3, 0)
    assert (tot['2']['elig'], tot['2']['flat'], tot['2']['call'], tot['2']['false']) == \
        (rb['qualified'], rb['flat'], rb['hit'], rb['false_designation']) == (68, 9, 5, 0)
    hits2 = sorted(f"{r['scaffold']}|{r['floppy']}" for r in t2 if r['qualified'] == 'True' and r['prediction'] == 'predicted_flat')
    assert hits2 == sorted(['kpl|uro', 'ckq|nfa', 'nia|ats', 'svd|knc', 'wfm|hua']), hits2
    ab2 = sorted(f"{x['scaffold']}|{x['floppy']}" for x in rb['abstained_flat'])
    assert ab2 == sorted(['ttr|omy', 'tca|hyb', 'sta|smv', 'wfm|wih']), ab2
    # values quoted in S3.6 for the three designations of test 1
    v1 = {(r['scaffold'], r['floppy']): sorted({round(x, 3) for x in json.loads(r['values'])}) for r in t1
          if r['qualified'] == 'True' and r['prediction'] == 'predicted_flat'}
    assert v1 == {('mob', 'scp'): [1.092, 1.333], ('xfs', 'nfc'): [0.134], ('npo-e', 'ybh'): [5.708]}, v1
    shat = '\\texttt{' + '\\allowbreak '.join(sha[i:i + 16] for i in range(0, 64, 16)) + '}'
    head = 'Test & ID & Stratum & Pair & Route & Designated $\\omega^2/\\kc$ & Outcome & First-order spectrum'
    s1 = J('classifier/sampling.json')
    s2 = J('classifier/stratified_test/sample.json')
    st = s2['strata']
    assert len(t1) == len(s1['prospective']) and len(t2) == s2['n'] == sum(v['n'] for v in st.values())
    assert Counter(r['stratum'] for r in t2) == Counter({k: v['n'] for k, v in st.items()})
    # the two close flat values of tca (x) hyb, printed at the precision that separates them
    tm = sorted(json.loads(next(r for r in t2 if (r['scaffold'], r['floppy']) == ('tca', 'hyb'))['branch_means']))
    tv = [tm[0], tm[-1]]
    assert 1e-6 < tv[1] - tv[0] < 1e-3
    tca = [val(x, dec=distinct_dec(tv)) for x in tv]
    cap = ('Every eligible pair of the two prospective tests of the frozen classifier, whose source file has SHA-256 '
           + shat + f'. Test 1: {len(t1)} ordered pairs of distinct three-dimensional nets sharing a space-group setting, drawn '
           f"uniformly (seed {s1['seed']}). Test 2: {len(t2):,} further pairs, drawn uniformly within the "
           + ', '.join(f"{k} ({v['n']} pairs)" for k, v in list(st.items())[:-1]) + ' and '
           + ''.join(f"{k} ({v['n']} pairs)" for k, v in list(st.items())[-1:]) +
           f" space groups (seed {s2['seed']}), excluding every pair screened before. In both "
           'tests the predictions were sealed before any spectrum was computed. Eligible: rigid cage that lifts every '
           'mechanism. Outcome: flat or dispersive at 32 generic wavevectors of the common primitive cell, and whether '
           'the classifier designated it; flatness, $s$ and worst are defined as in Table~\\ref{tab:full_retro}. Flat '
           'values that differ beyond the grouping tolerance of $10^{-6}$ are printed with as many digits as separate '
           f'them, as for the two bands of \\net{{tca}}$\\otimes$\\net{{hyb}} at ${tca[0]}\\kc$ and ${tca[1]}\\kc$. The last three rows are the route-U and route-C designations of test 2, which '
           'fall outside the eligible set because the scaffold has $M_S$ generic mechanisms; their spectrum is that of the '
           'fixed scaffold and they are not scored. ID: index in the sealed sample.')
    n = longtable('full_prospective', 'lrllllll', head, cap, 'tab:full_prospective', rows, size='scriptsize')
    prot = {'Eligible': tot['1']['elig'] + tot['2']['elig'], 'Flat': tot['1']['flat'] + tot['2']['flat'],
            'FlatCalled': tot['1']['call'] + tot['2']['call']}
    assert prot == {'Eligible': 88, 'Flat': 14, 'FlatCalled': 8}, prot
    ms = J('classifier/misses.json')
    tcab = [b for q in ms['pairs'] if (q['scaffold'], q['floppy']) == ('tca', 'hyb') for b in q['bands']]
    assert len(tcab) == 1 and tcab[0]['multiplicity'] == 8
    wid = max(f['width'] for q in ms['pairs'] if (q['scaffold'], q['floppy']) in {('ttr', 'omy'), ('pdb', 'xav'), ('ith-e', 'urg')}
              for b in q['bands'] if not b['in_spectrum_of_S'] for f in b['fixed_cage'])
    assert all(f['kc_over_k'] in (0.1, 1.0, 3.0) for q in ms['pairs'] for b in q['bands'] for f in b['fixed_cage'])
    e = math.floor(math.log10(wid))
    prot_extra = {'missGroup': f"{tcab[0]['value']:.6f}", 'missWidth': f"{math.ceil(wid / 10 ** e * 10) / 10:g}\\times10^{{{e}}}"}
    (T.OUT / 'prot_values.tex').write_text(
        '% generated by make_full_tables.py from the classifier result files; do not edit\n'
        '% sums of the uniform test (300 pairs) and the stratified test (1000 pairs)\n' +
        ''.join(f'\\newcommand{{\\prot{k}}}{{{v}}}\n' for k, v in prot.items()) +
        ''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in prot_extra.items()), encoding='utf8')
    return n, tot, prot


# ------------------------------------------------------------------ 7. unbiased control (H5)
def unbiased_table():
    u = J('controls/unbiased_control.json')
    h = [{k: _s(v) for k, v in r.items()} for r in u['rows']]
    assert len(h) == len(u['rows']) == 128
    for a, b in zip(h, u['rows']):
        for k in ['floppy', 'scaffold', 'kind']:
            assert a[k] == b[k]
        for k in ['rigid', 'fully_lifted', 'exact_flat', 'near_flat']:
            assert (a[k] == 'True') == bool(b[k]), (a['floppy'], k)
    by = defaultdict(list)
    for r in h:
        by[r['floppy']].append(r)

    def cell(r):
        f = ''.join(c for c, k in [('R', 'rigid'), ('L', 'fully_lifted'), ('F', 'exact_flat')] if r[k] == 'True')
        return f"{net(r['scaffold'])} ({json.loads(r['scaffold_M'])[0]}) {f or '--'}"
    rows = []
    cnt = Counter()
    for fl in sorted(by):
        rs = sorted(by[fl], key=lambda r: r['kind'] != 'dual')
        assert [r['kind'] for r in rs] == ['dual', 'nondual', 'nondual', 'nondual']
        for r in rs:
            for k in ['rigid', 'fully_lifted', 'exact_flat']:
                cnt[(r['kind'], k)] += r[k] == 'True'
        rows.append(f"{net(fl)} & {rs[0]['nu_F']} & " + ' & '.join(cell(r) for r in rs))
    s = u['summary']
    for kind in ['dual', 'nondual']:
        assert (cnt[(kind, 'rigid')], cnt[(kind, 'fully_lifted')], cnt[(kind, 'exact_flat')]) == \
            (s[kind]['rigid'], s[kind]['fully_lifted'], s[kind]['exact_flat'])
    assert (cnt[('dual', 'rigid')], cnt[('nondual', 'rigid')]) == (int(MAC['rigidDual']), int(MAC['rigidOther'])) == (21, 18)
    assert (cnt[('dual', 'fully_lifted')], cnt[('dual', 'exact_flat')], cnt[('nondual', 'fully_lifted')],
            cnt[('nondual', 'exact_flat')]) == (12, 4, 15, 1)
    head = 'Floppy net & $\\nu_F$ & Dual cage & \\multicolumn{3}{c}{Same-group non-dual cages}'
    pre = J('controls/preregistration.json')
    assert pre['population_nets'] == u['population_nets'] and len(by) == u['sample_floppy_nets']

    def tot(kind):
        return (f"{cnt[(kind, 'rigid')]} R, {cnt[(kind, 'fully_lifted')]} L, {cnt[(kind, 'exact_flat')]} F of "
                f"{s[kind]['computed']}")
    cap = (f"Every row of the unbiased control of Section~S3.4. Each of the {len(by)} floppy nets, drawn uniformly from the "
           f"{u['population_nets']} nets with $\\nu>0$ and a same-group dual (seed {pre['seed']}), is caged by its dual and "
           'by three nets of the same space group drawn uniformly. Each cage is listed with its generic mechanism count $M_S$ '
           'in parentheses and the flags R, rigid ($M_S=0$); L, every mechanism lifted; F, exactly flat at first order; '
           f"--, none. Totals: dual cages {tot('dual')}; non-dual cages {tot('nondual')}.")
    return longtable('full_h5', 'lrllll', head, cap, 'tab:full_h5', rows)


# ------------------------------------------------------------------ 8. finite stiffness: uncertainty of the zone fractions
def zerovol_table():
    # eight splitting pairs, then fcu|flu and she-d|she (computed later with the same script and wavevectors)
    z = CSV('finite_stiffness/zone_fraction_summary.csv')
    rows = []
    for r in z:
        lo, hi = json.loads(r['volume_ci95'])
        olo, ohi = json.loads(r['outside_gamma_ci95'])
        rows.append(f"{pair(r['scaffold'], r['floppy'])} & {float(r['kc']):g} & {100 * float(r['threshold']):g} & "
                    f"{100 * lo:.1f}--{100 * hi:.1f} & {100 * float(r['gamma_contribution']):.2f} & "
                    f"{100 * olo:.1f}--{100 * ohi:.1f} & {float(r['selected_projector_weight_mean']):.3f}")
        if abs(float(r['kc']) - 1e-3) < 1e-12 and abs(float(r['threshold']) - 0.01) < 1e-12:
            want = MAC.get('zv@' + r['scaffold'] + '@' + r['floppy'])
            got = f"{100 * float(r['volume_fraction']):.1f}"
            sm = (T.OUT / 'sm_values.tex').read_text(encoding='utf8')
            assert f"\\csname zv@{r['scaffold']}@{r['floppy']}\\endcsname{{{got}}}" in sm, (r['scaffold'], got)
    assert len(rows) == 60
    assert {(int(r['nq']), len(json.loads(r['replicate_fractions']))) for r in z} == {(8192, 8)}
    head = 'Pair & $\\kc/\\ks$ & $\\delta$ (\\%) & 95\\% CI (\\%) & in $\\Gamma$ pocket (\\%) & outside $\\Gamma$, 95\\% CI (\\%) & weight'
    cap = ('Uncertainty and composition of the zone fractions of Table~\\ref{tab:zerovol}. 95\\% CI: interval over the eight '
           'scrambled Sobol blocks of 1,024 wavevectors. $\\Gamma$ pocket: part of the counted fraction that lies in the '
           'zone-centre pocket where the lifted band meets the acoustic branch, defined as $\\rho\\le2\\sqrt{\\kc/\\ks}$. '
           'The pocket itself covers about 0.1\\%, 3\\% and 17\\% of the zone at the three stiffness ratios. The column lists only the part of it '
           'that exceeds the threshold, which depends on the pair. '
           'Here $\\rho$ is the distance from the wavevector to the nearest reciprocal-lattice point, in Cartesian '
           'coordinates of the primitive reciprocal basis (without the factor $2\\pi$), divided by the cube root of the '
           'volume of the reciprocal cell, so that the radius scales as the pocket, as $\\sqrt{\\kc}$. Outside $\\Gamma$: '
           'interval of the remainder. The last twelve rows, \\net{fcu}$\\otimes$\\net{flu} and \\net{she-d}$\\otimes$'
           '\\net{she}, were computed later with the same script and wavevectors. '
           'Weight: mean overlap of the assigned eigenstates with the mechanism projector.')
    return longtable('full_zerovol', 'lrrcrcr', head, cap, 'tab:full_zerovol', rows)


# ------------------------------------------------------------------ 9. drive: gain statistics
def gain_table():
    rows = []
    nm = {'nbo': ('bcu', 'nbo'), 'flu': ('fcu', 'flu'), 'hcb': ('hxl', 'hcb'), 'dia': ('dia', 'dia')}
    key = {'nbo': 'Nbo', 'flu': 'Flu', 'hcb': 'Hcb', 'dia': 'Dia'}
    for m in ['nbo', 'flu', 'hcb', 'dia']:
        d = J(f'drive/gain/{m}.json')
        i = d['Omega_over_reference'].index(2.0)
        g = sorted(row[i] for row in d['gain_over_reference'])
        ref = d['h'] / 4
        win = sum(abs(x / ref - 1) < 0.04 for x in g)
        pos = sum(x > 1e-5 for x in g)
        assert (str(win), str(pos)) == (MAC['gainWin' + key[m]], MAC['gainPos' + key[m]])

        def qn(p):
            return g[min(len(g) - 1, int(p * (len(g) - 1) + 0.5))]
        assert len(g) == 256 and d['h'] == 0.1
        rows.append(f"{pair(*nm[m])} & {float(d['kc']):g} & ${val(float(d['reference_squared_over_kc']))}$ & {len(g)} & "
                    f"{pos} & {win} & {sum(g) / len(g) / ref:.3f} & {qn(.05):.5f}, {qn(.5):.5f}, {qn(.95):.5f}")
    for r in CSV('finite_stiffness/gain_other_pairs_summary.csv'):
        qs = json.loads(r['gain_quantiles_05_50_95'])
        n = int(r['nq'])
        assert n == 256
        rows.append(f"{pair(r['scaffold'], r['floppy'])} & {float(r['kc']):g} & ${val(float(r['reference_squared_over_kc']))}$ & "
                    f"{n} & {round(float(r['gain_q_fraction']) * n)} & -- & {float(r['mean_gain_fraction_of_Mathieu']):.3f} & "
                    + ', '.join(f'{x:.5f}' for x in qs))
    # caption: bcu (x) nbo at k_c = 1e-3 appears on both wavevector sets with the same printed mean
    dup = [r.split(' & ') for r in rows if r.startswith(pair('bcu', 'nbo') + ' & 0.001 &')]
    assert len(dup) == 2 and dup[0][6] == dup[1][6], dup
    head = 'Pair & $\\kc$ & $\\omega_{\\rm ref}^2/\\kc$ & $n_q$ & grow & window & mean$/(h/4)$ & $g_{\\max}/\\omega_{\\rm ref}$: 5, 50, 95\\%'
    cap = ('Linear gain of the parametric drive at $\\Omega=2\\omega_{\\rm ref}$, $h=0.1$, $\\ks=1$, no damping, for every '
           'pair and contact stiffness computed. $\\omega_{\\rm ref}$: flat first-order frequency of the band addressed '
           '(for \\net{dia}$\\otimes$\\net{dia}, $\\omega^2=2\\kc$). Grow: wavevectors with $g_{\\max}/\\omega_{\\rm ref}>10^{-5}$. '
           'Window: wavevectors within 4\\% of $h/4$ (computed for the first four rows). Mean: the average of '
           '$g_{\\max}/\\omega_{\\rm ref}$ divided by the single-mode Mathieu value $h/4$. The first four rows use 256 uniform wavevectors of the primitive zone, the '
           'others 256 scrambled Sobol wavevectors. \\net{bcu}$\\otimes$\\net{nbo} at $\\kc=0.001$ therefore appears twice, '
           'once on each wavevector set; the second row belongs to the stiffness series of the eight splitting pairs, and '
           'the two rows agree to the printed digits in the mean and differ in the quantiles only by sampling.')
    return longtable('full_gain', 'lrlrrrrl', head, cap, 'tab:full_gain', rows)


# ------------------------------------------------------------------ the body fragment
FRAG = r"""\section{Data tables}\label{sec:data}

This section lists, pair by pair, the data behind every count and value quoted in the main text and in the sections above. All entries are generated from the result files of the code release; RCSR coordinates are not reproduced, only net names, symmetry symbols and derived quantities.

\textbf{Atlas.} Table~\ref{tab:full_atlas} gives the structure and the funnel stage of all 76 same-group dual pairs, and the mean scaffold coordination behind the rigidity bound of Section~S3.4. Table~\ref{tab:full_flat} details the seventeen flat and near-flat pairs.

\input{si_tables/full_atlas}

\input{si_tables/full_flat}

\textbf{Exactness and counting.} Table~\ref{tab:full_exact} lists the fixed-cage widths behind Lemma~\ref{lem:exact} and the second-order coefficients of two sampled route-L pairs with non-eigenvector normals. Table~\ref{tab:counting} compares every flat pair with the counting rule.

\input{si_tables/full_exact}

\input{si_tables/full_counting}

\textbf{Classifier.} Table~\ref{tab:full_retro} lists the 58 eligible pairs of the retrospective test, and Table~\ref{tab:full_prospective} the 88 eligible pairs of the two prospective tests together with the three route-U and route-C designations made outside the eligible set.

\input{si_tables/full_retro}

\input{si_tables/full_prospective}

\textbf{Unbiased control.} Table~\ref{tab:full_h5} lists all 128 cages of Section~S3.4.

\input{si_tables/full_h5}

\textbf{Finite stiffness and drive.} Table~\ref{tab:full_zerovol} gives the sampling intervals of the zone fractions of Table~\ref{tab:zerovol}, and Table~\ref{tab:full_gain} the gain statistics of every driven pair.

\input{si_tables/full_zerovol}

\input{si_tables/full_gain}
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', help='archive root with the result files (default: the folder above tables/)')
    ap.add_argument('--out', help='output directory, the same as for make_tables.py (default: ROOT/outputs/tables)')
    E.add_argument(ap); a = ap.parse_args(argv); E.apply_argument(a); T.configure(a.root, a.out)
    MAC.update(macros('sm_values.tex', 'np_values.tex', 'gain_values.tex'))
    counts = {}
    counts['full_atlas'], stages = atlas_table()
    counts['full_flat'] = flat_table()
    counts['full_exact'] = exact_order_table()
    counts['full_counting'] = counting_table()
    counts['full_retro'] = retro_table()
    counts['full_prospective'], tot, prot = prospective_table()
    counts['full_h5'] = unbiased_table()
    counts['full_zerovol'] = zerovol_table()
    counts['full_gain'] = gain_table()
    # numbers written in the body fragment
    assert counts['full_atlas'] == 76 and counts['full_flat'] == 17 and counts['full_retro'] == 58
    assert prot['Eligible'] == 88 and counts['full_h5'] * 4 == 128 and counts['full_prospective'] == 88 + 3
    (T.OUT / 'si_data_tables.tex').write_text(FRAG, encoding='utf8')
    for p in list(T.OUT.glob('full_*.tex')) + [T.OUT / 'si_data_tables.tex']:
        s = p.read_text(encoding='utf8')
        assert '?' not in s and '\u2014' not in s, p
    print('rows:', counts)
    print('stages:', dict(stages))
    print('prospective:', {k: dict(v) for k, v in tot.items()}, 'pooled:', prot)
    print('\n'.join(REPORT))


if __name__ == '__main__':
    main()
