"""Quick test: reproduce headline numbers of the article.

  python smoke_test.py                 # stored results and the planar pair (no external data)
  python smoke_test.py --rcsr DIR      # also rebuild prototypes from the RCSR files and regenerate every table

Exits with a nonzero code if a check fails.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'classifier')); sys.path.insert(0, str(ROOT / 'screen'))
sys.path.insert(0, str(ROOT / 'controls')); sys.path.insert(0, str(ROOT / 'planar_pair'))
sys.path.insert(0, str(ROOT / 'independent_check')); sys.path.insert(0, str(ROOT / 'tables'))

FAILED = []


def check(label, got, want, tol=0.0):
    if isinstance(want, float): ok = abs(got - want) <= tol
    elif tol and isinstance(want, list): ok = len(got) == len(want) and all(abs(a - b) <= tol for a, b in zip(got, want))
    else: ok = got == want
    print(f"{'ok  ' if ok else 'FAIL'} {label}: {got}" + ('' if ok else f' (expected {want})'), flush=True)
    if not ok: FAILED.append(label)


def load(rel):
    return json.loads((ROOT / rel).read_text(encoding='utf8'))


def stored():
    from first_order_screen import counts
    s = counts(load('screen/dual_screen.json')['rows'])
    check('dual pairs screened', s['pairs'], 76); check('floppy net with generic mechanisms', s['computed'], 70)
    check('rigid scaffold', s['rigid'], 49); check('every mechanism lifted', s['lifted'], 30)
    check('spread below 5%', len(s['near_flat']), 17); check('exactly flat', len(s['exact']), 10)
    n = counts(load('screen/nondual_screen.json')['rows'])
    check('non-dual controls rigid (of 250)', (n['pairs'], n['rigid']), (250, 21))
    co = load('screen/coordination.json')['summary']
    check('dual scaffolds with mean degree >= 6, rigid', (co['dual_degree_at_least_6'], co['rigid_with_dual_degree_at_least_6']), (50, 49))
    check('dual scaffolds with mean degree < 6, rigid', co['rigid_with_dual_degree_below_6'], 0)

    import pipeline
    res = pipeline.count()['cohorts']
    p = res['prospective']
    check('prospective: rigid, fully lifting cages', p['qualified'], 20); check('prospective: exactly flat', p['actual_flat'], 5)
    check('prospective: designated flat (correct)', (p['predicted_flat_qualified'], p['true_positive']), (3, 3))
    check('prospective: false designations', p['false_positive'], 0)
    u = res['retrospective_unique_ordered_pairs']
    check('retrospective: flat pairs designated', (u['actual_flat'], u['true_positive'], u['false_positive']), (22, 22, 0))
    d = res['dual']; check('retrospective: flat dual pairs designated', (d['actual_flat'], d['true_positive']), (10, 10))

    import site_symmetry_rule as ssr
    from common.records import screen_records
    conf = ssr.count(load('classifier/site_symmetry.json')['rows'], screen_records())
    six = sum(x['count'] for x in conf if x['predicted'] == 'predicted_flat' and x['actual'] == 'exact_flat')
    wrong = sum(x['count'] for x in conf if x['predicted'] == 'predicted_flat' and x['actual'] != 'exact_flat')
    check('symmetry rule alone: flat dual pairs designated, false', (six, wrong), (6, 0))

    import unbiased_control as uc
    hs, boot = uc.summarize(load('controls/unbiased_control.json')['rows'])
    check('unbiased control: dual cages rigid', (hs['dual']['rigid'], hs['dual']['computed']), (21, 32))
    check('unbiased control: other cages rigid', (hs['nondual']['rigid'], hs['nondual']['computed']), (18, 96))
    check('unbiased control: rigidity difference and 95% bootstrap interval (31 to 63 points)',
          [round(boot['rigid']['dual_minus_nondual'], 5)] + [round(x, 5) for x in boot['rigid']['cluster_bootstrap_95CI']], [0.46875, 0.3125, 0.625])
    check('unbiased control: flat among rigid, fully lifting (dual, other)',
          (hs['dual']['exact_flat'], hs['dual']['fully_lifted'], hs['nondual']['exact_flat'], hs['nondual']['fully_lifted']), (4, 12, 1, 15))

    for m, want in [('nbo', 2.0), ('flu', 14 / 9), ('hcb', 1.5)]:
        g = load(f'drive/gain/{m}.json'); i = g['Omega_over_reference'].index(2.0)
        med = float(np.median([r[i] for r in g['gain_over_reference']]))
        check(f'drive {m}: median gain / omega_flat at Omega = 2 omega_flat (h/4 = 0.025)', round(med, 4), 0.025, 2e-4)
    q = load('second_order/nbo_scaling.json')
    check('nbo: median deviation / k_c, bcu scaffold, k_c = 1e-3', round(q['bcu']['0.001']['median_dev_far_from_Gamma'] / 0.001, 2), 0.89)


def second_implementation():
    """The second implementation (independent_check/) against the primary calculation."""
    import csv
    import prospective_test as pt
    c = pt.compare(load('independent_check/prospective_outcomes.json')['rows'])
    check('second implementation, prospective: qualified, flat, designated, correct, false',
          (c['qualified'], c['actual_flat'], c['predicted_flat_qualified'], c['true_positive'], len(c['false_positive'])), (20, 5, 3, 3, 0))
    check('second implementation, prospective: same qualified set, flat set and designations as primary',
          (c['same_qualified_set_as_primary'], c['same_flat_set_as_primary'], c['same_designations_as_primary']), (True, True, True))
    zf = load('independent_check/zone_fractions.json')
    prim = {(r['scaffold'], r['floppy'], float(r['kc']), float(r['threshold'])): r
            for r in csv.DictReader(open(ROOT / 'finite_stiffness' / 'zone_fraction_summary.csv', encoding='utf-8-sig'))}
    worst = 0.0
    for key, v in zf.items():
        sc, fl = key.split('|')
        for eps in ('0.001', '0.01'):
            for d in ('0.01', '0.05'):
                a = v['eps'][eps]['A_' + d]; p = prim[(sc, fl, float(eps), float(d))]
                lo, hi = json.loads(p['volume_ci95']); se_p = (hi - lo) / 2 / 2.3646      # t(0.975, 7)
                diff = abs(a['mean'] - float(p['volume_fraction'])); se = (a['se'] ** 2 + se_p ** 2) ** .5
                worst = max(worst, diff / se if se > 0 else (0.0 if diff < 1e-3 else np.inf))
    check('second implementation, zone fractions of 3 pairs (12 cases): largest difference in standard errors < 3', worst < 3, True)
    q = zf['qtz-x|uom']['eps']['0.001']
    check('qtz-x (x) uom: wavevectors flagged by assignment but not by eigenvalue counting (1%, 5%)', (q['A_notB_0.01']['mean'], q['A_notB_0.05']['mean']), (0.0, 0.0))
    sm = load('independent_check/soft_modes.json')['qtz-x|uom']
    check('qtz-x (x) uom: median lowest scaffold eigenvalue / contact scale', round(sm['median_scaffold_lowest_over_ref'], 1), 12.3)
    check('qtz-x (x) uom: zone fraction outside 1% with the scaffold frozen', round(sm['viol_frozen_scaffold_1pct'], 2), 0.58)
    gs = load('independent_check/gain_sectors.json')['bcu|nbo']['summary']
    g = load('drive/gain/nbo.json'); i = g['Omega_over_reference'].index(2.0)
    med = float(np.median([r[i] for r in g['gain_over_reference']]))
    check('second implementation, bcu (x) nbo gain: primitive wavevectors, fraction growing', (gs['n_primitive_q'], gs['frac_gain']), (52, 1.0))
    check('second implementation, bcu (x) nbo gain: median g/omega_flat equals primary to 5 decimals', round(gs['median'], 5), round(med, 5))


def stratified():
    """Second prospective test (1,000 pairs by crystal system) and both tests together."""
    import stratified as ST
    import pipeline
    res, _ = ST.count()
    a = res['all']
    check('test 2: pairs, eligible, flat, designated (correct), false designations',
          (a['n'], a['qualified'], a['flat'], a['hit'], a['false_designation']), (1000, 68, 9, 5, 0))
    check('test 2: designations by route L, U, C (all pairs)', [r['designated_all'] for r in a['routes'][:3]], [49, 1, 2])
    check('test 2: no overlap with earlier pairs or listed duals, roles kept', tuple(res['integrity'][k] for k in ('no_prior_overlap', 'no_listed_duals', 'roles_preserved')), (True, True, True))
    stored = load('classifier/stratified_test/results.json')
    check('test 2: counts from the stored outcomes equal the stored results', json.dumps(res, sort_keys=True) == json.dumps(stored, sort_keys=True), True)
    p = pipeline.count()['cohorts']['prospective']
    check('tests 1 and 2 together: eligible, flat, designated', (p['qualified'] + a['qualified'], p['actual_flat'] + a['flat'], p['true_positive'] + a['hit']), (88, 14, 8))
    f = load('classifier/freezing.json')
    check('frozen classifier: checksum of flat_classifier.py equals the recorded one',
          hashlib.sha256((ROOT / 'classifier' / 'flat_classifier.py').read_bytes()).hexdigest(), f['frozen_classifier']['SHA-256'])
    check('input correction: corrected nets', load('classifier/stratified_test/input_correction.json')['corrected_names'], ['tep'])


def misses():
    """Post hoc diagnosis of the flat pairs the frozen classifier missed (stored classifier/misses.json, sector_rule.json)."""
    m = load('classifier/misses.json')
    check('misses: eligible pairs and flat pairs of both prospective tests', (m['eligible_pairs'], m['eligible_flat']), (88, 14))
    got = sorted((x['test'], x['scaffold'], x['floppy']) for x in m['missed_flat_pairs'])
    check('misses: the six flat pairs the frozen classifier abstained on', got,
          sorted([('uniform_300', 'pdb', 'xav'), ('uniform_300', 'ith-e', 'urg'), ('stratified_1000', 'ttr', 'omy'),
                  ('stratified_1000', 'tca', 'hyb'), ('stratified_1000', 'sta', 'smv'), ('stratified_1000', 'wfm', 'wih')]))
    P = {(r['scaffold'], r['floppy']): r for r in m['pairs']}

    def band(sc, fl, v):
        return next(b for b in P[(sc, fl)]['bands'] if abs(b['value'] - v) < 1e-3)
    want = {('pdb', 'xav'): [(0.0, 3), (1.5978, 1)], ('ith-e', 'urg'): [(0.0849, 6), (1.9058, 4)], ('ttr', 'omy'): [(1.2546, 3), (1.8667, 2)],
            ('tca', 'hyb'): [(1.9307, 8)], ('sta', 'smv'): [(0.9423, 2)], ('wfm', 'wih'): [(0.2372, 2)]}
    for (sc, fl), w in want.items():
        check(f'misses: {sc} (x) {fl} flat values and multiplicities', [(round(b['value'], 4), b['multiplicity']) for b in P[(sc, fl)]['bands']], w)
    b = band('pdb', 'xav', 1.5978)
    check('misses: pdb (x) xav 1.5978, inter-class tensors t\'_0, t\'_1', [round(s['t_inter'], 6) for s in b['sectors']], [2.0, 0.5])
    check('misses: pdb (x) xav 1.5978, balance shares from t\' (as measured)', (b['balance_inter_class']['shares_fraction'], b['balance_inter_class']['matches_shares']), (['1/5', '4/5'], True))
    check('misses: pdb (x) xav 1.5978, shares from full bond tensors (wrong)', (b['balance_full_bond_tensors']['shares_fraction'], b['balance_full_bond_tensors']['matches_shares']), (['3/7', '4/7'], False))
    check('misses: pdb (x) xav balance value', round(b['balance_inter_class']['predicted_value'], 4), 1.5978)
    b = band('ith-e', 'urg', 1.9058)
    check('misses: ith-e (x) urg 1.9058, t\'_1, t\'_2 and balance shares', ([round(s['t_inter'], 6) for s in b['sectors']], b['balance_inter_class']['shares_fraction'],
          b['balance_inter_class']['matches_shares']), ([1.5, 2.0], ['4/7', '3/7'], True))
    b = band('ttr', 'omy', 28 / 15)
    check('misses: ttr (x) omy band at 28/15, balance shares 1/5 : 4/5', (b['exact_fraction'], b['balance_inter_class']['shares_fraction'], b['balance_inter_class']['matches_value']), ('28/15', ['1/5', '4/5'], True))
    tet = next(u for u in b['rigid_units'] if u['kind'] == 'class_cluster')
    check('misses: ttr (x) omy tetrahedron translates rigidly (translation-only residual < 1e-12)', (len(tet['sites']), tet['translation_residual'] < 1e-12), (4, True))
    check('misses: ttr (x) omy tetrahedron summed S isotropic, 4 k_c', [round(x, 6) for x in tet['summed_S_eigenvalues']], [4.0, 4.0, 4.0])
    b = band('tca', 'hyb', 1.9307)
    tri = [u for u in b['rigid_units'] if u['kind'] == 'support']
    check('misses: tca (x) hyb eigenvalue case carried by two rigid triangles', (b['mechanism'], [len(u['sites']) for u in tri], all(u['rigid_residual'] < 1e-12 for u in tri)), ('eigenvalue', [3, 3], True))
    check('misses: tca (x) hyb summed dim(E_a cap ker R_F) = multiplicity', sum(c['dim_E_a_cap_kernel'] for c in b['counting']), 8)
    b = band('sta', 'smv', 49 / 52)
    check('misses: sta (x) smv at 49/52 on fixed local lines', (b['exact_fraction'], b['mechanism']), ('49/52', 'fixed_local_line'))
    b = band('wfm', 'wih', 0.2372)
    check('misses: wfm (x) wih fixed local lines, local quotients', (b['mechanism'], sorted({round(x, 4) for x in b['local_quotients']})), ('fixed_local_line', [0.1107, 0.3637]))
    exact = [band('pdb', 'xav', 1.5978), band('ith-e', 'urg', 1.9058), band('ttr', 'omy', 28 / 15)]
    worst = max(c['width'] for b in exact for c in b['fixed_cage'])
    check('misses: three collective bands outside spec(S) exact in a fixed cage (max width < 5e-14)', worst < 5e-14, True)
    first = [band('sta', 'smv', 49 / 52), band('wfm', 'wih', 0.2372)]
    check('misses: sta (x) smv and wfm (x) wih flat at first order only (width at k_c/k = 0.1 > 1e-3)', all(b['fixed_cage'][0]['width'] > 1e-3 for b in first), True)
    r = load('classifier/sector_rule.json')
    s = r['sector_rule']
    check('sector rule (post hoc): eligible, flat, calls, on flat, on dispersive', (s['eligible'], s['flat'], s['calls'], s['calls_on_flat'], s['calls_on_dispersive']), (88, 14, 14, 14, 0))
    check('sector rule (post hoc): predicted values equal the first-order spectrum', s['calls_matching_spectrum'], 14)
    check('sector rule (post hoc): dispersive abstentions (not commuting, q-dependent)', (s['dispersive_abstention_reasons']['gram_operators_do_not_commute'], s['dispersive_abstention_reasons']['joint_spectrum_varies_with_q']), (41, 33))
    l = r['local_rule']
    check('local rule (post hoc): calls, on dispersive', (l['calls'], l['calls_on_dispersive']), (10, 0))
    check('local rule (post hoc): flat pairs abstained on', sorted((x['scaffold'], x['floppy']) for x in l['flat_abstained']),
          sorted([('pdb', 'xav'), ('ith-e', 'urg'), ('ttr', 'omy'), ('tca', 'hyb')]))


def counting():
    rows = load('counting/counting.json')['rows']
    dual = [r for r in rows if r['cohort'] == 'dual']
    check('counting rule: guaranteed flat bands for the ten flat dual pairs (all zero)', [r['count_total'] for r in dual], [0] * 10)
    check('counting rule: flat dual pairs with a flat value that is not an eigenvalue of S', sum(r['has_flat_value_not_in_S'] for r in dual), 5)
    h = rows[-1]
    check('counting rule: hxl (x) hcb fully counted (class i)', (h['count_total'], h['classification']), (1, 'i'))


def fixed_cage_orbital():
    """Fixed-cage exactness, second-order coefficients and the p-orbital crystal field (stored results)."""
    import csv
    r = load('fixed_cage/results.json')
    check('fixed cage, bsn (x) lvt: second-order branch-width coefficient (gap >= 0.1)',
          round(r['lvt']['coefficient_domains']['0.1']['maximum_branch_width_coefficient'], 2), 2.80, 1e-9)
    check('fixed cage, qtz-x (x) uom: second-order branch-width coefficient (gap >= 0.1)',
          round(r['uom']['coefficient_domains']['0.1']['maximum_branch_width_coefficient'], 2), 4.38, 1e-9)
    check('fixed cage, lvt and uom: full bandwidth scales as k_c^2 (fitted exponent, 1e-4 to 1e-3)',
          [round(r[fl]['full_bandwidth_scaling_exponent'], 1) for fl in ('lvt', 'uom')], [2.0, 2.0])
    bw = list(csv.DictReader(open(ROOT / 'fixed_cage' / 'bandwidths.csv', encoding='utf-8')))
    check('fixed cage, nbo/bnn/flu at k_c = 0.1, 1, 3: cases x wavevectors', (len(bw), {int(x['sample_count']) for x in bw}), (12, {1032}))
    check('fixed cage, nbo/bnn/flu: largest exact bandwidth < 1e-13', max(float(x['bandwidth']) for x in bw) < 1e-13, True)
    tb = r['sigma_TB']
    check('flu sigma tight binding: flat-band energy 4 sqrt(2)/3', tb['expected'], 4 * np.sqrt(2) / 3, 1e-15)
    check('flu sigma tight binding: widths of the +- bands < 1e-13', max(tb['positive_bandwidth'], tb['negative_bandwidth']) < 1e-13, True)
    s = load('orbital/summary.json'); o = load('independent_check/orbital_bands.json')
    check('orbital, hcb route U: first-order shift / lambda (min, max)', s['hcb_U']['first_range'], [1.5, 1.5], 1e-12)
    check('orbital, hcb route U: full-diagonalization width at lambda = 0.03 < 1e-12', s['hcb_U']['full_width'] < 1e-12, True)
    check('orbital, flu route C: first-order shift / lambda (min, max)', s['flu_C']['first_range'], [14 / 9, 14 / 9], 1e-12)
    check('orbital, flu route C: closed-form finite-field energy, residual < 1e-12', s['validation']['flu_exact_finite_formula_residual'] < 1e-12, True)
    check('orbital, tilted p_z (route L): first-order values', s['L']['first_eigenvalues'], [0.8, 0.8])
    check('orbital, anisotropic controls disperse (hcb width > 0.5, flu widest branch > 0.1)',
          (s['hcb_U_broken']['first_width'] > .5, max(s['flu_C_broken']['first_branch_widths']) > .1), (True, True))
    check('second implementation, hcb route U: lowest full energy / lambda (min, max)', o['hcb_U_lowest_over_lambda_range'], [1.5, 1.5], 1e-12)
    check('second implementation, flu route C range equals orbital.py', o['flu_first_order_range'], s['flu_C']['first_range'], 1e-12)
    check('second implementation, flu finite-field energy minus closed form < 1e-12', o['flu_exact_lowest4_minus_Eminus_max'] < 1e-12, True)
    lo, hi = s['hcb_U_broken']['first_range']; blo, bhi = o['hcb_broken_first_order_range']
    check('hcb field diag(2.1, 0.9, 0): both implementations within [0.9, 2.1] and spread > 0.5',
          (0.9 - 1e-12 <= min(lo, blo), max(hi, bhi) <= 2.1 + 1e-12, hi - lo > .5, bhi - blo > .5), (True, True, True, True))


def metamaterial():
    s1 = load('metamaterial/sweep_contact_bending.json'); s2 = load('metamaterial/sweep_branches.json')
    check('beam frame: pairs and design points (contact-bending map, branch map)',
          (len(s1), {len(v['rows']) for v in s1.values()}, len(s2), {len(v['rows']) for v in s2.values()}), (10, {32}, 10, {56}))
    for key, want in [('bcu|nbo', 2.0), ('hex|bnn', 12 / 7), ('fcu|flu', 14 / 9)]:
        r = next(x for x in s1[key]['rows'] if x['kc'] == 1e-3 and x['kb_over_kc'] == 0)
        check(f'beam frame, {key}, pin joints, k_c = 1e-3: mean omega^2/k_c within 0.5% of the bar value', abs(r['mean'] / want - 1) < 5e-3, True)
        check(f'beam frame, {key}: first-order mean equals the bar value', r['fo_mean'], want, 1e-9)


def zinc_cyanide():
    d = load('zinc_cyanide/contact_rigid_units.json'); c, b = d['contact_real_onsite'], d['intrinsic_band']
    check('Zn(CN)2: rigid-unit mechanisms per net at generic wavevectors', d['nu_generic'], 8)
    check('Zn(CN)2: contact operator mean (THz^2), 5-95% range / mean, single-net band mean, ratio',
          (round(c['mean'], 2), round(c['rel_spread'], 1), round(b['mean']), round(b['mean'] / c['mean'])), (0.44, 2.0, 31, 70))
    bl = load('zinc_cyanide/balance.json')
    check('Zn(CN)2: kinetic shares of the two tetrahedra', [round(x, 3) for x in bl['kinetic_shares']], [0.483, 0.517])
    check('Zn(CN)2: predicted on-site doublet equals the computed one (THz^2)', max(abs(x - bl['translation_doublet']) for x in bl['recorded_doublet']) < 1e-9, True)
    try:
        sys.path.insert(0, str(ROOT / 'zinc_cyanide')); import balance
    except ImportError:
        print('(Zn(CN)2 balance recomputation skipped: ase not installed)'); return
    check('Zn(CN)2: balance recomputed from the contact tensors equals the stored one', balance.balance(d) == bl, True)


def tables(rcsr=None):
    """Regenerate the tables and macros of the Supplemental Material and compare them with tables/reference/."""
    import tempfile
    import make_tables as T
    tmp = Path(tempfile.mkdtemp())
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        T.main(['--root', str(ROOT), '--out', str(tmp)])
        if rcsr:
            import make_full_tables as FT
            FT.main(['--root', str(ROOT), '--out', str(tmp), '--rcsr', str(rcsr)])
    import subprocess
    for gen in ('make_meta_tables.py', 'make_extra_values.py'):
        subprocess.run([sys.executable, str(ROOT / 'tables' / gen), '--root', str(ROOT), '--out', str(tmp)],
                       check=True, stdout=subprocess.DEVNULL)
    made = sorted(p.name for p in tmp.glob('*.tex')) + ['meta_values.json']
    same = [n for n in made if (tmp / n).read_bytes() == (ROOT / 'tables' / 'reference' / n).read_bytes()]
    check(f'tables regenerated byte-identical to tables/reference ({len(made)} files)', same, made)
    if rcsr: check('all reference tables regenerated', len(made), len(list((ROOT / 'tables' / 'reference').glob('*.*'))))


def planar():
    import hcb_hxl as P
    fo = P.first_order(50)
    check('planar pair: contact tensor S_i / k_c', np.round(fo['S_A_over_kc'], 12).tolist(), [[1.5, 0.0], [0.0, 1.5]])
    check('planar pair: lifted band omega^2 / k_c (min)', fo['lifted_min_over_kc'], 1.5, 1e-9)
    check('planar pair: lifted band omega^2 / k_c (max)', fo['lifted_max_over_kc'], 1.5, 1e-9)
    rng = np.random.default_rng(0); v = np.median([P.overlap_band(qq, 0.003) / 0.003 for qq in rng.random((400, 2))])
    check('planar pair: full-composite median at k_c = 0.003', round(float(v), 4), round(load('planar_pair/hcb_band_quantiles.json')['0.003']['q05_q50_q95'][1], 4))


def prototypes(rcsr):
    from common import nets as E
    from common import composite as C
    from common.records import find
    import flat_classifier as fc
    import pipeline
    E.set_dir(rcsr)
    for sc, fl, want, route in [('bcu', 'nbo', [2.0], 'L'), ('hex', 'bnn', [12 / 7], 'U'), ('fcu', 'flu', [14 / 9], 'C'), ('acs', 'gra', [0.75, 2.0], 'mixed_L_P2')]:
        p = C.pair(sc, fl)                                   # placement optimized from scratch
        check(f'{sc} (x) {fl}: optimized origin equals the stored one', p['shift'].tolist(), find(sc, fl)['shift'])
        st = C.generic_stats(p, 20)
        vals = sorted({round(float(x), 6) for x in st['branch_means']})
        check(f'{sc} (x) {fl}: first-order omega^2/k_c', vals, sorted({round(float(x), 6) for x in want}))
        check(f'{sc} (x) {fl}: band width (max over branches)', round(max(st['branch_ranges']), 9), 0.0, 1e-9)
        pr = pipeline.classify_pair(p)
        check(f'{sc} (x) {fl}: frozen classifier route', (pr['prediction'], pr['route']), ('predicted_flat', route))
    import engine as V
    out = V.screen_pair(V.parse_cgd(str(Path(rcsr) / 'rcsr3d.cgd')), 'bcu', 'nbo', nq=10)[0]
    check('second implementation: bcu (x) nbo first-order omega^2/k_c', sorted(set(out['bands'])), [2.0])
    # beam frame: rebuild bcu (x) nbo and recompute both design maps
    sys.path.insert(0, str(ROOT / 'metamaterial'))
    import sweep_branches as SB
    import sweep_contact_bending as SC
    qs = np.random.default_rng(SB.SEED).random((48, 3)) - 0.5
    for mod, name in [(SC, 'sweep_contact_bending'), (SB, 'sweep_branches')]:
        got = mod.run_pair('bcu', 'nbo', qs)['rows']; want = load(f'metamaterial/{name}.json')['bcu|nbo']['rows']
        worst = max(abs(np.asarray(a[k], float) - np.asarray(b[k], float)).max() for a, b in zip(got, want) for k in a)
        check(f'beam frame, bcu (x) nbo: {name} recomputed (largest difference < 1e-9)', worst < 1e-9, True)
    # second prospective test: the tep input correction and one prediction and outcome
    import stratified as ST
    names = {x for r in ST.load('sample.json')['rows'] for x in (r['scaffold'], r['floppy'])}
    check('test 2: input correction', ST.apply_input_corrections(names), ['tep'])
    r = ST.load('predictions.json')['rows'][0]; pr = ST.predict_row(ST.load('sample.json')['rows'][0])
    check(f"test 2, id 0 ({r['scaffold']} (x) {r['floppy']}): placement and frozen-classifier call equal the sealed ones",
          (pr['shift'].tolist(), pr['prediction']['prediction'], pr['prediction']['route']), (r['shift'], r['prediction']['prediction'], r['prediction']['route']))
    o = ST.evaluate_row(r); so = ST.load('outcomes.json')['rows'][0]
    check('test 2, id 0: outcome (qualified, flat) equals the stored one', (o['qualified'], o['shape_exact_flat']), (so['qualified'], so['shape_exact_flat']))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rcsr', help='directory with rcsr3d.cgd (enables the prototype checks)')
    args = ap.parse_args(); t0 = time.time()
    stored(); second_implementation(); planar(); stratified(); misses(); counting(); fixed_cage_orbital(); metamaterial(); zinc_cyanide()
    tables(args.rcsr)
    if args.rcsr: prototypes(args.rcsr)
    else: print('(prototype checks skipped: pass --rcsr DIR)')
    print(f'{len(FAILED)} failed checks; {time.time() - t0:.0f} s')
    sys.exit(1 if FAILED else 0)


if __name__ == '__main__':
    main()
