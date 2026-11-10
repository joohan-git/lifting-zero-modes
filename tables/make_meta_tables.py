"""Metamaterial tables and value macros of the Supplemental Material from metamaterial/*.json (sim_*.py).

Writes meta_values.tex (newcommand macros), tab_meta_map.tex, tab_meta_normals.tex, tab_meta_phys.tex,
tab_meta_time.tex and meta_values.json (headline values for the main text).

  python tables/make_meta_tables.py [--root DIR] [--out DIR]     (defaults as for make_tables.py)
"""
import argparse, json, math, os
_ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument('--root', default=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
_ap.add_argument('--out')
_a = _ap.parse_args()
META = os.path.join(_a.root, 'metamaterial')
HERE = _a.out or os.path.join(_a.root, 'outputs', 'tables')
os.makedirs(HERE, exist_ok=True)
L = lambda n: json.load(open(os.path.join(META, n), encoding='utf-8'))
DM, NM, MS, TM = L('design_map.json'), L('normals.json'), L('mass.json'), L('time.json')
TL = L('tolerance.json') if os.path.exists(os.path.join(META, 'tolerance.json')) else None


def net(k):
    a, b = k.split('|')
    return f'\\net{{{a}}}$\\otimes$\\net{{{b}}}'


def pct(x, sig=2):
    v = 100 * x
    if v == 0:
        return '0'
    nd = max(0, sig - 1 - int(math.floor(math.log10(abs(v)))))
    return f'{v:.{nd}f}'


def dm(key, des, rb, kc=1e-3, kfa=1.0, q='filtered'):
    r = [x for x in DM['pairs'][key]['rows'] if x['q'] == q and x['kf_a'] == kfa and x['kc'] == kc and x['design'] == des
         and x['kb_over_kc'] == rb]
    assert len(r) == 1, (key, des, rb, kc, kfa, q)
    return r[0]['rel_width']


def ms(pair, des, r, mat='polymer', nel=6):
    z = [x for x in MS['rows'] if x['pair'] == pair and x['design'] == des and abs(x['r_m'] - r) < 1e-9 and x['material'] == mat and x['nel'] == nel]
    assert len(z) == 1
    return z[0]


def late(name, car='on', Q=None, field='PR', span=20):
    v = TM[name]; r = [x for x in v['runs'] if x['carrier'] == car and x['Q'] == Q][0]
    Ton = 2 * math.pi / v['w_on']; t = [a / Ton for a in r['t']]
    vals = [y for a, y in zip(t, r[field]) if a > t[-1] - span]
    return sum(vals) / len(vals)


M = {}
# ---------------- design map
for tag, key in (('Nbo', 'bcu|nbo'), ('Nia', 'nia|nia-d'), ('Urk', 'pcu|urk')):
    M[f'meta{tag}Lock'] = pct(dm(key, 'lock', 1.0)); M[f'meta{tag}Free'] = pct(dm(key, 'rigid', 1.0))
    M[f'meta{tag}LockTen'] = pct(dm(key, 'lock', 1.0, kfa=10.0)); M[f'meta{tag}LockRaw'] = pct(dm(key, 'lock', 1.0, q='raw'))
    M[f'meta{tag}LockLow'] = pct(dm(key, 'lock', 0.1)); M[f'meta{tag}FreeLow'] = pct(dm(key, 'rigid', 0.1))
    M[f'meta{tag}Rel'] = pct(dm(key, 'release', None))
    ks = DM['residual'][key]['kfa_scan']
    M[f'meta{tag}LockHundred'] = pct([r for r in ks if r['kf_a'] == 100 and r['kb_over_kc'] == 1.0][0]['rel_width'])
    cs = DM['residual'][key]['cutoff_scan']
    M[f'meta{tag}CutZero'] = pct(cs[0]['rel_width']); M[f'meta{tag}CutTwo'] = pct(cs[-1]['rel_width'])
M['metaNboLockMid'] = pct(dm('bcu|nbo', 'lock', 0.3)); M['metaNboFreeMid'] = pct(dm('bcu|nbo', 'rigid', 0.3))
g = DM['pairs']['bcu|nbo']['gap_raw']
M['metaNboGapOne'] = str(sum(x < 0.01 for x in g)); M['metaNboGapFive'] = str(sum(x < 0.05 for x in g))
M['metaNq'] = str(DM['params']['nq']); M['metaGapMin'] = str(DM['params']['gap_min'])
rels = [dm(k, 'release', None) for k in DM['pairs']]
others = [dm(k, 'lock', 1.0) for k in DM['pairs'] if k not in ('bcu|nbo', 'nia|nia-d', 'pcu|urk')]
M['metaOtherLockMin'] = pct(min(others)); M['metaOtherLockMax'] = pct(max(others))
fr = [dm(k, 'rigid', 1.0) for k in DM['pairs']]
M['metaFreeMin'] = pct(min(fr)); M['metaFreeMax'] = pct(max(fr))
# ---------------- normals
M['metaIdentity'] = f"{max(v['identity_residual'] for v in NM.values()):.0e}".replace('e-', '\\times10^{-') + '}'
dots = [v['max_normal_dot'] for v in NM.values() if v['max_normal_dot'] is not None and v['max_normal_dot'] > 1e-9]
M['metaDotMin'] = f"{min(dots):.2f}"; M['metaDotMax'] = f"{max(dots):.2f}"
# ---------------- mass
main = ms('bcu|nbo', 'lock', 2e-4); mains = ms('bcu|nbo', 'lock', 2e-4, 'steel')
M['metaHzPoly'] = f"{main['f_mean_Hz']:.0f}"; M['metaHzSteel'] = f"{mains['f_mean_Hz']:.0f}"
M['metaMainWidth'] = pct(main['rel_width_w2']); M['metaMainStrut'] = f"{main['strut_first_Hz']:,.0f}".replace(',', '{,}')
M['metaMainRatio'] = f"{main['gap_ratio']:.1f}"
M['metaMainKb'] = f"{main['kb_N_m'][0]:.0f}"; M['metaMainKa'] = f"{main['ka_N_m'][0] / 1e3:.1f}"
M['metaSteelFactor'] = f"{mains['f_mean_Hz'] / main['f_mean_Hz']:.3f}"
rel = ms('bcu|nbo', 'release', 4e-4); M['metaRelHz'] = f"{rel['f_mean_Hz']:.1f}"; M['metaRelWidthPhys'] = f"{100 * rel['rel_width_w2']:.1e}".replace('e-0', r'\times10^{-') + '}'
wel = ms('bcu|nbo', 'rigid', 2e-4); M['metaWeldHz'] = f"{wel['f_mean_Hz']:.0f}"; M['metaWeldWidth'] = pct(wel['rel_width_w2'])
M['metaWeldRot'] = f"{wel['strut_first_Hz']:.0f}"; M['metaWeldRatio'] = f"{wel['gap_ratio']:.2f}"
ln = ms('bcu|nbo', 'lock', 4e-4); M['metaLockFourRatio'] = f"{ln['kb_N_m'][0] / ln['kc_N_m']:.1f}"; M['metaLockFourHz'] = f"{ln['f_mean_Hz']:.0f}"; M['metaLockFourWidth'] = pct(ln['rel_width_w2'])
nl = ms('nia|nia-d', 'lock', 2e-4); M['metaNiaHz'] = f"{nl['f_mean_Hz']:.0f}"; M['metaNiaWidthPhys'] = pct(nl['rel_width_w2']); M['metaNiaRatio'] = f"{nl['gap_ratio']:.1f}"
fl = ms('fcu|flu', 'release', 4e-4); M['metaFluRelWidth'] = pct(fl['rel_width_w2']); M['metaFluRelHz'] = f"{fl['f_mean_Hz']:.0f}"
conv = []
for p in ('bcu|nbo', 'nia|nia-d', 'fcu|flu'):
    for d in ('lock', 'rigid', 'release'):
        a, b = ms(p, d, 4e-4, nel=4), ms(p, d, 4e-4, nel=8)
        conv.append((abs(a['f_mean_Hz'] / b['f_mean_Hz'] - 1), abs(a['strut_first_Hz'] / b['strut_first_Hz'] - 1)))
M['metaConvBand'] = f"{max(c[0] for c in conv):.0e}".replace('e-0', '\\times10^{-') + '}'
M['metaConvStrut'] = f"{100 * max(c[1] for c in conv):.2f}"
# ---------------- time
M['metaPRon'] = f"{late('nbo_lock'):.1f}"; M['metaPRoff'] = f"{late('nbo_lock', 'off'):.0f}"
M['metaPRfree'] = f"{late('nbo_free'):.0f}"; M['metaPRbnn'] = f"{late('bnn_free'):.0f}"; M['metaPRnia'] = f"{late('niad_lock'):.1f}"
M['metaPRniaFree'] = f"{late('niad_free'):.0f}"; M['metaPRnboRel'] = f"{late('nbo_release'):.1f}"; M['metaPRbnnRel'] = f"{late('bnn_release'):.1f}"
M['metaLocCells'] = f"{late('nbo_lock', field='radius'):.2f}"; M['metaRoff'] = f"{late('nbo_lock', 'off', field='radius'):.1f}"
M['metaRfree'] = f"{late('nbo_free', field='radius'):.1f}"
M['metaFracOne'] = f"{100 * late('nbo_lock', field='frac_within_1cell'):.0f}"
M['metaPRonQfifty'] = f"{late('nbo_lock', Q=50.0):.1f}"
M['metaPRfreeOne'] = f"{late('nbo_free'):.1f}"; M['metaPRfreeQfifty'] = f"{late('nbo_free', Q=50.0):.1f}"
M['metaAfter'] = f"{(TM['nbo_lock']['t_end'] - TM['nbo_lock']['t0']) * TM['nbo_lock']['w_on'] / (2 * math.pi):.0f}"
_on = [x for x in TM['nbo_lock']['runs'] if x['carrier'] == 'on' and x['Q'] is None][0]
_off = [x for x in TM['nbo_lock']['runs'] if x['carrier'] == 'off' and x['Q'] is None][0]
M['metaEoffExp'] = str(int(math.floor(math.log10(_off['energy'][-1] / _on['energy'][-1]))))
M['metaWoffRel'] = f"{TM['nbo_release']['w_off'] ** 2 / 1e-3:.0f}"; M['metaPRnboRelOff'] = f"{late('nbo_release', 'off'):.1f}"
M['metaBurstBW'] = f"{100 * TM['nbo_lock']['burst_rel_bandwidth']:.0f}"
M['metaNdof'] = f"{TM['nbo_lock']['n_dof']:,}".replace(',', '{,}')
M['metaWonNbo'] = f"{TM['nbo_lock']['w_on'] ** 2 / 1e-3:.2f}"; M['metaWoffNbo'] = f"{TM['nbo_lock']['w_off'] ** 2 / 1e-3:.0f}"
# ---------------- tolerance
if TL:
    def tl(name, d, field):
        return [r for r in TL['cases'][name]['rows'] if r['delta'] == d][0][field]
    for tag, name in (('Lock', 'nbo_lock'), ('Free', 'nbo_free'), ('Rel', 'nbo_release'), ('Nia', 'niad_lock')):
        for dtag, d in (('Zero', 0.0), ('One', 0.01), ('Three', 0.03), ('Ten', 0.1)):
            M[f'metaTol{tag}{dtag}'] = pct(tl(name, d, 'rel_std'))
            M[f'metaTolPR{tag}{dtag}'] = f"{100 * tl(name, d, 'pr_over_nmc'):.0f}"
            M[f'metaTolXi{tag}{dtag}'] = f"{tl(name, d, 'xi_cells'):.2f}"
    M['metaTolSeeds'] = str(TL['params']['seeds'])
    for tag, name in (('Lock', 'nbo_lock'), ('Rel', 'nbo_release'), ('Nia', 'niad_lock')):
        M[f'metaTolSlope{tag}'] = f"{tl(name, 0.1, 'rel_std') / 0.1:.2f}"
        M[f'metaTolNodes{tag}'] = f"{tl(name, 0.1, 'pr_mean'):.1f}"
        M[f'metaTolNodesOne{tag}'] = f"{tl(name, 0.01, 'pr_mean'):.1f}"
    for tag, name in (('NiaFree', 'niad_free'), ('Bnn', 'bnn_release'), ('Flu', 'flu_release'), ('Free', 'nbo_free')):
        M[f'metaTolPR{tag}Zero'] = f"{100 * tl(name, 0.0, 'pr_over_nmc'):.0f}"
        M[f'metaTolPR{tag}Ten'] = f"{100 * tl(name, 0.1, 'pr_over_nmc'):.0f}"
        M[f'metaTol{tag}Zero'] = pct(tl(name, 0.0, 'rel_std')); M[f'metaTol{tag}Ten'] = pct(tl(name, 0.1, 'rel_std'))
    M['metaTolNmc'] = str(tl('nbo_lock', 0.0, 'n_mc'))
    M['metaTolFluGap'] = f"{tl('flu_release', 0.0, 'min_folded_gap'):.3f}"

with open(os.path.join(HERE, 'meta_values.tex'), 'w', encoding='utf-8') as fh:
    fh.write('% generated by make_meta_tables.py from the metamaterial result files; do not edit\n')
    for k, v in M.items():
        fh.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')

json.dump(dict(metaNiaLock=float(pct(dm('nia|nia-d', 'lock', 1.0), 6)), metaNiaFree=float(pct(dm('nia|nia-d', 'rigid', 1.0), 6)),
               metaNboLock=float(pct(dm('bcu|nbo', 'lock', 1.0), 6)), metaNboFree=float(pct(dm('bcu|nbo', 'rigid', 1.0), 6)),
               metaHzPoly=round(main['f_mean_Hz'], 2), metaLocCells=round(late('nbo_lock', field='radius'), 3)),
          open(os.path.join(HERE, 'meta_values.json'), 'w'), indent=1)

# ---------------- table: design map
rows = []
for k in DM['pairs']:
    c = [dm(k, 'rigid', 0.1), dm(k, 'rigid', 1.0), dm(k, 'lock', 0.1), dm(k, 'lock', 1.0), dm(k, 'lock', 1.0, kfa=10.0),
         dm(k, 'lock', 1.0, q='raw'), dm(k, 'release', None)]
    rows.append(f"{net(k)} & {DM['pairs'][k]['M']} & " + ' & '.join(pct(x) for x in c) + ' \\\\')
tab = r"""\begin{table}[h]\centering
\caption{Largest relative branch width (\%) of the lifted band in the beam frame, $\kc=10^{-3}$, cage $k_{s,a}=10$, $k_{s,b}=1$, on """ + M['metaNq'] + r""" random wavevectors with floppy-net gap $g\ge""" + M['metaGapMin'] + r"""$ unless marked. $M$, mechanisms per conventional cell. Welded and locked columns at $k_b/\kc=0.1$ and 1; $k_a$, floppy axial stiffness; raw, the same number of unfiltered draws; released, moment-free strut ends at any $k_b$.}\label{tab:meta_map}
\begin{tabular}{lcccccccc}
\hline
Pair & $M$ & \multicolumn{2}{c}{welded} & \multicolumn{2}{c}{locked} & locked, $k_a=10$ & locked, raw & released \\
 & & 0.1 & 1 & 0.1 & 1 & 1 & 1 & \\
\hline
""" + '\n'.join(rows) + r"""
\hline
\end{tabular}
\end{table}
"""
open(os.path.join(HERE, 'tab_meta_map.tex'), 'w', encoding='utf-8').write(tab)

# ---------------- table: normals
rows = []
for k, v in NM.items():
    deg = ','.join(str(d) for d in v['mech_degrees'])
    pur = '--' if v['normal_purity'] is None else f"{v['normal_purity']:.3f}"
    dot = ('--' if v['normal_purity'] is None else ('none' if v['max_normal_dot'] is None else f"{v['max_normal_dot']:.3f}"))
    lk = f"{v['lock_min']:.3f}" if v['lock_rel_width'] < 1e-10 else f"{v['lock_min']:.2f}--{v['lock_max']:.2f}"
    fr_ = f"{v['free_min']:.2f}--{v['free_max']:.2f}"
    rows.append(f"{net(k)} & {v['M']} & {v['mech_vertices']}/{v['nf']} & {deg} & {v['mech_planar']} & {pur} & {dot} & {lk} & {fr_} \\\\")
tab = r"""\begin{table}[h]\centering
\caption{Exact condition for flat locked bending. Mechanism vertices: vertices carrying nonzero mechanism weight, out of the $n$ floppy vertices per conventional cell. The next columns give their coordination and planar count. Normal purity is the fraction of the mechanism norm on the plane normals. The dot-product column gives the largest $|\mathbf n_i\cdot\mathbf n_j|$ over bonds between planar mechanism vertices; ``none'' denotes no such bond. Locked spectra are eigenvalues of $Z^\dagger(\mathcal L\otimes I_3)Z$, in units of $k_b$. Welded spectra use the condensed bending operator. Both use 30 wavevectors with $g\ge0.02$. The identity $Z^\dagger K_b^{(\theta=0)}Z=k_bZ^\dagger(\mathcal L\otimes I_3)Z$ holds to $""" + M['metaIdentity'] + r"""$ for every pair.}\label{tab:meta_normals}
\begin{tabular}{lcccccccc}
\hline
Pair & $M$ & mech.\ vertices & coordination & planar & purity & $|\mathbf n_i\cdot\mathbf n_j|$ & locked & welded \\
\hline
""" + '\n'.join(rows) + r"""
\hline
\end{tabular}
\end{table}
"""
open(os.path.join(HERE, 'tab_meta_normals.tex'), 'w', encoding='utf-8').write(tab)

# ---------------- table: physical
rows = []
for p in ('bcu|nbo', 'nia|nia-d', 'fcu|flu'):
    for d, nm in (('lock', 'locked'), ('rigid', 'welded'), ('release', 'released')):
        for r in (2e-4, 4e-4):
            z = ms(p, d, r); zs = ms(p, d, r, 'steel')
            # print lengths and bending stiffnesses in corresponding order: k_b = 12 E I / L^3 with I = pi r^4 / 4
            Lm = sorted(z['strut_lengths_m'])
            kbs = sorted(z['kb_N_m'], reverse=True)
            assert len(Lm) == len(kbs)
            for Lx, kx in zip(Lm, kbs):
                assert abs(12 * 2e9 * math.pi * r ** 4 / 4 / Lx ** 3 / kx - 1) < 0.05, (p, d, r, Lx, kx)
            Ls = '/'.join(f'{x * 1e3:.1f}' for x in Lm)
            kb = '/'.join(f'{x:.0f}' for x in kbs)
            sf = f"{z['strut_first_Hz']:.0f}" if z['strut_first_Hz'] else '--'
            ratio = f"{z['gap_ratio']:.1f}" if z['gap_ratio'] else '--'
            rows.append(f"{net(p)} & {nm} & {r * 1e3:.1f} & {Ls} & {kb} & {z['f_mean_Hz']:.1f} & {zs['f_mean_Hz']:.0f} & {pct(z['rel_width_w2'])} & {sf} & {ratio} \\\\")
tab = r"""\begin{table}[h]\centering
\caption{Physical realization with distributed strut mass in a 20~mm conventional cell. Nodes have volume 83.3~mm$^3$, which is 100~mg in polymer. Polymer has $E=2$~GPa and $\rho=1200$~kg\,m$^{-3}$; steel has $E=210$~GPa and $\rho=7850$~kg\,m$^{-3}$. The normal contact stiffness is 100~N\,m$^{-1}$ in polymer and scales with $E$ for steel. Each floppy strut has six consistent-mass elements, the cage is fixed, and spectra use 24 wavevectors with $g\ge0.05$. The strut columns give the radius $r$, the lengths $L$ and the bending stiffnesses $k_b=12EI/L^3$ in corresponding order, with $I=\pi r^4/4$. $f$ is the mean frequency of the lifted band, and width is its relative branch width in $\omega^2$. The polymer frequency $f_1$ is that of the first mode whose kinetic energy is mainly strut bending or hub rotation. Ratio is $f_1$ over the top of the band.}\label{tab:meta_phys}
\begin{tabular}{llcccccccc}
\hline
Pair & joints & $r$ (mm) & $L$ (mm) & $k_b$ (N\,m$^{-1}$) & $f$ polymer (Hz) & $f$ steel (Hz) & width (\%) & $f_1$ (Hz) & ratio \\
\hline
""" + '\n'.join(rows) + r"""
\hline
\end{tabular}
\end{table}
"""
open(os.path.join(HERE, 'tab_meta_phys.tex'), 'w', encoding='utf-8').write(tab)

# ---------------- table: time
NAME = dict(nbo_lock=('bcu|nbo', 'locked'), nbo_lock_kfa10=('bcu|nbo', 'locked, $k_a=10$'), nbo_free=('bcu|nbo', 'welded'),
            nbo_release=('bcu|nbo', 'released'), niad_lock=('nia|nia-d', 'locked'), niad_free=('nia|nia-d', 'welded'),
            bnn_free=('hex|bnn', 'welded'), bnn_release=('hex|bnn', 'released'))
rows = []
for n, (p, nm) in NAME.items():
    v = TM[n]
    rows.append(f"{net(p)} & {nm} & {v['w_on'] ** 2 / 1e-3:.2f} & {late(n):.1f} & {late(n, field='radius'):.2f} & "
                f"{100 * late(n, field='frac_within_1cell'):.0f} & {late(n, Q=50.0):.1f} &{late(n, 'off'):.0f} & {late(n, 'off', field='radius'):.2f} \\\\")
tab = r"""\begin{table}[h]\centering
\caption{Finite specimens of $6\times6\times6$ conventional cells after a point burst on one floppy hub, $\kc=10^{-3}$, $k_b=\kc$. Here $\omega_{\rm on}^2$ is the squared on-band carrier frequency and $N$ the participation number of the nodal energy. The energy-weighted RMS radius $R$ is measured from the driven hub in conventional cells. The next column gives the energy share within one cell of the hub. All observables are averaged over the final 20 periods, without damping unless marked.}\label{tab:meta_time}
\begin{tabular}{llccccccc}
\hline
Pair & joints & $\omega_{\rm on}^2/\kc$ & $N$ & $R$ & within one cell (\%) & $N$, $Q=50$ & $N$, off band & $R$, off band \\
\hline
""" + '\n'.join(rows) + r"""
\hline
\end{tabular}
\end{table}
"""
open(os.path.join(HERE, 'tab_meta_time.tex'), 'w', encoding='utf-8').write(tab)
print('\n'.join(f'{k} = {v}' for k, v in M.items()))
