# Lifting mechanical zero modes without dispersion

Code and data for J. Nam, *Lifting mechanical zero modes without dispersion*.

Archive: <https://doi.org/10.5281/zenodo.24043085>

## Requirements

Python 3.12 with the packages in `requirements.txt`:

```text
python -m pip install -r requirements.txt
```

The Zn(CN)2 calculation (`zinc_cyanide/`) also needs `ase`, `phonopy` and `mace-torch` (listed as optional in `requirements.txt`); nothing else uses them.

## External data

The nets come from the Reticular Chemistry Structure Resource (RCSR, <https://rcsr.net>; O'Keeffe et al., *Acc. Chem. Res.* **41**, 1782, 2008). They are not redistributed. Download the RCSR Systre file of 3D nets (distributed as `RCSRnets-<date>.cgd`; the version used here is `RCSRnets-2019-06-01.cgd`) and save it as `rcsr3d.cgd`, together with the 3D data file `3dall.txt`, in one directory. Pass it with `--rcsr DIR`, or set `RCSR_DIR`. `rcsr_files.json` lists the size and SHA-256 checksum of both files used here, so you can check your copy. `screen/dual_pairs.json` holds only the names of the dual pairs listed by the RCSR.

The stored results, the smoke test without `--rcsr`, the table generator `tables/make_tables.py` and the planar-pair, orbital, control and second-order models need no external data.

## Quick test

```text
python smoke_test.py
python smoke_test.py --rcsr DIR
```

The first command checks the counts and values of the article from the stored results: the screen, both prospective tests of the classifier (together 88 eligible pairs, 14 flat, 8 designated, none wrongly), the counting rule, the fixed-cage coefficients 2.80 and 4.38, the orbital crystal-field test and its second implementation, the beam-frame design maps and simulations (through their tables), Zn(CN)2, the controls and the drive. It recomputes the planar pair (ω²/k_c = 3/2) and regenerates the tables of the Supplemental Material that need no RCSR file, comparing them byte by byte with `tables/reference/`. The second command also rebuilds bcu⊗nbo, hex⊗bnn, fcu⊗flu and acs⊗gra from the RCSR files (ω²/k_c = 2, 12/7, 14/9, and 3/4 with 2), runs the classifier on them, recomputes the beam frame of bcu⊗nbo and one pair of the second prospective test, and regenerates every table. Each takes a few minutes.

## Running calculations

Results are written to `outputs/` and do not overwrite the stored files.

```text
python screen/first_order_screen.py dual --rcsr DIR --pairs bcu:nbo hex:bnn fcu:flu
python screen/first_order_screen.py summary
python classifier/pipeline.py count
python classifier/pipeline.py predict --rcsr DIR --ids 0 1 2
python classifier/pipeline.py evaluate --rcsr DIR --ids 0 1 2
python classifier/stratified.py count
python classifier/stratified.py sample --rcsr DIR
python classifier/stratified.py predict --rcsr DIR --ids 0 1 2
python classifier/stratified.py evaluate --rcsr DIR --ids 0 1 2
python classifier/stratified.py routes --rcsr DIR
python classifier/misses.py --rcsr DIR
python classifier/sector_rule.py --rcsr DIR
python counting/counting.py --rcsr DIR
python counting/rank_count.py --rcsr DIR
python counting/eeff_count.py --rcsr DIR
python fixed_cage/fixed_cage.py --rcsr DIR
python fixed_cage/urk_fixed_cage.py --rcsr DIR
python orbital/orbital.py
python independent_check/orbital_bands.py
python controls/unbiased_control.py count
python controls/dia_dia.py
python controls/unbiased_coordination.py --rcsr DIR
python planar_pair/hcb_hxl.py quantiles
python planar_pair/planar_band.py
python second_order/nbo_scaling.py --nq 100
python drive/gain.py spectrum hcb --nq 4
python drive/gain.py count
python independent_check/prospective_test.py compare
python metamaterial/sweep_branches.py --rcsr DIR --pairs bcu:nbo
python metamaterial/sweep_contact_bending.py --rcsr DIR --pairs bcu:nbo
python metamaterial/sweep_near_flat.py --rcsr DIR
python metamaterial/sim_normals.py --rcsr DIR
python metamaterial/sim_bands.py --rcsr DIR
python metamaterial/sim_design.py --rcsr DIR
python metamaterial/sim_mass.py --rcsr DIR
python metamaterial/sim_tolerance.py --rcsr DIR
python metamaterial/sim_time.py --rcsr DIR nbo_lock
python zinc_cyanide/structure.py
python zinc_cyanide/balance.py
python zinc_cyanide/zn_branches.py
python tables/make_tables.py
python tables/make_full_tables.py --rcsr DIR
python tables/make_meta_tables.py
python tables/make_extra_values.py
```

Every script prints its options with `--help`. Rerunning the screens reproduces the stored counts and roles; spreads in the fourth decimal and `mech_vertex_locality` vary with the random wavevectors. Without the size options shown here they reproduce the full calculations; the screens, scans, zone fractions, gain spectra, the 1,000 outcomes of the second prospective test and the distributed-mass, tolerance and time-domain beam-frame simulations take minutes to hours.

The Zn(CN)2 force constants were computed with MACE-MP (model `medium-mpa-0`, float64):

```text
python zinc_cyanide/contact_operator.py --name ZnCN2_rigid --structure zinc_cyanide/ZnCN2_fcc.cif --pairs Zn-C,Zn-N,C-N --rigid Zn --partner none --partner_net --sc 2,2,2 --out ZnCN2_rigid.json
python zinc_cyanide/contact_operator.py --name ZnCN2_pin --structure zinc_cyanide/ZnCN2_fcc.cif --pairs Zn-C,Zn-N,C-N --partner none --partner_net --sc 2,2,2 --out ZnCN2_pin.json
```

## Contents

| Folder | Computations |
|---|---|
| `common/` | RCSR reader and space-group expansion, Bloch compatibility matrices, placement of a net in its dual, contacts, site symmetry, primitive cells |
| `screen/` | First-order screen of the 76 same-group dual pairs and of the 250 non-dual controls; scaffold coordination, Euler relation and Maxwell counts (Section S3) |
| `high_symmetry/` | Generic spectra, dense scans of lines, planes and zone faces, split sets, the seventeen flat and near-flat pairs, and the floppy-side scan by a separate implementation |
| `classifier/` | Symmetry-only rule; frozen classifier `flat_classifier.py` and its freezing record; retrospective test, first prospective test (300 pairs), second prospective test (`stratified.py`, 1,000 pairs by crystal system, in `stratified_test/`); post hoc diagnosis of the six flat pairs the frozen classifier missed in the two tests (inter-class balance, fixed local lines, rigid units, fixed-cage exactness) and the post hoc class-and-sector rule on all 88 eligible pairs (`misses.py`, `sector_rule.py`; Sections S3.6, S3.7) |
| `counting/` | Counting rule of Calugaru et al. against the flat bands found, the rank-resolved count d_a and the effective-row count per eigenvalue of S (Section S2.7) |
| `fixed_cage/` | Fixed cage: exact flat bands of nbo, bnn and flu at k_c/k = 0.1, 1, 3, the σ tight-binding bands of flu at ±4√2/3, the second-order bandwidth coefficients of lvt and uom, and the second-order width of pcu⊗urk (Sections S2.6, S2.7, S9) |
| `orbital/` | p orbitals in a crystal field on hcb, flu and the Lieb lattice: dual-star fields, anisotropic and tilted fields, perturbative scaling (Section S9) |
| `controls/` | Unbiased control, its preregistered sample and the mean coordination of its scaffolds (Section S3.4), and dia⊗dia (Section S5.1) |
| `finite_stiffness/` | Zone fractions at finite k_c/k_s for the ten exactly flat pairs and the drive on the eight that split on a measure-zero set (Sections S3.5, S8.2) |
| `second_order/` | Second-order bandwidth of nbo in bcu (Section S4) |
| `bending/` | Joint bending (Section S7) |
| `drive/` | Floquet gain of the full composite (Section S8) |
| `planar_pair/` | hxl⊗hcb: closed form, band along Γ–K–M–Γ, single-band quantiles and second-order coefficient (Sections S5.2, S6) |
| `zinc_cyanide/` | Zn(CN)2 with a machine-learned potential: structure, contact operator of one net on the rigid-unit and pin-joint mechanisms of the other, the balance law for its rigid units and the branch widths of the single-net band (Section S5.3) |
| `metamaterial/` | Beam-frame model of the printed lattice (`frame.py`, `sim_core.py`): design maps over contact stiffness, strut bending, rotation locking and joint design for the ten exactly flat pairs and seven near-flat pairs, the exact condition for flat locked bending, band paths, distributed strut mass in physical units, finite specimens in the time domain and a fabrication-tolerance ensemble (Section S10) |
| `independent_check/` | Second, separately written implementation: prospective classifier test, zone fractions and soft modes, gain in primitive sectors, and the p-orbital crystal field (Sections S3.5, S3.6, S8.2, S9); its engine also runs the floppy-side scan |
| `tables/` | Generators of the tables and macros of the Supplemental Material from the files above (`make_tables.py`, `make_full_tables.py`, `make_meta_tables.py`, `make_extra_values.py`), and the printed versions in `tables/reference/` |

`classifier/flat_classifier.py` is the classifier frozen before the prospective tests; SHA-256 `cdb601e1c1c00d92b18a33aa3576f4981b5535a88978aeb077a8e5e44c8711d7`. `classifier/freezing.json` records the order of freezing, sampling, sealed predictions and outcomes for both tests. The working files were sealed with checksums before timestamps and local paths were removed for this archive, so those file checksums do not apply to the files here; the classifier file is byte-identical. One net of the second test, tep, is written in the RCSR file with ATOM lines that the reader in `common/nets.py` does not take; `classifier/stratified.py` rebuilds it from the same file (`classifier/stratified_test/input_correction.json`).

The second implementation in `independent_check/prospective_test.py` has two geometry rules, rule N (local normals, as route L) and rule B (bipartite net with isotropic tensors, as route C), and was run on the first prospective test; it does not implement route U.


JSON and CSV files hold the computed results. Per-pair files are named `scaffold__floppy.json`. Wavevectors are fractional coordinates of the reciprocal cell, with Bloch phase exp(2πi q·s). `SHA256SUMS.json` lists file checksums. Figure scripts are not included; the figures plot the values in these files.

## Citation

See `CITATION.cff`.

## License

MIT (`LICENSE`).
