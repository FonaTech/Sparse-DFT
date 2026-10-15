# Source provenance

[简体中文](PROVENANCE.zh-CN.md)

This page explains how the Sparse-DFT source was checked against two widely used plane-wave codes, VASP 6.4.1 and Quantum ESPRESSO 7.5, and what the check found. It states technical facts only and draws no legal conclusion. The licence boundaries are in [`LEGAL.md`](LEGAL.md).

## 1. What was compared

| Corpus | Content | Role |
|---|---|---|
| Sparse-DFT 1.0.0 | C++ core in `src/` and `tests/native/`; Python packages, tests and scripts; documentation | audited tree |
| VASP 6.4.1 | Fortran, C and Python source of the licensed distribution | reference |
| Quantum ESPRESSO 7.5 | Fortran, C and Python source | reference |
| Eigen 3.4.0, Qbox, ASE 3.29.0 | C++ and Python code with no shared history with Sparse-DFT | negative controls (noise floor) |
| Qbox `PBEFunctional.cpp`; a QE f2c file; 40 QE files transliterated to C by script | code with a known common origin | positive controls (sensitivity) |

The reference trees were processed only by local, offline statistics scripts that read them as data. The reports keep statistics, file names and line numbers. They contain no reference code or comment text. A final leak check compares every line of every report with the reference comments and code.

## 2. Methods

1. **Literal text.** Word 6-grams of comments, strings, docstrings and prose are compared with the same items in each reference. Runs that also occur in a control corpus count as generic.
2. **Numeric tables.** Runs of five or more non-trivial numeric literals are compared in order and as sets. Integers, short mantissas, small rationals and multiples of π are excluded. Each remaining match is checked against published constants.
3. **Structure.**
   - Winnowed fingerprints on a language-neutral token stream and on an arithmetic skeleton.
   - Identifier-sequence n-grams (n = 3, 4, 5) weighted by rarity.
   - Function-to-routine pairs scored by skeleton similarity, identifier 3-gram containment and longest common identifier subsequence. Each pair is compared with the 95th percentile over 40 random reference routines of similar size.
   - Exact and near-verbatim matches of code lines (token ratio ≥ 0.7).
4. **Identifiers.** Code names that mirror reference-internal names, and comments that cite reference routines or source files, are flagged. Public file-format keywords are excluded:
   - INCAR tags, POSCAR/POTCAR/OUTCAR/vasprun.xml/PROCAR/DOSCAR/EIGENVAL fields;
   - UPF `PP_*` tags.
5. **Notices and data.** Searches cover copyright and licence notices, Numerical Recipes, libxc and FFTW source, and POTCAR or UPF data blocks. Test fixtures are inspected.
6. **Third-party files.** The Eigen headers are hashed file by file against upstream 3.4.0. GUI glyphs are compared with five open icon sets: Feather, Lucide, Tabler, Bootstrap Icons and Material Design Icons, 26,556 SVG files in total.

The noise floor for each structural score is the largest value reached by a negative control. A Sparse-DFT score is flagged only when it is above that floor.

## 3. Results for this release

<!-- provenance-results:start -->
The audited files are those this package ships: 136 C, C++, CUDA and Objective-C++ files (24,555 lines), 133 Python files (30,781 lines) and the documentation. Each line below compares Sparse-DFT with VASP 6.4.1 and Quantum ESPRESSO 7.5 unless stated otherwise.

| Check | Result |
|---|---|
| Literal text (word 6-grams) | No comment, docstring or prose run reproduces reference text. The remaining runs are a public output header (the PROCAR orbital order), the licence wording of Quantum ESPRESSO and pslibrary quoted in [`LEGAL.md`](LEGAL.md), article titles in the reference lists, and generic phrases. |
| Numeric tables | Twelve lines of shipped code share runs of five or more non-trivial literals with a reference. All twelve hold published values: PW92 and PBE parameters, the PW92 spin-interpolation constant $2^{4/3}-2$, and Gauss–Legendre nodes and weights (§4). |
| Winnowed fingerprints | Largest region: 4 (C++, VASP), 6 (C++, QE), 5 (Python, VASP) and 12 (Python, QE) fingerprints. The negative controls reach 4–8 against VASP and 4–12 against QE. The 12-fingerprint region is a 3×3 determinant (cell volume). A literal translation is detected: 31 of 40 machine-transliterated QE routines reach at least 8 fingerprints. |
| Identifier sequences (n = 5) | Largest score 6.43 against VASP and 5.0 against QE; the controls reach 6.42 and 27.7. The 6.43 region is a synthetic triple loop in a test program and covers 6 % of the matched reference region. |
| Routine pairs | 81 pairs of a Sparse-DFT function and a VASP routine that computes a related quantity, each scored against the 95th percentile of 40 random VASP routines. 55 pairs stay inside the baseline on every score. 4 exceed it on skeleton similarity only, by 0.002–0.045, with no shared identifier sequence beyond the baseline. 19 exceed it on identifier overlap only, and 3 by small margins on both (see below). |
| Exact and near-verbatim code lines | 0 exact and 0 near-verbatim lines (token ratio ≥ 0.7) among 413 candidate spans. |
| Reference-internal names | No shipped identifier or comment uses a VASP or Quantum ESPRESSO routine, array or source-file name. The remaining hits are public format names (vasprun.xml, PROCAR, INCAR tags such as LASPH and NLSPLINE), the optional `vasp_legacy` unit mode, interoperability helpers such as `parse_vasprun_bands`, and generic words such as `multi`. |
| Notices and data | No third-party copyright notice, Numerical Recipes, libxc or FFTW source in the shipped code; FFTW is called through its public API. The 19 test fixtures contain no POTCAR body and no VASP copyright text. |
| Eigen | 337 of 337 headers are byte-identical to upstream Eigen 3.4.0. |
| GUI glyphs | 30 glyphs; none is a whole-icon match in the five icon sets (26,556 SVG files). |
| Leak check | 0 of 7,728 report lines contain reference text. |

**Identifier overlap.** Identifier overlap with a routine that computes the same quantity comes mostly from the standard symbols of the formulas: $j_1, j_2, j_3, m_1, m_2, m_3$ and $l_1, l_2, L, M$ from DLMF §34, and the loop indices $r, k, i, j$ of a radial mesh. The largest cases are the Wigner 3j function and the Gaunt table builder and its test (against the VASP Clebsch–Gordan and Gaunt routines), and the spherical one-centre density (against the VASP radial potential routine). Renaming only these symbols lowers their longest common identifier subsequence from 16–42 to 0–6, inside the baselines of 3–10.

**Pairs above both baselines.** Three pairs exceed the skeleton baseline by 0.01–0.05 and the identifier baseline by 2–26 identifiers:
- the real-harmonic Gaunt sum (skeleton 0.42 against 0.39; 6 shared identifiers against 4);
- the cubic spline (0.40 against 0.35). The independent Qbox spline scores higher against the same VASP routine (0.48);
- the spherical one-centre density (0.26 against 0.25), whose shared identifiers are the loop indices above.

**Formula kernels.** The kernels that evaluate shared textbook formulas are written in forms of their own:
- Wigner 3j symbols: an alternating sum of binomial coefficients from Racah's formula (DLMF 34.2.4). It agrees with the factorial form to within $7\times10^{-17}$ for every argument the code uses.
- Reciprocal projector tables: four-point Lagrange cubic interpolation. Against the VASP projector routine its functions score 0.03–0.14 in skeleton similarity, each below its own baseline (0.11–0.20).
- Cubic spline: solved for the node slopes (Hermite form).
- Radial Simpson weights on logarithmic meshes: assigned node by node from the distance to the outer end (skeleton 0.22, baseline 0.41).

<!-- provenance-results:end -->

## 4. Shared formulas and constants

Some numbers and formulas appear in Sparse-DFT and in the references because both implement the same published method. They are expected matches:

- PW92 correlation and PBE exchange-correlation parameters: Perdew & Wang, Phys. Rev. B 45, 13244 (1992); Perdew, Burke & Ernzerhof, Phys. Rev. Lett. 77, 3865 (1996).
- Gauss–Legendre nodes and weights (n = 12 and 16); these agree with `numpy.polynomial.legendre.leggauss` to within 5.2e-16.
- Wigner 3j symbols and real-harmonic Gaunt coefficients from the closed forms in the NIST Digital Library of Mathematical Functions, §34.2 and §34.3.
- CODATA constants. The optional `vasp_legacy` unit mode uses the older Bohr radius and Rydberg values (0.529177249 Å, 13.605826 eV) so that numbers can be compared with VASP output.
- Natural cubic splines, four-point Lagrange interpolation and 3×3 determinants.
- Public output layouts, such as the PROCAR orbital column order, which Sparse-DFT writes so that its files can be read by common tools.

## 5. Limits

- Statistical checks find copying of text, tables, structure and names. They cannot prove that code was written independently. A translation that changes every name and the order of every operation would score near the noise floor.
- The icon comparison covers five icon sets and cannot exclude other sources.
- The audit covered this release only. Later contributions are checked when they are reviewed (see [`CONTRIBUTING.md`](../CONTRIBUTING.md)).

## 6. Reports

If you believe a file contains material its licence does not allow, report it as described in [`SECURITY.md`](../SECURITY.md). Name the file, the lines and the suspected source.
