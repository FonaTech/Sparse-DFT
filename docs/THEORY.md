**English** · [简体中文](THEORY.zh-CN.md) · [README](../README.md) · [Sparsification](SPARSIFICATION.md) · [Accuracy](ACCURACY.md) · [Performance](PERFORMANCE.md) · [User guide](USER_GUIDE.md)

# Theory and implementation

This document starts with the Sparse Density Function method, the part that is specific to Sparse-DFT: which Fourier support the densities keep, how the omitted part is estimated before a run, and how the result is certified after it (§1). The remaining sections state the standard plane-wave projector augmented-wave (PAW) equations with the Perdew–Burke–Ernzerhof (PBE) functional on which the method acts, as given in the literature [1–3, 16, 17] and in textbook form [19], and name the Sparse-DFT source file for each. The full description of the method, with its calibration and measured savings, is in [SPARSIFICATION.md](SPARSIFICATION.md).

Atomic units are used in equations unless a unit is given. Paths refer to the source tree of this release.

## 1. The Sparse Density Function method

A plane-wave PAW calculation stores the densities on a Fourier sphere $\tfrac12|\mathbf G|^2\le E_\rho$. The smooth band density needs $E_\rho = 4E_\mathrm{cut}$. The compensation charges $\hat n$ and the partial core $\tilde n_c$ are localised in the augmentation spheres, so their spectra are long. The exact reference of Sparse-DFT therefore keeps $E_\rho = 16E_\mathrm{cut}$. The method replaces this fixed choice with the smallest support factor $f = E_\rho/E_\mathrm{cut}$ whose omitted spectrum is negligible for the given datasets.

**Error estimate.** Before the self-consistent field (SCF) cycle, a spectrum probe integrates the compensation form factors $g_{s,L}(q)$ of §4 beyond each candidate sphere, weighted by the free-atom multipoles $w_{s,L}$:

$$T(f) = \sum_s\sum_L w_{s,L}^2\int_{q(fE_\mathrm{cut})}^{q(24E_\mathrm{cut})}|g_{s,L}(q)|^2\,q^2\,dq,\qquad q(E) = \sqrt{2E}.$$

The sum runs over distinct datasets, so $T$ does not grow with the cell.

**Choice.** Auto mode takes the smallest $f$ from $\{4,5,6,8,10,12,16\}$ with $T(f)\le\tau$, where $\tau = 2.5\times10^{-7}$ at the balanced level, with $f\ge6$ for metals. The number of density modes then falls as $(f/16)^{3/2}$, to 1/8 at $f = 4$. Rules fitted on measured errors also set the exchange-correlation (XC) quadrature padding and the eigensolver search controls.

**Certificate.** Components that change only the search of the eigensolver are accepted only if every eigenpair passes the FP64 residual test of §9 on the unchanged operators, $\|\tilde Hx-\varepsilon\tilde Sx\|\le10^{-12}$ Ha, and if the SCF meets its unchanged density thresholds (§8). Every result records the choice, the tails and the operator counts in a sparsity ledger.

**Evidence.** On six fit and four hold-out systems, auto passes the same gates as $f = 16$ with 2.1–3.9 times fewer Hamiltonian operator calls (measured; [SPARSIFICATION.md §5](SPARSIFICATION.md#5-what-auto-saves)).

## 2. Units and constants

Internally every term is evaluated in Hartree atomic units. Two constant sets convert to and from Å and eV (`src/atomic_units.hpp`):

| convention | Å per bohr | eV per Hartree | use |
|---|---|---|---|
| `codata` (default) | 1/1.8897261254578281 | 27.211386245988 | CODATA 2018 values [18] |
| `vasp_legacy` | 0.529177249 | 2 × 13.605826 | the values used by VASP 6.4.1; for matched comparisons |

One convention applies to the radial grids, the atomic reference, the frozen core, the plane waves, the FFT grids and the Ewald sum together; mixing them is refused. The two sets differ in eV per Hartree by a relative $-9.77\times10^{-6}$. The PAW datasets give the atomic reference energies in eV, while every other term scales with the Hartree constant. The total energy therefore shifts by an amount that grows with the size of those terms; [ACCURACY.md §3](ACCURACY.md#3-codata-versus-vasp_legacy) shows the measured shift. Select the set with `--atomic-units`.

## 3. Plane-wave basis

Wavefunctions at wavevector $\mathbf k$ are expanded in plane waves

$$\tilde\psi_{n\mathbf k}(\mathbf r) = \frac{1}{\sqrt\Omega}\sum_{\mathbf G} c_{n\mathbf k}(\mathbf G)\,e^{i(\mathbf k+\mathbf G)\cdot\mathbf r},\qquad \tfrac12|\mathbf k+\mathbf G|^2 \le E_\mathrm{cut},$$

with $\Omega$ the cell volume (`enumerate_g_vectors`, `src/pw_core.cpp`). Without `--cutoff`, $E_\mathrm{cut} = \max(300\ \mathrm{eV},\ m\cdot\max_s E_{\max,s})$ rounded up, with $m$ = 1.15, 1.30, 1.40, 1.40 for the accuracy levels `low`, `standard`, `high`, `production` and $E_{\max,s}$ the recommended cutoff of each dataset (`automatic_cutoff`, `sparse_dft/settings.py`).

Densities use a separate sphere $\tfrac12|\mathbf G|^2\le E_\rho$. The smooth density $\tilde n$ needs $E_\rho\ge4E_\mathrm{cut}$; the compensation and partial-core charges need more. The default is $E_\rho = 16E_\mathrm{cut}$ (`run_scf`, `src/scf_core.cpp`), set with `--density-cutoff` or as a factor with `--density-cutoff-factor`. Values below $4E_\mathrm{cut}$ are refused. The default basis is chosen on the current lattice (`basis_policy: current_lattice_cutoff`).

## 4. Projector augmented waves

The PAW method of Blöchl [1], in the form of Kresse and Joubert [2], maps smooth pseudo wavefunctions $\tilde\psi$ to all-electron wavefunctions $\psi$ with the linear transformation

$$|\psi\rangle = |\tilde\psi\rangle + \sum_{a,i}\big(|\phi^a_i\rangle - |\tilde\phi^a_i\rangle\big)\langle\tilde p^a_i|\tilde\psi\rangle .$$

Here $a$ runs over atoms and $i = (\ell, m, \text{radial index})$ over partial waves. $\phi^a_i$ and $\tilde\phi^a_i$ are the all-electron and pseudo partial waves and $\tilde p^a_i$ the projectors, all taken from the PAW dataset. Inside each augmentation sphere the expansion is complete for the bound states the dataset was built for.

**Generalised eigenproblem.** The Kohn–Sham equations [16, 17] become

$$\tilde H|\tilde\psi_{n\mathbf k}\rangle = \varepsilon_{n\mathbf k}\,\tilde S|\tilde\psi_{n\mathbf k}\rangle,\qquad \tilde S = 1 + \sum_{a,ij}|\tilde p^a_i\rangle\,Q^a_{ij}\,\langle\tilde p^a_j|,$$

$$\tilde H = -\tfrac12\nabla^2 + \tilde v_\mathrm{eff} + \sum_{a,ij}|\tilde p^a_i\rangle\,D^a_{ij}\,\langle\tilde p^a_j|,$$

with the augmentation charges $Q^a_{ij} = \langle\phi^a_i|\phi^a_j\rangle - \langle\tilde\phi^a_i|\tilde\phi^a_j\rangle$ over the sphere. The non-local terms are applied as low-rank products $B^\dagger D B$ and $B^\dagger Q B$, with $B$ the projector matrix in the plane-wave basis; no $N_\mathrm{PW}\times N_\mathrm{PW}$ matrix is formed (`add_paw_low_rank_operator`, `src/pw_core.cpp`; `src/plane_wave_operator.cpp`).

**Occupation matrix.** The one-centre density matrix of atom $a$ is

$$\rho^a_{ij} = \sum_{n\mathbf k} w_{\mathbf k}\,f_{n\mathbf k}\,\langle\tilde\psi_{n\mathbf k}|\tilde p^a_i\rangle\langle\tilde p^a_j|\tilde\psi_{n\mathbf k}\rangle .$$

**Density.** The all-electron density splits into a smooth part on the Fourier grid and one-centre corrections on radial grids:

$$n = \tilde n + \hat n + \tilde n_c + \sum_a\big(n^1_a - \tilde n^1_a - \hat n_a\big),\qquad \tilde n(\mathbf r)=\sum_{n\mathbf k}w_{\mathbf k}f_{n\mathbf k}|\tilde\psi_{n\mathbf k}(\mathbf r)|^2 .$$

$\tilde n_c$ is the partial (pseudo) core charge. The compensation charge restores the multipole moments of the difference between the all-electron and pseudo densities inside each sphere:

$$\hat n_a(\mathbf r) = \sum_{LM}\rho^a_{LM}\,\bar g_{a,L}(|\mathbf r-\mathbf R_a|)\,Y_{LM}(\widehat{\mathbf r-\mathbf R_a}),\qquad \rho^a_{LM} = \sum_{ij}\rho^a_{ij}\,\hat Q^{a,LM}_{ij},$$

where $\bar g_{a,L}$ is the dataset's compensation shape, normalised to unit $L$-th moment, and $\hat Q^{a,LM}_{ij}$ are the multipole moments of $\phi^a_i\phi^a_j-\tilde\phi^a_i\tilde\phi^a_j$. In reciprocal space $\hat n_a(\mathbf G) = e^{-i\mathbf G\cdot\mathbf R_a}\sum_{LM}(-i)^L\rho^a_{LM}\,g_{a,L}(|\mathbf G|)\,Y_{LM}(\hat{\mathbf G})/\Omega$ (up to the normalisation of $Y_{LM}$), with the radial form $g_{a,L}(q) = \int\bar g_{a,L}(r)\,r^2j_L(qr)\,dr$ (`src/paw_compensation.cpp`, `src/paw_augmentation.cpp`). This radial form is the quantity the sparsity probe integrates.

**Total energy.** With $n_v = \tilde n + \hat n$ the valence pseudo density,

$$E = \sum_{n\mathbf k}w_{\mathbf k}f_{n\mathbf k}\langle\tilde\psi_{n\mathbf k}|-\tfrac12\nabla^2|\tilde\psi_{n\mathbf k}\rangle + \int\tilde v_\mathrm{loc}\,n_v + E_\mathrm H[n_v] + E_\mathrm{xc}[\tilde n+\hat n+\tilde n_c] + \sum_a\big(E^1_a - \tilde E^1_a\big) + E_\mathrm{Ewald} + \sum_a\big(E^\mathrm{ref}_a + \Delta E^\mathrm{cut}_a\big) - TS .$$

The Hartree energy is evaluated in reciprocal space, $E_\mathrm H = 2\pi\Omega\sum_{\mathbf G\neq0}|n_v(\mathbf G)|^2/|\mathbf G|^2$ (`src/density_functionals.cpp`). The one-centre energies $E^1_a$ and $\tilde E^1_a$ are computed on the radial grids from $\rho^a_{ij}$ with the same functional (`src/paw_onecenter.cpp`, `src/paw_augmentation.cpp`). $E^\mathrm{ref}_a$ is the reference value stored in the dataset for the free atom, so $E$ is measured relative to the reference atoms. $\Delta E^\mathrm{cut}_a$ is the dataset's tabulated kinetic-energy error at $E_\mathrm{cut}$, and $-TS$ the Fermi–Dirac entropy term of §7. The code assembles the same energy in the equivalent band-energy (double-counting) form (`run_scf`, `src/scf_core.cpp`) and writes each term of that form to `scf.json`.

**Non-local strengths.** The variation of $E$ with $\rho^a_{ij}$ gives

$$D^a_{ij} = \sum_{LM}\int\tilde v_\mathrm{eff}(\mathbf r)\,\hat Q^{a,LM}_{ij}\,\bar g_{a,L}Y_{LM}\,d\mathbf r + D^{1,a}_{ij} - \tilde D^{1,a}_{ij},$$

where the first term projects the smooth potential onto the compensation shapes and $D^1$, $\tilde D^1$ are the one-centre derivatives (`src/paw_augmentation.cpp`). The one-centre terms are always included; `SDFT_ENABLE_PAW_CLOSURE_DIAGNOSTIC=0` removes them for diagnosis only and is refused for spin-polarised runs.

## 5. Exchange and correlation

The functional is PBE [3] in its spin-unpolarised (`src/pbe_xc.cpp`) and collinear spin (`src/spin_pbe_xc.cpp`) forms. On the Fourier grid it is evaluated on $\tilde n+\hat n+\tilde n_c$ with gradients from reciprocal space. The real-space quadrature grid has $N_a = p\max(3, 2m_a+1)$ points along axis $a$, with $m_a$ the largest Miller index of the density support and padding $p = 2$ by default (sparsity component S2). On the radial grids the one-centre XC energy is integrated with an angular quadrature (`angular_pbe_xc`, `src/paw_augmentation.cpp`).

## 6. Ion–ion energy

The Ewald sum [4] splits the Coulomb energy of the ionic point charges $Z_a$ into real- and reciprocal-space parts,

$$E_\mathrm{Ewald} = \frac12\sum_{a,b,\mathbf T}{}^{'}\frac{Z_aZ_b\,\mathrm{erfc}(\eta|\mathbf R_a-\mathbf R_b+\mathbf T|)}{|\mathbf R_a-\mathbf R_b+\mathbf T|} + \frac{2\pi}{\Omega}\sum_{\mathbf G\neq0}\frac{e^{-G^2/4\eta^2}}{G^2}\Big|\sum_aZ_ae^{i\mathbf G\cdot\mathbf R_a}\Big|^2 - \frac{\eta}{\sqrt\pi}\sum_aZ_a^2 - \frac{\pi}{2\Omega\eta^2}\Big(\sum_aZ_a\Big)^2,$$

with $\eta = \sqrt\pi/\Omega^{1/3}$ by default. Both sums are cut where their tails fall below the requested tolerance, using reciprocal-lattice norms so that skewed cells keep every needed translation (`evaluate_ewald`, `src/ewald.cpp`; forces and stress in `src/ionic_ewald.cpp`).

## 7. Occupations

All k points share one chemical potential $\mu$ (`fill_occupations`, `src/occupations.cpp`). With `--smearing-ev` $\sigma = k_BT > 0$ the occupations follow Fermi–Dirac statistics, the finite-temperature functional of Mermin [5]:

$$f_{n\mathbf k} = \frac{2}{1+e^{(\varepsilon_{n\mathbf k}-\mu)/\sigma}},\qquad \sum_{n\mathbf k}w_{\mathbf k}f_{n\mathbf k} = N_e,$$

$$-TS = \sigma\sum_{n\mathbf k}2w_{\mathbf k}\big[\tilde f\ln\tilde f + (1-\tilde f)\ln(1-\tilde f)\big],\qquad \tilde f = f_{n\mathbf k}/2 .$$

$\mu$ is found by bisection on the electron count. In a gap that is wide compared with $\sigma$, both Fermi tails fall below the double-precision resolution of $N_e$, so the count is flat there and cannot locate $\mu$. For any filling in which the lowest levels hold exactly $N_e$ and the next level lies higher, the solver bisects instead on the balance of the two tails, each summed in logarithmic form with $\ln\tilde f(x) = -\ln(1+e^{x})$:

$$\ln\sum_{\varepsilon_{n\mathbf k}>\varepsilon_v}2w_{\mathbf k}\tilde f_{n\mathbf k} = \ln\sum_{\varepsilon_{n\mathbf k}\le\varepsilon_v}2w_{\mathbf k}\big(1-\tilde f_{n\mathbf k}\big),$$

where $\varepsilon_v$ is the highest filled level. For $\sigma$ much smaller than the gap the root tends to $\mu = \tfrac12(\varepsilon_v+\varepsilon_c) + \tfrac{\sigma}{2}\ln(g_v/g_c)$, with $g_v$ and $g_c$ the $k$-weighted capacities at the two band edges. The Fermi level of a smeared insulator thus depends on $\sigma$ and on the edge degeneracies; band comparisons therefore refer gapped spectra to the VBM (§12). The reported total energy is the free energy $E - TS$ and `internal_energy_ev` is $E$. At $\sigma = 0$ the levels are filled from the bottom at zero temperature, keeping degenerate levels equally occupied. Collinear spin uses the same solver with half weights per channel. If the top computed band still has an occupation above $10^{-6}$ at some k point, an automatic band count grows by $\max(4, 25\,\%)$ and the SCF restarts; an explicit `--bands` fails instead.

## 8. Self-consistency

The SCF iterates on the joint state $X = (\tilde n(\mathbf G), \rho^a_{ij})$. Each step builds $\tilde H$ and $\tilde S$ from $X_\mathrm{in}$, solves the eigenproblem, and forms $X_\mathrm{out}$. A multisecant (Anderson/Broyden type) update [6, 7] combines the history of $(X_\mathrm{in}, X_\mathrm{out}-X_\mathrm{in})$ pairs into the next input (`src/joint_multisecant.hpp`, `src/scf_core.cpp`). The smooth density is preconditioned with the Kerker form [8]

$$\chi(G) = \max\Big(a_\mathrm{min},\ a\,\frac{G^2}{G^2+q_0^2}\Big),$$

with $a$ = `--mixing` (default 0.2), $q_0$ = 1 Å$^{-1}$ and the floor $a_\mathrm{min} = \min(0.1, a, a_\mathrm{mag})$ (`kerker_response`, `src/collinear_mixing.hpp`). For collinear spin the magnetisation uses its own mixing $a_\mathrm{mag} = \min(2, 4a)$ in the same history. Proposed one-centre matrices that are not positive semidefinite are retracted along the whole step, never per atom or per channel.

The SCF stops when both residuals fall below their thresholds:

- the density residual, the largest change of any density coefficient between input and output, below `--density-tolerance` (default $10^{-6}$; $10^{-8}$ in relaxations);
- the one-centre residual, the per-atom normalised Frobenius norm of the change of $\rho^a_{ij}$, below `--onecenter-density-tolerance` (default $10^{-6}$).

`--scf-acceptance-margin` $\in(0,1]$ scales both thresholds.

## 9. Eigensolver and certificate

Each k point is solved by a block Davidson method [9] on the matrix-free operators $\tilde H$ and $\tilde S$ (`solve_generalized_hermitian_davidson`, `src/pw_core.cpp`). The local potential is applied by FFT; kinetic energy is diagonal; the non-local parts are the low-rank products of §4. Correction vectors are preconditioned with the Teter–Payne–Allan form [10]

$$K(x) = \frac{2}{s}\,\frac{27+18x+12x^2+8x^3}{27+18x+12x^2+8x^3+16x^4},\qquad x = \tfrac12|\mathbf k+\mathbf G|^2/s,$$

with $s = 1.5\max\big(\sum_{\mathbf G}|x_{\mathbf G}|^2\,\tfrac12|\mathbf k+\mathbf G|^2,\ 2\ \mathrm{eV}\big)$ from the current vector $x$. The search subspace holds up to 8 times the band count by default (`--davidson-subspace-factor`, 4–16).

A Ritz pair $(\varepsilon, x)$ is accepted only when it passes the certificate on the original operators in FP64 (IEEE double precision):

$$\|\,\tilde Hx - \varepsilon\tilde Sx\,\|_2 \le \epsilon_\mathrm{eig},\qquad x^\dagger\tilde Sx = 1,$$

evaluated as one block after re-normalisation (`davidson_fp64_certification`). In the final SCF iterations $\epsilon_\mathrm{eig}\le10^{-12}$ Ha. Sparsity components and solver controls may change how the Ritz pairs are found; they never change this test. Converged runs report the largest residual as `eigensolver_max_residual_hartree`.

## 10. Forces and stress

**Forces.** At self-consistency the force on atom $a$ is the derivative of $E$ at fixed occupations and fixed plane-wave set. It collects the local-potential, non-local projector, compensation-charge, partial-core and Ewald contributions; the projector term is

$$\mathbf F^{\mathrm{nl}}_a = -\sum_{n\mathbf k}w_{\mathbf k}f_{n\mathbf k}\sum_{ij}\big(D^a_{ij}-\varepsilon_{n\mathbf k}Q^a_{ij}\big)\,2\,\mathrm{Re}\,\langle\tilde\psi_{n\mathbf k}|\partial_{\mathbf R_a}\tilde p^a_i\rangle\langle\tilde p^a_j|\tilde\psi_{n\mathbf k}\rangle .$$

Because plane waves do not move with the atoms, there is no basis (Pulay) force. Projector derivatives use the derivative of the interpolation actually used for the projectors (`src/projector_derivative.cpp`; assembly in `src/scf_core.cpp`).

**Stress.** The stress is the derivative with respect to a homogeneous strain $\epsilon_{\alpha\beta}$, $\sigma_{\alpha\beta} = \Omega^{-1}\,\partial E/\partial\epsilon_{\alpha\beta}$, following Nielsen and Martin [11]. The derivative is taken with the integer plane-wave indices held fixed, so it is the exact derivative of the finite-basis energy (`electronic_paw_stress`, `src/scf_core.cpp`). A cell change at fixed cutoff can add or remove plane waves, and that discontinuity is not in the analytic stress. The difference, the basis-set (Pulay) stress, shrinks as $E_\mathrm{cut}$ grows; converge the cutoff before using stress quantitatively.

## 11. Relaxation

`sparse-dft relax` minimises the energy over atomic positions (fixed cell) or, with `--cell`, over the cell at fixed fractional coordinates (`sparse_dft/analytic_relax.py`). Each step uses the limited-memory BFGS (L-BFGS) two-loop recursion [12] in the native program `sdft_relax_step` (`lbfgs_direction`, `src/relax_optimizer.cpp`):

- curvature pairs $(\mathbf s,\mathbf y)$ with $\mathbf s^\top\mathbf y \le 10^{-10}\|\mathbf s\|\|\mathbf y\|$ are rejected;
- the initial inverse Hessian is $\gamma I$ with $\gamma = \mathbf s^\top\mathbf y/\mathbf y^\top\mathbf y$ of the latest pair, clipped to $[10^{-6},10^{6}]$;
- a non-descent direction falls back to steepest descent, and the step is capped at `--max-displacement` (default 0.15 Å) or `--max-strain` (default 0.02).

A step is accepted after a full SCF at the trial point satisfies the Armijo condition $E(\mathbf x+\alpha\mathbf p)\le E(\mathbf x)+10^{-4}\alpha\,\nabla E^\top\mathbf p$, trying $\alpha = 1, \tfrac12, \tfrac14,\dots$ (up to 16 trials). The history keeps 8 pairs. Convergence requires the largest force norm below `--force-tolerance` (default 0.02 eV/Å) or the largest stress component below `--stress-tolerance` (default 0.05 GPa). Every accepted state is checkpointed.

## 12. Band structures and DOS

**Non-self-consistent (NSCF) runs.** `bands` and `dos` first converge (or reuse with `--scf-checkpoint`) an SCF, then freeze its density, $\rho^a_{ij}$ and potentials and solve the eigenproblem at new k points (`run_nscf`, `src/scf_core.cpp`). The same certificate applies. The default band count is the number of occupied bands plus 4.

**Band paths.** `make-band-path --auto` returns the standard path of Hinuma et al. [13] from spglib [14] and SeeK-path; the sample density is set by `--spacing` in Å$^{-1}$ (including $2\pi$). Each segment includes both endpoints.

**DOS.** `sparse-dft dos` solves a full uniform mesh without symmetry reduction and broadens every level with a normalised Gaussian,

$$D_\sigma(E) = \sum_{n\mathbf k}w_{\mathbf k}\,g\,\frac{1}{\sigma\sqrt{2\pi}}\,e^{-(E-\varepsilon_{n\mathbf k})^2/2\sigma^2},$$

with $g$ the spin degeneracy and $\sigma$ = `--dos-sigma` (`gaussian_broaden`, `sparse_dft/bands.py`). The energy step must not exceed $\sigma/3$. The DOS is complete only below the energy at which the highest solved band starts plus $3\sigma$; `dos.json` records this as `complete_below_ev`.

**Comparing band structures.** `bands-compare` and the `bands_compare` figure default to `--align auto`. When both sources are gapped at their $E_F$ in every spin channel, with the same occupied band count per channel, each is referred to its own sampled VBM; otherwise each is referred to its own SCF $E_F$. Both spin channels of a source share one zero. A fixed-density VASP run (`ICHARG` $\ge$ 10) refills $E_F$ on its own k list, so `auto` refuses a metallic pair with such a source. `--align none` compares raw energies; this is valid when both codes use the same PAW datasets, and the accuracy report uses it.

## 13. Orbital projections

The lm-resolved weight of state $|\tilde\psi_{n\mathbf k}\rangle$ on atom $a$ is its all-electron charge inside the augmentation sphere, resolved by $(\ell, m)$:

$$w^a_{n\mathbf k,\ell m} = \sum_{ij\in\ell}\overline{c^a_{i,\ell m}}\;q^a_{ij}\;c^a_{j,\ell m},\qquad c^a_{i,\ell m} = \langle\tilde p^a_{i}|\tilde\psi_{n\mathbf k}\rangle,\qquad q^a_{ij} = \int_0^{r_a}\phi^a_i(r)\,\phi^a_j(r)\,dr .$$

The sum runs over radial channels $i, j$ of the same $\ell$, and $\phi^a_i$ are the all-electron radial partial waves (with $r^2$ included). The integral uses Simpson weights on the radial mesh truncated at the sphere radius $r_a$ of the dataset (`paw_sphere_ae_overlap`, `src/paw_hubbard.cpp`; `run_nscf`). This is Sparse-DFT's sphere-charge definition, recorded as `projection_weight_definition` in every output. [ACCURACY.md](ACCURACY.md#5-band-paths-dos-and-projections) compares it with the sphere-projected weights that VASP 6.4.1 writes to PROCAR with `LORBIT = 11`. Real spherical harmonics run $m = -\ell\dots\ell$ with $\sin(|m|\varphi)$ for $m<0$ and $\cos(m\varphi)$ for $m>0$, giving the order s; p$_y$, p$_z$, p$_x$; d$_{xy}$, d$_{yz}$, d$_{z^2}$, d$_{xz}$, d$_{x^2-y^2}$.

Weights are not normalised: the sphere does not contain the whole state, and high-lying states can have weights above one. Totals are therefore compared only inside an energy window.

## 14. Collinear spin and DFT+U

`--spin-channels 2` solves both spin channels with one chemical potential and the spin-polarised PBE functional. `--initial-moments` sets the starting moment per atom (in $\mu_B$). `--hubbard EL:L:UEFF` adds the rotationally invariant correction of Dudarev et al. [15],

$$E_U = \frac{U_\mathrm{eff}}{2}\sum_\sigma\Big[\mathrm{Tr}\,n^{\sigma} - \mathrm{Tr}\big(n^{\sigma}n^{\sigma}\big)\Big],$$

where $n^\sigma$ is the on-site occupation matrix of shell $\ell$, built with the same all-electron overlaps $q^a_{ij}$ (`dudarev_paw`, `src/paw_hubbard.cpp`).

## References

1. P. E. Blöchl, Projector augmented-wave method, *Phys. Rev. B* **50**, 17953–17979 (1994). doi:[10.1103/PhysRevB.50.17953](https://doi.org/10.1103/PhysRevB.50.17953)
2. G. Kresse, D. Joubert, From ultrasoft pseudopotentials to the projector augmented-wave method, *Phys. Rev. B* **59**, 1758–1775 (1999). doi:[10.1103/PhysRevB.59.1758](https://doi.org/10.1103/PhysRevB.59.1758)
3. J. P. Perdew, K. Burke, M. Ernzerhof, Generalized gradient approximation made simple, *Phys. Rev. Lett.* **77**, 3865–3868 (1996). doi:[10.1103/PhysRevLett.77.3865](https://doi.org/10.1103/PhysRevLett.77.3865)
4. P. P. Ewald, Die Berechnung optischer und elektrostatischer Gitterpotentiale, *Ann. Phys.* **369**, 253–287 (1921). doi:[10.1002/andp.19213690304](https://doi.org/10.1002/andp.19213690304)
5. N. D. Mermin, Thermal properties of the inhomogeneous electron gas, *Phys. Rev.* **137**, A1441–A1443 (1965). doi:[10.1103/PhysRev.137.A1441](https://doi.org/10.1103/PhysRev.137.A1441)
6. P. Pulay, Convergence acceleration of iterative sequences. The case of SCF iteration, *Chem. Phys. Lett.* **73**, 393–398 (1980). doi:[10.1016/0009-2614(80)80396-4](https://doi.org/10.1016/0009-2614(80)80396-4)
7. L. D. Marks, D. R. Luke, Robust mixing for ab initio quantum mechanical calculations, *Phys. Rev. B* **78**, 075114 (2008). doi:[10.1103/PhysRevB.78.075114](https://doi.org/10.1103/PhysRevB.78.075114)
8. G. P. Kerker, Efficient iteration scheme for self-consistent pseudopotential calculations, *Phys. Rev. B* **23**, 3082–3084 (1981). doi:[10.1103/PhysRevB.23.3082](https://doi.org/10.1103/PhysRevB.23.3082)
9. E. R. Davidson, The iterative calculation of a few of the lowest eigenvalues and corresponding eigenvectors of large real-symmetric matrices, *J. Comput. Phys.* **17**, 87–94 (1975). doi:[10.1016/0021-9991(75)90065-0](https://doi.org/10.1016/0021-9991(75)90065-0)
10. M. P. Teter, M. C. Payne, D. C. Allan, Solution of Schrödinger's equation for large systems, *Phys. Rev. B* **40**, 12255–12263 (1989). doi:[10.1103/PhysRevB.40.12255](https://doi.org/10.1103/PhysRevB.40.12255)
11. O. H. Nielsen, R. M. Martin, Quantum-mechanical theory of stress and force, *Phys. Rev. B* **32**, 3780–3791 (1985). doi:[10.1103/PhysRevB.32.3780](https://doi.org/10.1103/PhysRevB.32.3780)
12. D. C. Liu, J. Nocedal, On the limited memory BFGS method for large scale optimization, *Math. Program.* **45**, 503–528 (1989). doi:[10.1007/BF01589116](https://doi.org/10.1007/BF01589116)
13. Y. Hinuma, G. Pizzi, Y. Kumagai, F. Oba, I. Tanaka, Band structure diagram paths based on crystallography, *Comput. Mater. Sci.* **128**, 140–184 (2017). doi:[10.1016/j.commatsci.2016.10.015](https://doi.org/10.1016/j.commatsci.2016.10.015)
14. A. Togo, K. Shinohara, I. Tanaka, Spglib: a software library for crystal symmetry search, *Sci. Technol. Adv. Mater.: Methods* **4**, 2384822 (2024). doi:[10.1080/27660400.2024.2384822](https://doi.org/10.1080/27660400.2024.2384822)
15. S. L. Dudarev, G. A. Botton, S. Y. Savrasov, C. J. Humphreys, A. P. Sutton, Electron-energy-loss spectra and the structural stability of nickel oxide: an LSDA+U study, *Phys. Rev. B* **57**, 1505–1509 (1998). doi:[10.1103/PhysRevB.57.1505](https://doi.org/10.1103/PhysRevB.57.1505)
16. P. Hohenberg, W. Kohn, Inhomogeneous electron gas, *Phys. Rev.* **136**, B864–B871 (1964). doi:[10.1103/PhysRev.136.B864](https://doi.org/10.1103/PhysRev.136.B864)
17. W. Kohn, L. J. Sham, Self-consistent equations including exchange and correlation effects, *Phys. Rev.* **140**, A1133–A1138 (1965). doi:[10.1103/PhysRev.140.A1133](https://doi.org/10.1103/PhysRev.140.A1133)
18. E. Tiesinga, P. J. Mohr, D. B. Newell, B. N. Taylor, CODATA recommended values of the fundamental physical constants: 2018, *Rev. Mod. Phys.* **93**, 025010 (2021). doi:[10.1103/RevModPhys.93.025010](https://doi.org/10.1103/RevModPhys.93.025010)
19. R. M. Martin, *Electronic Structure: Basic Theory and Practical Methods*, 2nd ed. (Cambridge University Press, 2020). doi:[10.1017/9781108555586](https://doi.org/10.1017/9781108555586)
