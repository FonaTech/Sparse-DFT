"""Structure classification and recommended input parameters.

``analyze_structure`` reads only geometry, element classes and (when a PAW
root is given) the PAW valence/ENMAX of the selected datasets. ``suggest``
turns the profile into per-command option values with a basis for each.

Every recommendation carries ``basis``:

* ``anchor``      reproduces the rule used for the VASP anchor
                  (tools/vasp_anchor.py, docs/NUMERICAL_STATUS.md);
* ``code``        the Sparse-DFT code default, unchanged;
* ``sweep``       chosen from the v15 sparsity sweeps (16x-ENCUT reference);
* ``heuristic``   a rule of thumb, not validated against VASP;
* ``literature``  an external value with its source (Hubbard U only).

Nothing here starts a calculation. Hubbard U is suggested and never applied
unless the caller asks for it explicitly.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

from .errors import InputError
from .geometry import axis_vacuum, bonded_dimensionality, element
from .structure import Structure

TASKS = ('scf', 'bands', 'dos', 'relax')

# Category id -> short description. Order matters: classify() returns the
# first category whose rule matches.
CATEGORIES = {
    'molecule': 'molecule or cluster in a box (all axes vacuum)',
    'wire': 'one-dimensional wire or chain (two vacuum axes)',
    'monolayer': 'two-dimensional monolayer (one vacuum axis, one bonded layer)',
    'slab': 'surface slab (one vacuum axis, several atomic layers)',
    'f_electron': 'f-electron compound (lanthanoid or actinoid; outside the validated range)',
    'tm_oxide': 'transition-metal oxide, chalcogenide or halide with open-shell 3d ions',
    'ferromagnetic_metal': 'itinerant 3d ferromagnet (Fe, Co or Ni metal or alloy)',
    'vdw_bulk': 'van der Waals bulk (bonded layers, chains or molecules without vacuum)',
    'metal': 'metal or intermetallic alloy',
    'ionic_insulator': 'ionic insulator (closed-shell cations with oxide, nitride or halide anions)',
    'semiconductor': 'covalent or compound semiconductor (no open-shell d or f ions)',
    'large_cell': 'large cell (supercell, defect, interface or amorphous)',
    'other': 'none of the above: generic settings',
}

FERROMAGNETS = ('Fe', 'Co', 'Ni')
# Open-shell 3d ions that keep a moment in oxides and halides.
MAGNETIC_3D = ('Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu')
ANION_FAMILIES = ('chalcogen', 'halogen')
ANIONS_EXTRA = ('N',)
LARGE_CELL_ATOMS = 64
# Above this many atoms the sparsity sweeps measured wall-time gains at a
# certified Table 1 error (memory: Si64 3.41x, Al slab 693 -> 126 s).
SPARSITY_AUTO_ATOMS = 32

# Fermi-Dirac kBT for metals used for the VASP anchor (tools/vasp_anchor.py
# METAL_KT_EV); both codes agree to the Table 1 gates at this width.
METAL_KT_EV = 0.1
# Materials Project GGA+U values (eV, d channel, oxides and fluorides),
# fitted to oxide formation enthalpies with PBE PAW datasets.
MP_U_SOURCE = {
    'url': 'https://docs.materialsproject.org/methodology/materials-methodology/'
           'calculation-details/gga+u-calculations/hubbard-u-values',
    'references': ['L. Wang, T. Maxisch, G. Ceder, Phys. Rev. B 73, 195107 (2006), doi:10.1103/PhysRevB.73.195107',
                   'A. Jain et al., Phys. Rev. B 84, 045115 (2011), doi:10.1103/PhysRevB.84.045115'],
    'scope': 'oxides and fluorides only; values are tied to the MP PAW set and fitted to formation '
             'enthalpies, not to band gaps',
    'retrieved': '2026-10-15',
}
MP_U_EV = {'Co': 3.32, 'Cr': 3.7, 'Fe': 5.3, 'Mn': 3.9, 'Mo': 4.38, 'Ni': 6.2, 'V': 3.25, 'W': 6.2}
MP_U_ANIONS = ('O', 'F')
# Thickest sheet (atom extent along the vacuum normal) still treated as a
# monolayer: MoS2 3.1 A, phosphorene 2.1 A, graphene 0 A (heuristic).
MONOLAYER_MAX_THICKNESS_ANGSTROM = 4.5


@dataclass
class StructureProfile:
    formula: str
    natoms: int
    elements: list[str]
    counts: dict
    category: str
    tags: list[str] = field(default_factory=list)
    periodic_axes: list[bool] = field(default_factory=lambda: [True, True, True])
    vacuum: dict = field(default_factory=dict)
    geometric_dimensionality: int = 3
    bonded_dimensionality: int | None = None
    bonded_components: int | None = None
    thickness_angstrom: float | None = None
    symmetry: dict | None = None
    metals: list[str] = field(default_factory=list)
    anions: list[str] = field(default_factory=list)
    magnetic_3d: list[str] = field(default_factory=list)
    f_elements: list[str] = field(default_factory=list)
    oxidation_states: dict | None = None
    open_shell_d: list[str] = field(default_factory=list)
    metallic: bool | None = None
    electrons: float | None = None
    odd_electrons: bool | None = None
    paw: dict | None = None
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        out = asdict(self)
        out['category_description'] = CATEGORIES[self.category]
        return out


def _is_anion(symbol: str) -> bool:
    row = element(symbol)
    return row['family'] in ANION_FAMILIES or symbol in ANIONS_EXTRA


def _formula(species) -> tuple[str, dict]:
    counts = {}
    for s in species:
        counts[s] = counts.get(s, 0) + 1
    return ''.join(f'{e}{n if n > 1 else ""}' for e, n in counts.items()), counts


def _paw_summary(structure: Structure, paw_root, family: str) -> dict:
    from .paw import Paw54JsonArchive, PawSelectionPolicy
    archive = Paw54JsonArchive(paw_root)
    elements = list(dict.fromkeys(structure.species))
    choices, _ = PawSelectionPolicy(family).resolve(archive, elements)
    return {'family': family,
            'datasets': {e: {'record_id': c.record_id, 'title': c.title, 'zval': c.zval, 'enmax_ev': c.enmax_ev}
                         for e, c in choices.items()},
            'electrons': float(sum(choices[s].zval for s in structure.species)),
            'max_enmax_ev': max(c.enmax_ev for c in choices.values())}


def _symmetry(structure: Structure) -> dict | None:
    try:
        from .symmetry import analyze, choose_symprec
        report = choose_symprec(structure)
        info = analyze(structure, symprec=report['symprec'])
    except Exception as exc:  # spglib missing or failing: symmetry is optional
        return {'error': str(exc)}
    info['operations'] = report['operations']
    info['symprec_warnings'] = report['warnings']
    return info


# Nominal charges used only to seed moments (heuristic). Cations not listed
# are "variable" and share the remaining charge equally.
ANION_CHARGE = {'chalcogen': -2, 'halogen': -1}
FIXED_CATION_CHARGE = {'alkali': 1, 'alkaline_earth': 2}
FIXED_CATION_EXTRA = {'Sc': 3, 'Y': 3, 'La': 3, 'Al': 3, 'Ga': 3, 'In': 3, 'Zn': 2, 'Cd': 2,
                      'B': 3, 'Si': 4, 'P': 5}
IONIC_CATIONS = ('Al', 'Sc', 'Y', 'La')


def _anion_charge(symbol: str) -> int:
    return -3 if symbol == 'N' else ANION_CHARGE[element(symbol)['family']]


def _cation_charge(symbol: str) -> int | None:
    row = element(symbol)
    return FIXED_CATION_CHARGE.get(row['family'], FIXED_CATION_EXTRA.get(symbol))


def nominal_oxidation_states(counts: dict) -> dict | None:
    """Charge-balance oxidation states; None when the formula is ambiguous.

    Anions take their group charge (O/S/Se/Te -2, halogens -1, N -3). Fixed
    cations take ``_cation_charge``; one kind of variable cation (a transition
    metal or p-block metal) takes the remainder. Mixed valence gives a
    fractional value (Fe3O4: Fe +8/3). Polyanions (pyrite S2 2-) are not
    recognised, so the result is only a seed for initial moments.
    """
    anions = [e for e in counts if _is_anion(e)]
    if not anions:
        return None
    cations = [e for e in counts if e not in anions]
    variable = [e for e in cations if _cation_charge(e) is None]
    if len(variable) > 1:
        return None
    states = {a: float(_anion_charge(a)) for a in anions}
    fixed = sum(_cation_charge(e) * counts[e] for e in cations if e not in variable)
    remainder = -sum(states[a] * counts[a] for a in anions) - fixed
    for e in cations:
        if e not in variable:
            states[e] = float(_cation_charge(e))
    if variable:
        v = variable[0]
        charge = remainder / counts[v]
        if charge <= 0 or charge > (element(v)['group'] or 8):
            return None
        states[v] = charge
    elif abs(remainder) > 1e-9:
        return None
    return states


def d_electrons(symbol: str, oxidation: float) -> float:
    """d count of a transition-metal ion: group number minus oxidation state."""
    return max(0.0, min(10.0, element(symbol)['group'] - oxidation))


def f_electrons(symbol: str, oxidation: float) -> float:
    """4f/5f count of a lanthanoid/actinoid ion: Z - 54 (Xe) or Z - 86 (Rn) minus the charge."""
    row = element(symbol)
    core = 54 if row['series'] == 'lanthanoid' else 86
    return max(0.0, min(14.0, row['z'] - core - oxidation))


def high_spin_moment(d_count: float) -> float:
    """Hund high-spin unpaired d electrons (seed only; low-spin ions relax away)."""
    return d_count if d_count <= 5 else 10.0 - d_count


def analyze_structure(structure: Structure, paw_root=None, *, paw_family: str = 'standard',
                      with_symmetry: bool = True) -> StructureProfile:
    """Classify ``structure`` into one of CATEGORIES; geometry and element data only."""
    if structure.lattice is None:
        raise InputError('Sparse-DFT needs a cell: put the molecule in a box (extended XYZ Lattice=..., '
                         'or a POSCAR with at least 10 A of vacuum around it)')
    s = structure.normalized()
    formula, counts = _formula(s.species)
    elements = list(counts)
    rows = {e: element(e) for e in elements}
    vac = axis_vacuum(s)
    periodic = [not v for v in vac.vacuum_axes]
    geo_dim = sum(periodic)
    profile = StructureProfile(formula=formula, natoms=s.natoms, elements=elements, counts=counts,
                               category='other', periodic_axes=periodic, vacuum=vac.as_dict(),
                               geometric_dimensionality=geo_dim)
    profile.anions = [e for e in elements if _is_anion(e)]
    profile.metals = [e for e in elements if rows[e]['kind'] == 'metal']
    profile.f_elements = [e for e in elements if rows[e]['series'] in ('lanthanoid', 'actinoid')
                          and e not in ('La', 'Ac')]
    profile.magnetic_3d = [e for e in elements if e in MAGNETIC_3D]
    try:
        conn = bonded_dimensionality(s)
        profile.bonded_dimensionality = conn.dimensionality
        profile.bonded_components = len(conn.components)
    except InputError as exc:
        profile.warnings.append(f'connectivity skipped: {exc}')
    if geo_dim == 2:
        axis = periodic.index(False)
        profile.thickness_angstrom = (1.0 - vac.gap_fraction[axis]) * vac.spacing_angstrom[axis]
    if with_symmetry:
        profile.symmetry = _symmetry(s)
    if paw_root is not None:
        profile.paw = _paw_summary(s, paw_root, paw_family)
        profile.electrons = profile.paw['electrons']
        profile.odd_electrons = abs(profile.electrons % 2.0 - 1.0) < 1e-6
    classify(profile)
    return profile


def classify(p: StructureProfile) -> StructureProfile:
    """Set ``category``, ``metallic``, ``tags``, ``reasons`` and ``warnings`` in place."""
    gap = ', '.join(f'{g:.1f}' for g in p.vacuum['gap_angstrom'])
    p.reasons.append(f'largest empty gap per axis (along the plane normal): {gap} A; '
                     f'vacuum above 5 A on {3 - p.geometric_dimensionality} axis(es)')
    nonmetal = [e for e in p.elements if e not in p.metals]
    all_metal = not nonmetal
    if p.bonded_dimensionality is not None:
        p.reasons.append(f'bonded network dimensionality {p.bonded_dimensionality} '
                         f'({p.bonded_components} component(s), covalent radii x 1.20)')
    _chemistry_tags(p, all_metal)
    # Dimensionality first: vacuum settles molecule / wire / sheet.
    if p.geometric_dimensionality == 0:
        p.category = 'molecule'
        p.metallic = False
    elif p.geometric_dimensionality == 1:
        p.category = 'wire'
        p.metallic = _metallic_guess(p, all_metal)
    elif p.geometric_dimensionality == 2:
        thin = (p.thickness_angstrom or 0.0) <= MONOLAYER_MAX_THICKNESS_ANGSTROM
        p.category = 'monolayer' if thin else 'slab'
        p.reasons.append(f'sheet thickness {p.thickness_angstrom:.2f} A '
                         f'({"<=" if thin else ">"} {MONOLAYER_MAX_THICKNESS_ANGSTROM} A monolayer limit)')
        p.metallic = _metallic_guess(p, all_metal)
        p.warnings.append('no dipole correction in Sparse-DFT: asymmetric slabs carry a spurious field '
                          'across the vacuum; use a symmetric slab')
    if p.category in ('molecule', 'wire', 'monolayer', 'slab'):
        _common_warnings(p)
        return p
    # Bulk (3D periodic): chemistry decides.
    if p.f_elements:
        p.category = 'f_electron'
        p.metallic = None
    elif 'tm_oxide' in p.tags:
        p.category = 'tm_oxide'
        p.metallic = False
    elif all_metal and any(e in FERROMAGNETS for e in p.elements):
        p.category = 'ferromagnetic_metal'
        p.metallic = True
    elif p.bonded_dimensionality is not None and p.bonded_dimensionality < 3:
        p.category = 'vdw_bulk'
        p.metallic = _metallic_guess(p, all_metal)
        kind = {0: 'molecules', 1: 'chains', 2: 'layers'}[p.bonded_dimensionality]
        p.warnings.append(f'bonded {kind} held by dispersion: plain PBE has no van der Waals term, '
                          'so the interlayer distance relaxes too far; keep the experimental cell '
                          'or fix the stacking axis in cell relaxations')
    elif all_metal:
        p.category = 'metal'
        p.metallic = True
        afm = [e for e in p.elements if e in ('Cr', 'Mn')]
        if afm:
            p.warnings.append(f'{", ".join(afm)} order antiferromagnetically: the suggestion is non-spin-polarized; '
                              'an AFM state needs a magnetic supercell with alternating --initial-moments')
    elif p.anions and 'ionic' in p.tags:
        p.category = 'ionic_insulator'
        p.metallic = _metallic_guess(p, all_metal)
    elif not p.metals or 'semiconductor' in p.tags:
        p.category = 'semiconductor'
        p.metallic = False if not p.odd_electrons else None
    elif not [a for a in p.anions if a != 'N'] and \
            sum(p.counts[e] for e in p.metals) >= 0.5 * p.natoms:
        p.category = 'metal'
        p.metallic = True
        p.reasons.append('metal-rich compound without chalcogen/halogen anions (carbide, nitride, boride, '
                         'silicide): treated as metallic')
    else:
        p.category = 'other'
        p.metallic = None
        p.reasons.append('mixed metal/non-metal composition without a simple anion lattice')
    sym = p.symmetry or {}
    if sym.get('operations') == 1 and p.natoms > 8:
        p.tags.append('low_symmetry')
    if p.natoms >= LARGE_CELL_ATOMS and p.category in ('metal', 'semiconductor', 'ionic_insulator', 'other'):
        if p.category not in p.tags:
            p.tags.append(p.category)
        p.category = 'large_cell'
    _common_warnings(p)
    return p


def _metallic_guess(p: StructureProfile, all_metal: bool) -> bool | None:
    """Metal if all atoms are metals; gapped with anions; unknown otherwise (graphene, CNT, graphite).

    An odd electron count per cell leaves a partly filled band in a
    non-spin-polarized calculation, so such cells are never guessed gapped.
    Open-shell d ions make the gap depend on structure and magnetism
    (2H-MoS2 is gapped, 1T-MoS2 and NbSe2 are metals): unknown.
    """
    if all_metal:
        return True
    if p.odd_electrons:
        return None
    if p.anions and not p.open_shell_d:
        return False
    return None


def _chemistry_tags(p: StructureProfile, all_metal: bool) -> None:
    if all_metal:
        p.tags.append('all_metal')
        return
    states = nominal_oxidation_states(p.counts) if p.anions else None
    p.oxidation_states = states
    if states:
        p.tags.append('charge_balanced')
        p.reasons.append('nominal oxidation states: ' + ', '.join(f'{e} {v:+.3g}' for e, v in states.items()))
    d_block = [e for e in p.elements if element(e)['block'] == 'd' and element(e)['group'] < 12]
    for e in d_block:
        if states and e in states:
            if 0 < d_electrons(e, states[e]) < 10:
                p.open_shell_d.append(e)
        elif p.anions or e in MAGNETIC_3D:
            p.open_shell_d.append(e)  # valence unknown: assume open shell
    oxide_like = [a for a in p.anions if a != 'N']
    open_3d = [e for e in p.open_shell_d if e in MAGNETIC_3D]
    if open_3d and oxide_like:
        p.tags.append('tm_oxide')
        p.reasons.append(f'open-shell 3d ion(s) {", ".join(open_3d)} with {", ".join(oxide_like)} anions')
    cations = [e for e in p.elements if e not in p.anions]
    if p.geometric_dimensionality == 0:
        return
    if p.anions and not p.open_shell_d and not p.f_elements:
        ionic = all(element(e)['family'] in ('alkali', 'alkaline_earth') or e in IONIC_CATIONS
                    or element(e)['block'] == 'd' for e in cations)
        p.tags.append('ionic' if ionic and any(a in ('O', 'N') or element(a)['family'] == 'halogen'
                                               for a in p.anions) else 'semiconductor')
    elif not p.anions and not p.open_shell_d and not p.f_elements:
        p.tags.append('semiconductor')


def _common_warnings(p: StructureProfile) -> None:
    sym = p.symmetry or {}
    if sym.get('error'):
        p.warnings.append(f'symmetry analysis unavailable: {sym["error"]}')
    else:
        p.warnings.extend(sym.get('symprec_warnings', []))
        if sym.get('primitive_natoms') and sym['primitive_natoms'] < p.natoms:
            p.tags.append('supercell')
            p.reasons.append(f'primitive cell has {sym["primitive_natoms"]} of {p.natoms} atoms')
    if p.f_elements:
        p.warnings.append(f'f elements ({", ".join(p.f_elements)}): f-in-valence PAW datasets are outside '
                          'the VASP-anchored range; results are not validated')
    heavy = [e for e in p.elements if element(e)['period'] >= 6 and element(e)['kind'] != 'nonmetal']
    if heavy:
        p.warnings.append(f'heavy elements ({", ".join(heavy)}): Sparse-DFT has no spin-orbit coupling; '
                          'band splittings that need SOC are missing')
    if p.odd_electrons and p.metallic is not True:
        p.warnings.append('odd number of valence electrons: the cell is open-shell (radical, defect or '
                          'doping); a spin-polarized run is suggested')


# ---------------------------------------------------------------- suggestions
@dataclass
class Item:
    key: str
    value: object
    basis: str
    note: str
    option: str | None = None  # CLI flag; None = informational only

    @property
    def dest(self) -> str | None:
        return None if self.option is None else self.option.lstrip('-').replace('-', '_')

    def as_dict(self) -> dict:
        return {'key': self.key, 'option': self.option, 'dest': self.dest, 'value': self.value,
                'basis': self.basis, 'note': self.note}


COMMAND_FOR_TASK = {'scf': 'scf', 'bands': 'bands', 'dos': 'dos', 'relax': 'relax'}
# k spacing (1/A, 2 pi included) per accuracy level: sparse_dft.kpoints.recommend_kpoints.
KPOINT_SPACING = {'low': 0.30, 'standard': 0.18, 'high': 0.12, 'production': 0.10}
# Gapped systems converge exponentially in k (localized Wannier functions);
# 1.5x coarser spacing is a rule of thumb, not an anchored value.
GAPPED_SPACING_FACTOR = 1.5
# DOS NSCF mesh: twice as dense as the SCF mesh of the same accuracy level.
DOS_DENSITY_FACTOR = 2.0


def _moments(p: StructureProfile, species: list[str]) -> tuple[list[float] | None, str, str]:
    """Per-atom initial moments, basis and note; None when no spin polarization is suggested."""
    def free(e):
        return float(element(e)['free_atom_moment_muB'] or 0.0)
    if p.category == 'ferromagnetic_metal':
        seeded = FERROMAGNETS + ('Cr', 'Mn')
        return ([free(e) if e in seeded else 0.0 for e in species], 'anchor',
                'free-atom Hund moment on Cr/Mn/Fe/Co/Ni (rule of the VASP-anchored Fe and Ni runs)')
    if p.category == 'tm_oxide' or ('tm_oxide' in p.tags):
        states = p.oxidation_states or {}
        moments = []
        for e in species:
            if e in p.open_shell_d and e in MAGNETIC_3D:
                moments.append(high_spin_moment(d_electrons(e, states[e])) if e in states else free(e))
            else:
                moments.append(0.0)
        return (moments, 'heuristic', 'high-spin d count from nominal oxidation states, ferromagnetic seed; '
                'antiferromagnets (NiO, MnO, Cr2O3) need a magnetic supercell with alternating signs')
    if p.f_elements:
        states = p.oxidation_states or {}

        def seed(e):
            if e in p.f_elements:
                if e not in states:
                    return free(e)
                n = f_electrons(e, states[e])
                return n if n <= 7 else 14.0 - n
            return free(e) if e in p.open_shell_d else 0.0
        moments = [seed(e) for e in species]
        return (moments, 'heuristic', 'Hund f (and d) count from nominal oxidation states (Ce4+ 4f0, Gd3+ 4f7); '
                'f-electron results are not validated')
    if p.metallic and any(e in FERROMAGNETS for e in p.elements):
        return ([free(e) if e in FERROMAGNETS else 0.0 for e in species], 'anchor',
                'Fe/Co/Ni present: free-atom moment seed as for the anchored ferromagnets')
    if p.odd_electrons and p.metallic is not True:
        n = len(species)
        return ([round(1.0 / n, 6)] * n, 'heuristic', 'odd electron count: one unpaired electron spread over the cell')
    return None, 'anchor', 'closed shell: non-spin-polarized (Si, GaAs, MgO, LiF, Al, CO anchors)'


def _smearing(p: StructureProfile) -> tuple[float, str, str]:
    if p.metallic:
        return METAL_KT_EV, 'anchor', 'Fermi-Dirac kBT 0.1 eV, the width of the VASP-anchored metals'
    if p.category == 'tm_oxide':
        return 0.05, 'heuristic', ('open-shell 3d oxide: PBE without +U often closes the gap; a small Fermi-Dirac '
                                   'width keeps the SCF stable; set 0 once a gap is confirmed')
    if p.metallic is None:
        return 0.05, 'heuristic', ('gap not predictable from composition (open-shell d ions, odd electron count '
                                   'or sp2 carbon): 0.05 eV Fermi-Dirac keeps a metal stable and changes a gapped '
                                   'case by exp(-gap/2kT); set 0 once a gap is confirmed (the MoS2 anchor used 0)')
    return 0.0, 'anchor', 'gapped system: zero-temperature occupations (Si, MgO, LiF, MoS2, CO anchors)'


def _masked(structure: Structure, periodic) -> Structure:
    s = structure.normalized()
    return Structure(list(s.species), list(s.positions), s.lattice, True, tuple(periodic),
                     s.selective_dynamics, s.source_format, s.source_path, list(s.warnings))


def suggest(profile: StructureProfile, task: str = 'scf', structure: Structure | None = None, *,
            accuracy: str = 'standard') -> dict:
    """Recommended options for ``sparse_dft <task>``; ``structure`` enables the k mesh and band path."""
    from .kpoints import recommend_kpoints
    from .settings import automatic_cutoff
    task = str(task).lower()
    if task not in TASKS:
        raise InputError('unsupported preset task', details={'task': task, 'supported': list(TASKS)})
    p = profile
    items: list[Item] = []
    notes: list[str] = []

    # Basis: cutoff and empty bands stay with the Sparse-DFT rules the anchor validated.
    if p.paw:
        cutoff = automatic_cutoff([d['enmax_ev'] for d in p.paw['datasets'].values()], accuracy)
        items.append(Item('cutoff_ev', cutoff, 'anchor',
                          f'automatic: max(300, 1.30 x max ENMAX {p.paw["max_enmax_ev"]:g} eV), the default-tier rule '
                          'compared with VASP; left to Sparse-DFT (no --cutoff)'))
    else:
        items.append(Item('cutoff_ev', None, 'code', 'automatic from PAW ENMAX once a PAW root is set'))
    items.append(Item('accuracy', accuracy, 'anchor' if accuracy == 'standard' else 'code',
                      'standard: cutoff margin 1.30 x max ENMAX, as in the VASP-anchored default tier'
                      if accuracy == 'standard' else
                      f'{accuracy}: cutoff margin per sparse_dft.settings.automatic_cutoff; not compared with VASP',
                      '--accuracy'))
    items.append(Item('bands', 0, 'anchor', 'automatic: occupied + 4 (+8 spin), widened when the smeared top '
                      'band stays occupied (default tier, all anchors)'))

    # k sampling: vacuum axes fixed at 1.
    if structure is not None and p.geometric_dimensionality > 0:
        if accuracy not in KPOINT_SPACING:
            raise InputError('unsupported accuracy level', details={'accuracy': accuracy})
        gapped = p.metallic is False
        spacing = KPOINT_SPACING[accuracy] * (GAPPED_SPACING_FACTOR if gapped else 1.0)
        masked = _masked(structure, p.periodic_axes)
        rec = recommend_kpoints(masked, 'relax' if task == 'relax' else 'scf', accuracy, spacing, bool(p.metallic))
        items.append(Item('scf_grid', list(rec.grid), 'heuristic' if gapped else 'code',
                          ('gapped: spacing x1.5 (rule of thumb); ' if gapped else '') +
                          f'sparse_dft.kpoints.recommend_kpoints({accuracy}): spacing {rec.spacing_inverse_angstrom:.3f} 1/A '
                          f'(2 pi included), {math.prod(rec.grid)} points on the full mesh (no symmetry reduction); '
                          'converge separately with the convergence_scan spacings'
                          + ('; vacuum axes fixed at 1' if not all(p.periodic_axes) else ''), '--scf-grid'))
        items.append(Item('mesh_mode', 'gamma', 'anchor',
                          'Gamma-centred mesh, as in every anchor run (hexagonal and odd meshes need it)', '--mesh-mode'))
        if task == 'dos':
            dos = recommend_kpoints(masked, 'nscf', accuracy, spacing / DOS_DENSITY_FACTOR, bool(p.metallic))
            items.append(Item('dos_grid', list(dos.grid), 'heuristic',
                              f'NSCF DOS mesh at half the SCF spacing ({dos.spacing_inverse_angstrom:.3f} 1/A, '
                              f'{math.prod(dos.grid)} points)', '--dos-grid'))
    elif p.geometric_dimensionality == 0:
        items.append(Item('scf_grid', [1, 1, 1], 'anchor', 'molecule: Gamma only (CO anchor)', '--scf-grid'))
        items.append(Item('mesh_mode', 'gamma', 'anchor', 'Gamma only', '--mesh-mode'))

    kt, basis, note = _smearing(p)
    items.append(Item('smearing_ev', kt, basis, note, '--smearing-ev'))

    moments, basis, note = _moments(p, list(structure.normalized().species) if structure else
                                    [e for e in p.elements for _ in range(p.counts[e])])
    # bands: the same spin model as the SCF; with --scf-checkpoint run_bands
    # checks it against the frozen model instead of silently overriding it.
    if moments is not None and any(abs(m) > 0 for m in moments):
        items.append(Item('spin_channels', 2, basis, 'collinear spin polarization', '--spin-channels'))
        items.append(Item('initial_moments', moments, basis, note, '--initial-moments'))
    else:
        items.append(Item('spin_channels', 1, basis, note, '--spin-channels'))

    hubbard = suggested_hubbard(p)
    if hubbard:
        items.append(Item('hubbard', hubbard['values'], 'literature', hubbard['note'], None))

    if task == 'relax':
        items.append(Item('density_tolerance', 1e-8, 'code',
                          'relax default 1e-8 (forces need a tighter density than energies)', None))
        if p.category in ('vdw_bulk',):
            notes.append('cell relaxation of a van der Waals solid without a dispersion term expands '
                         'the stacking axis; prefer ionic relaxation at the experimental cell')
        if p.geometric_dimensionality < 3:
            items.append(Item('cell', False, 'heuristic',
                              'vacuum present: ionic relaxation only (cell relaxation would shrink the vacuum)',
                              '--cell'))
    sparsity = 'auto' if p.natoms >= SPARSITY_AUTO_ATOMS and p.category not in ('molecule', 'f_electron') else 'off'
    items.append(Item('sparsity', sparsity, 'sweep' if sparsity == 'auto' else 'code',
                      'auto: certified against the 16x-ENCUT reference (Table 1) at >= 32 atoms, where it '
                      'saves wall time' if sparsity == 'auto' else
                      'off: small cells gain little and the default tier is the VASP-anchored one',
                      '--sparsity' if task not in ('bands', 'dos') else None))

    if task in ('bands', 'dos') and p.geometric_dimensionality == 0:
        notes.append('molecule: no band dispersion; read the Gamma-point levels from scf.json instead')
    elif task == 'bands' and structure is not None:
        try:
            path = band_path(structure)
        except Exception as exc:  # spglib/seekpath missing or no segment in the periodic plane
            notes.append(f'automatic band path unavailable: {exc}')
        else:
            route = ' | '.join('-'.join(seg) for seg in path['segments'])
            items.append(Item('band_path', route, 'code',
                              f'seekpath (HPKOT) on the input cell, {len(path["kpoints"])} points at '
                              f'{path["spacing_inverse_angstrom"]} 1/A; dropped out-of-plane segments: '
                              f'{len(path["dropped_segments"])}', None))
            notes.extend(path['warnings'])
    return {'schema_version': 'sparse-dft-suggest-1', 'task': task, 'command': COMMAND_FOR_TASK[task],
            'accuracy': accuracy, 'category': p.category, 'items': [i.as_dict() for i in items],
            'options': options_from_items(items), 'notes': notes, 'warnings': list(p.warnings)}


def options_from_items(items) -> dict:
    """``{dest: value}`` for items that map to a CLI option (GUI and argv builders use this)."""
    out = {}
    for item in items:
        item = item if isinstance(item, Item) else Item(**{k: item[k] for k in ('key', 'value', 'basis', 'note', 'option')})
        if item.option is not None and item.value is not None:
            out[item.dest] = item.value
    return out


def suggested_hubbard(p: StructureProfile) -> dict | None:
    """MP U values for open-shell d ions in oxides/fluorides; a suggestion only."""
    anions = [a for a in p.anions if a in MP_U_ANIONS]
    targets = [e for e in p.open_shell_d if e in MP_U_EV]
    if not anions or not targets:
        return None
    return {'values': [f'{e}:2:{MP_U_EV[e]:g}' for e in targets],
            'note': ('not applied: Dudarev U_eff (eV) on d from the Materials Project table '
                     f'({MP_U_SOURCE["url"]}; {"; ".join(MP_U_SOURCE["references"])}); '
                     f'{MP_U_SOURCE["scope"]}; pass --hubbard to use it'),
            'source': MP_U_SOURCE}


def band_path(structure: Structure, **kwargs) -> dict:
    """``sparse_dft.symmetry.auto_band_path`` with the vacuum mask; JSON ready for ``sparse-dft bands --path``."""
    from .symmetry import auto_band_path
    return auto_band_path(structure, **kwargs)


def suggest_file(path, task: str = 'scf', paw_root=None, *, accuracy: str = 'standard',
                 band_path_out=None) -> dict:
    """Profile plus suggestions for a structure file; optionally write the band path JSON."""
    import json
    from pathlib import Path
    from .structure import read_structure
    structure = read_structure(path)
    profile = analyze_structure(structure, paw_root)
    result = suggest(profile, task, structure, accuracy=accuracy)
    if band_path_out is not None and task == 'bands' and profile.geometric_dimensionality > 0:
        data = band_path(structure)
        target = Path(band_path_out)
        target.write_text(json.dumps(data, indent=1) + '\n', encoding='utf-8')
        result['options']['path'] = str(target)
        result['band_path_file'] = str(target)
    return {'structure': str(path), 'profile': profile.as_dict(), 'suggestion': result}


def suggested_argv(payload: dict, paw_root=None) -> list[str]:
    """``sparse_dft`` command line for a ``suggest_file`` payload (quoted for a POSIX shell)."""
    import shlex
    s = payload['suggestion']
    argv = ['sparse-dft', s['command'], payload['structure']]
    if paw_root:
        argv += ['--paw-root', str(paw_root)]
    for dest, value in s['options'].items():
        flag = '--' + dest.replace('_', '-')
        if isinstance(value, bool):
            if value:
                argv.append(flag)
            continue
        argv.append(flag)
        argv += [str(v) for v in value] if isinstance(value, (list, tuple)) else [str(value)]
    return [shlex.quote(x) for x in argv]
