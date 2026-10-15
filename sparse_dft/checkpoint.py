"""Portable, checked export of a converged complex PAW state for C++ NSCF."""
import hashlib
import json
import math
from pathlib import Path
from .errors import InputError

# Canonical option values. Earlier releases wrote these with a "vasp_" prefix;
# the old spellings stay accepted on input and are normalised at parse time.
PROJECTOR_INTERPOLATION_ALIASES = {'vasp_cubic': 'cubic', 'vasp_spline': 'spline'}
ONECENTER_RADIAL_MODE_ALIASES = {'vasp_aspherical_gga': 'aspherical_gga', 'vasp_radial_gga': 'radial_gga'}
# Frozen one-centre state keys: canonical name -> name written by earlier releases.
ONECENTER_STATE_LEGACY_KEYS = {'onsite_occupation': 'crhode', 'onsite_dij_ev': 'cdij_ev',
                               'lm_moment_imag': 'rholm_imag_vasp'}


def canonical_projector_interpolation(value):
    return PROJECTOR_INTERPOLATION_ALIASES.get(value, value) if isinstance(value, str) else value


def canonical_onecenter_radial_mode(value):
    return ONECENTER_RADIAL_MODE_ALIASES.get(value, value) if isinstance(value, str) else value


def onecenter_state_field(atom, name):
    """Read a frozen one-centre field by its canonical key or its earlier spelling."""
    if name in atom:
        return atom[name]
    legacy = ONECENTER_STATE_LEGACY_KEYS.get(name)
    if legacy is not None and legacy in atom:
        return atom[legacy]
    raise InputError(f'frozen PAW state is missing {name}')


def export_frozen_scf(source, output, *, cutoff_ev=None, density_cutoff_ev=None, include_spectrum=False):
    source, output = Path(source), Path(output)
    state = json.loads(source.read_text(encoding='utf-8'))
    if not state.get('converged') or not state.get('onecenter_closure_diagnostic'):
        raise InputError('frozen NSCF requires a converged full PAW SCF result')
    if state.get('total_energy_valid'):
        raise InputError('frozen PAW diagnostic must preserve its validation status')
    provenance = state.get('provenance', {})
    source_cutoff = state.get('cutoff_ev', provenance.get('cutoff_ev'))
    if source_cutoff is None:
        for filename in ('scf_input.sdft', 'input.sdft', 'bands_input.sdft'):
            path = source.parent / filename
            if path.exists():
                for line in path.read_text(encoding='utf-8').splitlines():
                    if line.startswith('cutoff_ev '):
                        source_cutoff = float(line.split()[1])
                        break
            if source_cutoff is not None:
                break
    if source_cutoff is None or not math.isfinite(source_cutoff) or source_cutoff <= 0 or (
            cutoff_ev is not None and abs(source_cutoff-cutoff_ev) > 1e-10):
        raise InputError('frozen SCF cutoff is missing or differs from the requested NSCF cutoff')
    source_density_cutoff = state.get('density_cutoff_ev', provenance.get('density_cutoff_ev'))
    if source_density_cutoff is None or not math.isfinite(source_density_cutoff) or source_density_cutoff < 4*source_cutoff:
        raise InputError('frozen SCF density_cutoff_ev is missing or invalid; recompute SCF with fine-density provenance')
    if density_cutoff_ev is not None and (not math.isfinite(density_cutoff_ev) or
            abs(source_density_cutoff-density_cutoff_ev) > 1e-10):
        raise InputError('NSCF density_cutoff_ev differs from the frozen SCF')
    state['density_cutoff_ev'] = source_density_cutoff
    units = state.get('atomic_units')
    if units not in ('codata', 'vasp_legacy'):
        raise InputError('frozen SCF must record an unambiguous atomic-unit convention')
    upf = state.get('paw_framework') == 'upf_paw'
    common = state.get('onecenter_radial_mode') in ('per_site_original_variational_paw_development', 'per_site_original_energy_adjoint_paw_development')
    spin = state.get('spin_channels', 1)
    if spin not in (1, 2):
        raise InputError('unsupported frozen spin model')
    convention = 'per_site_original_paw_hubbard_v1' if common else 'upf_pseudo_qae_bound_v1' if upf else 'ae_partial_wave_rdep_angstrom_v1'
    corrected_hubbard = state.get('hubbard_occupation_convention') == convention
    if spin == 2:
        from .spin_model import validate_spin_model
        pairs = state.get('spin_onecenter_states')
        if not isinstance(pairs, list) or len(pairs) != 2 or not pairs[0] or len(pairs[0]) != len(pairs[1]):
            raise InputError('frozen spin state requires both complete PAW channels')
        atoms = pairs[0]
        _, specs = validate_spin_model([a['symbol'] for a in atoms], 2, None, state.get('hubbard_specs'))
        if 'hubbard_specs' not in state:
            raise InputError('frozen spin state is missing its explicit Hubbard model')
        if any(spec['ueff_ev'] > 0 for spec in specs) and not corrected_hubbard:
            raise InputError('legacy +U checkpoint uses an unverified PAW radial convention; recompute SCF')
        if state.get('eigensolver_max_residual_hartree', math.inf) > 1e-12:
            raise InputError('frozen spin state has not passed the final fresh 1e-12 Ha gate')
        include_spectrum = True
    else:
        atoms = state['onecenter_states']
    for atom in (atoms if spin == 1 else pairs[0] + pairs[1]):
        cache = Path(atom['paw_cache'])
        if hashlib.sha256(cache.read_bytes()).hexdigest() != atom['numeric_cache_sha256']:
            raise InputError('frozen PAW cache SHA-256 mismatch')
        if atom.get('atomic_units', units) != units:
            raise InputError('frozen PAW atoms use different atomic-unit conventions')
    radial_mode = canonical_onecenter_radial_mode(state.get('onecenter_radial_mode', 'radial_gga'))
    if radial_mode not in (('per_site_original_variational_paw_development', 'per_site_original_energy_adjoint_paw_development') if common else ('UPF_original_QijL_discrete_variational_multik_development',) if upf else ('radial_gga', 'aspherical_gga')):
        raise InputError('unsupported frozen PAW radial mode')
    interpolation = canonical_projector_interpolation(state.get('projector_interpolation', 'cubic'))
    if interpolation not in (('per_site_original_paw', 'cubic') if common else ('upf_original_radial_quadrature',) if upf else ('cubic', 'spline')):
        raise InputError('unsupported frozen projector interpolation')
    if interpolation == 'spline' or upf or common:
        include_spectrum = True
    lines = ['SDFT_FROZEN_SCF_V10' if common else 'SDFT_FROZEN_SCF_V9' if upf and spin == 2 and corrected_hubbard else 'SDFT_FROZEN_SCF_V8' if upf and spin == 1 else 'SDFT_FROZEN_SCF_V7' if spin == 2 and corrected_hubbard else
             'SDFT_FROZEN_SCF_V6' if spin == 2 else 'SDFT_FROZEN_SCF_V5' if interpolation == 'spline' else
             'SDFT_FROZEN_SCF_V4' if include_spectrum else 'SDFT_FROZEN_SCF_V3',
             f'{units} {source_cutoff:.17g} {source_density_cutoff:.17g}', radial_mode]
    if common:
        lines.insert(2, f"{state['paw_framework']} {spin}")
    if interpolation == 'spline' or spin == 2 or upf or common:
        lines.append(interpolation)
    def numbers(values):
        values = list(values)
        if any(not math.isfinite(float(x)) for x in values):
            raise InputError('nonfinite frozen SCF value')
        return ' '.join(f'{float(x):.17g}' for x in values)
    if spin == 2:
        if corrected_hubbard:
            lines.append(convention)
        lines.append(f'2 {len(specs)}')
        lines.extend(f"{a['l']} {a['ueff_ev']:.17g}" for a in specs)
    for row in state['onecenter_lattice_bohr']:
        lines.append(numbers(row))
    reference = state.get('basis_reference_lattice_bohr')
    lines.append('1' if reference is not None else '0')
    if reference is not None:
        if len(reference) != 3 or any(len(row) != 3 for row in reference):
            raise InputError('frozen SCF reference lattice must be a 3x3 matrix')
        for row in reference:
            lines.append(numbers(row))
    keys = ('converged', 'onecenter_closure_diagnostic', 'total_energy_valid', 'iterations',
            'occupied_bands', 'electron_count', 'fermi_energy_ev', 'density_residual',
            'onecenter_density_residual', 'onecenter_density_tolerance',
            'integrated_electron_count', 'compensation_electron_count')
    lines.append(numbers(state[k] for k in keys))
    for name in ('density_q', 'hamiltonian_density_q', 'frozen_core_q'):
        entries = state.get(name, [])
        lines.append(str(len(entries)))
        for entry in entries:
            lines.append(f"{entry['h']} {entry['k']} {entry['l']} " + numbers([entry['value_real'], entry['value_imag']]))
    points = state['scf_kpoints']
    lines.append(str(len(points)))
    for point in points:
        lines.append(numbers([*point['fractional'], point['weight']]))
    if include_spectrum:
        for point in points:
            eigenvalues, occupations = point['eigenvalues_ev'], point['occupations']
            if not eigenvalues or len(eigenvalues) != len(occupations) or any(x < 0 or x > (1 if spin == 2 else 2) for x in occupations):
                raise InputError('frozen spectrum or occupations have invalid dimensions/values')
            lines.append(str(len(eigenvalues)))
            lines.append(numbers(eigenvalues))
            lines.append(numbers(occupations))
    def append_atoms(atoms):
        lines.append(str(len(atoms)))
        for atom in atoms:
            lines.append(f"{atom['atom_index']} {atom['projector_dimension']} " +
                         json.dumps(atom['symbol'], ensure_ascii=False) + ' ' + numbers(atom['fractional_position']) + ' ' +
                         json.dumps(atom['paw_cache'], ensure_ascii=False) + ' ' +
                         json.dumps(atom['paw_record_id'], ensure_ascii=False) +
                         f" {atom['numeric_cache_version']} {atom['numeric_cache_checksum']:.17g} {atom['numeric_cache_sha256']} " +
                         f"{atom['projector_scale']:.17g} {int(atom['projector_q_bohr'])} {int(atom['reciprocal_linear'])}")
            channel = atom['radial_channel_l']
            lines.append(str(len(channel)) + ' ' + ' '.join(str(x) for x in channel))
            for name in ('onsite_occupation', 'onsite_dij_ev', 'closure_correction_ev', 'compensation_correction_ev'):
                matrix = onecenter_state_field(atom, name)
                dimension = atom['projector_dimension']
                if len(matrix) != dimension or any(len(row) != dimension for row in matrix):
                    raise InputError('frozen PAW matrix dimensions differ')
                lines.append(str(dimension*dimension))
                for row in matrix:
                    for value in row:
                        lines.append(numbers(value))
            imaginary = onecenter_state_field(atom, 'lm_moment_imag')
            lines.append(str(len(imaginary)) + ' ' + numbers(imaginary))
            lines.append(numbers(atom[k] for k in ('closure_ae_energy_ev', 'closure_ps_energy_ev',
                         'closure_energy_difference_ev', 'closure_potential_expectation_ev', 'closure_core_xc_energy_ev')))
    append_atoms(atoms)
    if spin == 2:
        for name in ('spin_density_q', 'spin_hamiltonian_density_q'):
            channels = state.get(name)
            if not isinstance(channels, list) or len(channels) != 2:
                raise InputError('frozen spin state is missing both density channels')
            for entries in channels:
                lines.append(str(len(entries)))
                for entry in entries:
                    lines.append(f"{entry['h']} {entry['k']} {entry['l']} " + numbers([entry['value_real'], entry['value_imag']]))
        append_atoms(pairs[1])
    lines.append('END')
    output.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    Path(str(output)+'.sha256').write_text(checksum+'\n', encoding='utf-8')
    return state, checksum
