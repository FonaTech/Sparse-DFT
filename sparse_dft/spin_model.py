"""Explicit collinear/Dudarev model; no implicit magnetic approximation."""
import math
from .errors import InputError


def validate_spin_model(species, spin_channels=1, initial_moments=None, hubbard=None, zvals=None):
    if type(spin_channels) is not int or spin_channels not in (1, 2):
        raise InputError('spin_channels must be 1 (nonmagnetic) or 2 (collinear)')
    try:
        moments = [0.0] * len(species) if initial_moments is None else [float(x) for x in initial_moments]
    except (TypeError, ValueError, OverflowError) as exc:
        raise InputError('initial_moments must be finite moments in muB, one per atom') from exc
    if len(moments) != len(species) or any(not math.isfinite(x) for x in moments):
        raise InputError('initial_moments must be finite moments in muB, one per atom')
    if zvals is not None and any(abs(m) > float(zvals[s]) for s, m in zip(species, moments)):
        raise InputError('initial magnetic moment exceeds the selected PAW valence count')
    if hubbard is None:
        specs = [{'l': -1, 'ueff_ev': 0.0} for _ in species]
    elif isinstance(hubbard, dict):
        if set(hubbard) - set(species):
            raise InputError('Hubbard element is absent from the structure')
        specs = [hubbard.get(s, {'l': -1, 'ueff_ev': 0.0}) for s in species]
    elif isinstance(hubbard, (list, tuple)) and len(hubbard) == len(species):
        specs = list(hubbard)
    else:
        raise InputError('hubbard must map elements to {l, ueff_ev}, or contain one specification per atom')
    checked = []
    for spec in specs:
        if not isinstance(spec, dict) or set(spec) != {'l', 'ueff_ev'}:
            raise InputError('Dudarev specification requires exactly l and ueff_ev; separate U/J is unsupported')
        l = spec['l']
        try:
            u = float(spec['ueff_ev'])
        except (TypeError, ValueError, OverflowError) as exc:
            raise InputError('Ueff must be a finite nonnegative energy in eV') from exc
        if type(l) is not int or l not in (-1, 0, 1, 2, 3) or not math.isfinite(u) or u < 0 or (l == -1 and u != 0):
            raise InputError('Dudarev l must be -1..3 and Ueff finite/nonnegative; l=-1 disables U')
        checked.append({'l': l, 'ueff_ev': u})
    if spin_channels == 1 and (any(moments) or any(x['l'] >= 0 for x in checked)):
        raise InputError('magnetic moments and Dudarev specifications require spin_channels=2')
    return moments, checked


def parse_cli_hubbard(values):
    result = {}
    for value in values or []:
        try:
            symbol, l, u = value.split(':')
            if symbol in result:
                raise ValueError('duplicate element')
            result[symbol] = {'l': int(l), 'ueff_ev': float(u)}
        except (ValueError, TypeError) as exc:
            raise InputError('use --hubbard ELEMENT:L:UEFF_EV once per element, e.g. Fe:2:4') from exc
    return result or None
