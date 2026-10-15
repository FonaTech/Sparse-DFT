#!/usr/bin/env python3
"""Generate sparse_dft/data/elements.json from ase.data (one-off; runtime needs no ase).

Usage: .venv/bin/python tools/build_element_table.py [--out sparse_dft/data/elements.json]

Numbers are copied, never edited:
* covalent radii: ase.data.covalent_radii, which cites Cordero et al.,
  Dalton Trans. 2008, 2832-2838, doi:10.1039/B801115J. ase stores a 2.0 A
  placeholder for Z >= 97; those entries are written as null.
* masses: ase.data.atomic_masses (IUPAC 2013 standard atomic weights,
  doi:10.1515/pac-2015-0305, as cited by ase).
* free-atom moments: ase.data.ground_state_magnetic_moments (webelements.com,
  as cited by ase), in Bohr magnetons (2 S for the atomic ground term).
Group, period, block and series come from the atomic number alone.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PERIOD_END = (2, 10, 18, 36, 54, 86, 118)
# IUPAC lanthanoids La-Lu and actinoids Ac-Lr.
LANTHANOIDS = range(57, 72)
ACTINOIDS = range(89, 104)
# Conventional non-metal and metalloid sets (categorical, not numerical).
NONMETALS = {1, 2, 6, 7, 8, 9, 10, 15, 16, 17, 18, 34, 35, 36, 53, 54, 86}
METALLOIDS = {5, 14, 32, 33, 51, 52}
D_SERIES = {'3d': range(21, 31), '4d': range(39, 49), '5d': range(72, 81), '6d': range(104, 113)}


def period_group(z: int) -> tuple[int, int | None]:
    start = 0
    for period, end in enumerate(PERIOD_END, start=1):
        if z <= end:
            offset = z - start
            break
        start = end
    else:
        raise ValueError(z)
    if period == 1:
        return 1, 1 if z == 1 else 18
    if period in (2, 3):
        return period, offset if offset <= 2 else offset + 10
    if period in (4, 5):
        return period, offset
    if offset <= 2:
        return period, offset
    if offset <= 17:
        return period, None  # f series
    return period, offset - 14


def classify(z: int) -> dict:
    period, group = period_group(z)
    if z in LANTHANOIDS or z in ACTINOIDS:
        block = 'f'
    elif group in (1, 2) or z == 2:
        block = 's'
    elif group is not None and 13 <= group <= 18:
        block = 'p'
    else:
        block = 'd'
    series = next((name for name, zs in D_SERIES.items() if z in zs), None)
    if z in LANTHANOIDS:
        series = 'lanthanoid'
    elif z in ACTINOIDS:
        series = 'actinoid'
    family = {1: 'alkali', 2: 'alkaline_earth', 13: 'triel', 14: 'tetrel', 15: 'pnictogen',
              16: 'chalcogen', 17: 'halogen', 18: 'noble_gas'}.get(group)
    if z == 1:
        family = 'hydrogen'
    kind = 'nonmetal' if z in NONMETALS else 'metalloid' if z in METALLOIDS else 'metal'
    return {'period': period, 'group': group, 'block': block, 'series': series,
            'family': family, 'kind': kind}


def _value(x):
    x = float(x)
    return None if math.isnan(x) else x


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=ROOT / 'sparse_dft' / 'data' / 'elements.json')
    args = ap.parse_args(argv)
    import ase
    import ase.data as data
    source = Path(data.__file__)
    rows = []
    for z in range(1, len(data.chemical_symbols)):
        radius = _value(data.covalent_radii[z])
        moments = data.ground_state_magnetic_moments
        moment = moments[z] if z < len(moments) else None
        rows.append({'z': z, 'symbol': data.chemical_symbols[z], 'name': data.atomic_names[z],
                     'mass_amu': _value(data.atomic_masses[z]),
                     'covalent_radius_angstrom': None if z >= 97 else radius,
                     'free_atom_moment_muB': None if moment is None else _value(moment),
                     **classify(z)})
    payload = {
        'schema_version': 'sparse-dft-elements-1',
        'provenance': {
            'generator': 'tools/build_element_table.py',
            'ase_version': ase.__version__,
            'ase_data_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'covalent_radii': 'Cordero et al., Dalton Trans. 2008, 2832-2838, doi:10.1039/B801115J (via ase.data); null for Z>=97 where ase stores a placeholder',
            'masses': 'IUPAC 2013 standard atomic weights, doi:10.1515/pac-2015-0305 (via ase.data)',
            'free_atom_moment_muB': 'webelements.com ground-state moments (via ase.data)',
            'classification': 'derived from Z: IUPAC groups/periods, lanthanoids La-Lu, actinoids Ac-Lr; conventional nonmetal/metalloid sets',
        },
        'elements': rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'elements': len(rows), 'out': str(args.out)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
