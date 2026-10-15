"""Compare a `sparse-dft dos` result with VASP on the same mesh and broadening.

VASP's DOSCAR uses its own smearing on its own grid (the anchors run
ISMEAR=-1, SIGMA 0.001 eV for insulators), so the reference DOS is rebuilt
here from the VASP eigenvalues, k weights and lm projections with the same
Gaussian, grid and band window as the Sparse-DFT file. Total DOS and every PDOS
column are compared by normalised L1 = int|D_sparse_dft - D_vasp| / int D_vasp.

    python tools/compare_vasp_dos.py SDFT_DOS_DIR VASP_SCF_DIR [--out JSON]

The VASP projections are read from vasprun.xml (4 decimals; PROCAR prints 3).
The two highest VASP bands are excluded, as in tools/anchor_report.py, and the
window stops 0.5 eV below the lowest of those excluded bands' minima.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sparse_dft.analysis.electronic import load_band_data, load_dos, load_projections  # noqa: E402
from sparse_dft.scf import gaussian_broaden  # noqa: E402

TOP_EXCLUDED = 2
EDGE_MARGIN_EV = 0.5


def compare(dos_dir: Path, vasp_dir: Path) -> dict:
    d = load_dos(dos_dir)
    vb = load_band_data(vasp_dir / 'vasprun.xml')
    vp = load_projections(vb, vasp_dir / 'vasprun.xml')
    energies = d.energies_ev
    step = energies[1] - energies[0]
    nb = vb.nbands - TOP_EXCLUDED
    # vasprun names the single ISPIN=1 channel 'up'; the Sparse-DFT file calls it 'total'
    chans = (vb.spin_channels if vb.spin_channels and len(vb.spin_channels) == 2
             else {'total': vb.eigenvalues})
    degeneracy = 1.0 if len(chans) == 2 else 2.0
    weights = [degeneracy * w / sum(vb.kweights) for w in vb.kweights]
    top = min(min(row[nb] for row in rows) for rows in chans.values())
    hi = min(top, d.metadata.get('complete_below_ev') or top) - EDGE_MARGIN_EV
    mask = [x <= hi for x in energies]

    def l1(a, b):
        diff = sum(abs(x - y) for x, y, m in zip(a, b, mask) if m) * step
        ref = sum(y for y, m in zip(b, mask) if m) * step
        return diff / ref if ref else None, ref, sum(x for x, m in zip(a, mask) if m) * step

    out = {'sparse_dft': str(dos_dir), 'vasp': str(vasp_dir), 'sigma_ev': d.sigma_ev, 'window_ev': [energies[0], hi],
           'vasp_bands_used': nb, 'projection_source': 'vasprun.xml', 'channels': {}}
    for name, rows in chans.items():
        proj = vp.channels[name if name in vp.channels else next(iter(vp.channels))]
        tensor = [[[v for atom in proj[k][b] for v in atom] for b in range(nb)] for k in range(vb.nkpoints)]
        curves = gaussian_broaden(energies, [r[:nb] for r in rows], weights, d.sigma_ev, tensor)
        spin = None if name == 'total' else name
        mine = next(c for c in d.total if c.spin == spin)
        rel, ref, own = l1(mine.values, curves[0])
        entry = {'total': {'l1_normalised': rel, 'integral_vasp': ref, 'integral_sparse_dft': own}, 'pdos': {}}
        no = len(vp.orbitals)
        for c in d.pdos_raw:
            if c.spin != spin or c.orbital not in vp.orbitals:
                continue
            column = curves[1 + (c.atoms[0] - 1) * no + vp.orbitals.index(c.orbital)]
            rel, ref, own = l1(c.values, column)
            entry['pdos'][f'{c.atoms[0]}:{c.label}'] = {'l1_normalised': rel, 'integral_vasp': ref,
                                                        'integral_sparse_dft': own}
        out['channels'][name] = entry
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('sparse_dft'); p.add_argument('vasp'); p.add_argument('--out')
    a = p.parse_args(argv)
    result = compare(Path(a.sparse_dft), Path(a.vasp))
    text = json.dumps(result, indent=1)
    if a.out:
        Path(a.out).write_text(text + '\n', encoding='utf-8')
    for name, entry in result['channels'].items():
        big = [v['l1_normalised'] for v in entry['pdos'].values() if v['integral_vasp'] >= 0.1]
        print(f"{name}: total L1 {entry['total']['l1_normalised']:.2e}; PDOS columns with integral >= 0.1: "
              f"max L1 {max(big, default=float('nan')):.2e} ({len(big)} of {len(entry['pdos'])})")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
