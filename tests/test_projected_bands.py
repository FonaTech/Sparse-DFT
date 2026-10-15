import json
import tempfile
import unittest
from pathlib import Path

from sparse_dft.bands import parse_procar, from_payload, load_bands
from sparse_dft.analysis import _normalize_projections, analyze_bands
from sparse_dft.errors import InputError


def procar(spin=1, soc=False):
    lines=['PROCAR lm decomposed', '# of k-points: 2 # of bands: 2 # of ions: 1']
    for s in range(spin):
        if spin>1:lines += [f'spin component {s+1}']
        for k in range(2):
            lines += [f'k-point {k+1} : {k*.5:.8f} 0.00000000 0.00000000 weight = 0.5']
            for b in range(2):
                lines += [f'band {b+1} # energy {b+s*.4+k*.1-1:.8f} # occ. {1-b}']
                for c in range(4 if soc else 1):
                    v=-.02 if c else .2
                    lines += ['ion s p tot',f'1 {v} 0.3 {v+.3}',f'tot {v} 0.3 {v+.3}']
    return '\n'.join(lines)+'\n'


class ProjectedBandTests(unittest.TestCase):
    def test_standard_title_coordinates_and_collinear_spin_identity(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'PROCAR';p.write_text(procar(2))
            data=parse_procar(p)
        self.assertEqual(data['projection_kpoints']['up'],[(0.,0.,0.),(.5,0.,0.)])
        self.assertAlmostEqual(data['projection_energies_ev']['down'][0][0],-.6)
        bands=from_payload({'kpoints':[[0,0,0],[.5,0,0]],'eigenvalues':[[-1,0],[-.9,.1]],
            'spin_channels':{'up':[[-1,0],[-.9,.1]],'down':[[-.6,.4],[-.5,.5]]}})
        self.assertEqual(_normalize_projections(bands,data)['nions'],1)
        bands.spin_channels['down'][0][0]+=.01
        with self.assertRaisesRegex(InputError,'energies'):_normalize_projections(bands,data)

    def test_repeated_header_spin_and_truncated_soc(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'PROCAR'
            text=procar(2).replace('spin component 1','').replace('spin component 2','# of k-points: 2 # of bands: 2 # of ions: 1')
            p.write_text(text);self.assertEqual(set(parse_procar(p)['projection_weights']),{'up','down'})
            p.write_text(procar(soc=True));data=parse_procar(p)
            self.assertEqual(set(data['projection_weights']),{'charge','mx','my','mz'})
            self.assertLess(data['projection_weights']['mx'][0][0][0][0],0)
            p.write_text('\n'.join(procar(soc=True).splitlines()[:-4]))
            with self.assertRaises(InputError):parse_procar(p)

    def test_native_tensor_directory_and_scientific_plots(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'bands.json'
            payload={'kpoints':[[0,0,0],[.5,0,0]],'eigenvalues':[[-1,1],[-.5,1.5]],
                     'labels':['Γ','X'],'fermi_energy_ev':0.,'projection_orbitals':['s','p'],
                     'projection_atom_symbols':['Si'],'projector_weights':[0,.2,.1,.3,.2,.1,.3,.1],
                     'spin_projection_available':False}
            p.write_text(json.dumps(payload))
            self.assertEqual(load_bands(td).nkpoints,2)
            report=analyze_bands(td,output_dir=Path(td)/'plots')
            self.assertIn('total',report['projections']['projection_weights'])
            artifacts=report['plots']['artifacts']
            self.assertTrue(any('projection_local' in f for f in artifacts))
            self.assertTrue(any('bands_local.pdf' in f for f in artifacts))
            self.assertTrue(all(Path(f).stat().st_size>100 for f in artifacts))
            payload['projector_weights'][0]=float('nan');p.write_text(json.dumps(payload))
            with self.assertRaisesRegex(InputError,'nonfinite'):analyze_bands(td,render=False)

    def test_spin_plot_uses_down_energies_and_soc_signed_weights(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'PROCAR';p.write_text(procar(2))
            b=Path(td)/'bands.json';b.write_text(json.dumps({
                'kpoints':[[0,0,0],[.5,0,0]],'eigenvalues':[[-1,0],[-.9,.1]],
                'spin_channels':{'up':[[-1,0],[-.9,.1]],'down':[[-.6,.4],[-.5,.5]]},
                'fermi_energy_ev':0.}))
            report=analyze_bands(b,procar_path=p,output_dir=Path(td)/'spin')
            self.assertTrue(report['projections']['spin_projection_available'])
            p.write_text(procar(soc=True))
            payload=json.loads(b.read_text());payload.pop('spin_channels');b.write_text(json.dumps(payload))
            report=analyze_bands(b,procar_path=p,output_dir=Path(td)/'soc')
            self.assertEqual(report['projections']['spin_mode'],'noncollinear_charge_and_magnetization')
