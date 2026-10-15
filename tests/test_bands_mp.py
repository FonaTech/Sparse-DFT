import json
import tempfile
import unittest
from pathlib import Path

from sparse_dft.bands import from_json, from_payload, compare_bands, parse_eigenval, parse_doscar, density_of_states_from_bands, density_of_states_from_scf
from sparse_dft.analysis import analyze_bands
from sparse_dft.errors import InputError
from sparse_dft.mp_data import MaterialsProjectClient, write_snapshot, load_snapshot, write_band_snapshot, load_band_snapshot


class BandsAndMPTests(unittest.TestCase):
    def test_vasp_eigenval_and_doscar_parsers(self):
        root = Path(__file__).resolve().parents[1]
        b = parse_eigenval(root / 'tests' / 'fixtures' / 'vasp' / 'si2_gamma_100' / 'EIGENVAL', fermi_energy=7.25904383)
        self.assertEqual((b.nkpoints, b.nbands), (1, 8))
        d = parse_doscar(root / 'tests' / 'fixtures' / 'vasp' / 'si2_gamma_100' / 'DOSCAR')
        self.assertEqual(d['nedos'], 301)
        self.assertAlmostEqual(d['fermi_energy_ev'], 7.25904383, places=5)
        self.assertEqual(len(d['energies_ev']), 301)

    def test_analysis_report_has_provenance_and_svg(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            report = analyze_bands(root / 'tests' / 'fixtures' / 'sparse_dft' / 'si_gamma_bands.json',
                                   structure=root / 'examples' / 'Si.POSCAR', output_dir=tmp)
            self.assertEqual(report['schema_version'], 'sparse-dft-band-analysis-1')
            self.assertTrue(Path(tmp, 'analysis.json').exists())
            self.assertTrue(Path(tmp, 'sparse_dft_bands.svg').exists())
            self.assertEqual(report['provenance']['parser'], 'sparse_dft.analysis/1')
            self.assertIsNone(report['dos'])

    def test_path_dos_rejected_and_weighted_mesh_normalized(self):
        payload = {'kpoints': [[0,0,0], [.5,0,0]], 'eigenvalues': [[-1,1],[-.5,1.5]],
                   'kweights': [.25,.75]}
        with self.assertRaises(InputError):
            density_of_states_from_bands(from_payload(payload))
        payload['metadata'] = {'sampling': 'uniform_mesh'}
        dos = density_of_states_from_bands(from_payload(payload), sigma=.05, npoints=5001)
        xs,ys = dos['energies_ev'],dos['total']['total']
        integral = sum((xs[i+1]-xs[i])*(ys[i+1]+ys[i])*.5 for i in range(len(xs)-1))
        self.assertAlmostEqual(integral,4.0,places=8)
        self.assertEqual(dos['metadata']['kweights'], [.25,.75])
        self.assertFalse(dos['metadata']['mesh_convergence_verified'])

    def test_dos_scf_rejects_unconverged_source(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'scf.json'
            path.write_text(json.dumps({'converged':False}))
            with self.assertRaises(InputError): density_of_states_from_scf(path)
    def test_band_alignment_and_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = {'kpoints': [[0, 0, 0], [0.5, 0, 0]], 'eigenvalues': [[-1, 1], [-0.5, 1.5]], 'fermi_energy': 0.0}
            shifted = {'kpoints': base['kpoints'], 'eigenvalues': [[-0.8, 1.2], [-0.3, 1.7]], 'fermi_energy': 0.2}
            (root / 'a.json').write_text(json.dumps(base)); (root / 'b.json').write_text(json.dumps(shifted))
            result = compare_bands(from_json(root / 'a.json'), from_json(root / 'b.json'), align='none')
            self.assertAlmostEqual(result['mae_ev'], 0.2, places=8)
            self.assertAlmostEqual(result['rmse_ev'], 0.2, places=8)

    def test_band_dimension_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'a.json').write_text(json.dumps({'kpoints': [[0, 0, 0], [1, 0, 0]], 'eigenvalues': [[0], [1]]}))
            (root / 'b.json').write_text(json.dumps({'kpoints': [[0, 0, 0], [1, 0, 0]], 'eigenvalues': [[0, 1], [1, 2]]}))
            with self.assertRaises(InputError): compare_bands(from_json(root / 'a.json'), from_json(root / 'b.json'))

    def test_mp_snapshot_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = write_snapshot(tmp, [{'material_id': 'mp-1', 'formula_pretty': 'Si', 'structure': {'lattice': {}}}], 'https://example.test/mp', requested_count=1)
            self.assertEqual(manifest['count'], 1)
            self.assertEqual(load_snapshot(Path(tmp) / 'mp_structures.json')['records'][0]['material_id'], 'mp-1')

    def test_mp_client_requires_key(self):
        with self.assertRaises(InputError): MaterialsProjectClient(api_key='')

    def test_raw_mp_band_snapshot_requires_eigenvalues(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = {'kpoints': [[0,0,0],[.5,0,0]], 'eigenvalues': [[-1,1],[-.5,1.5]], 'efermi': 0.0}
            manifest = write_band_snapshot(tmp, [{'material_id':'mp-1','bandstructure':payload}], 'https://example.test/bands', requested_ids=['mp-1'])
            self.assertTrue(manifest['raw_eigenvalues'])
            snap = load_band_snapshot(Path(tmp) / 'mp_bandstructures.json')
            self.assertEqual(snap['records'][0]['material_id'], 'mp-1')
            self.assertEqual(from_payload(payload).nbands, 2)
            with self.assertRaises(InputError):
                write_band_snapshot(tmp, [{'material_id':'mp-2','bandstructure':{'band_gap':1.2}}], 'x', requested_ids=['mp-2'])
