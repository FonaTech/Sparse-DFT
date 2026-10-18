"""Meaningful contract/failure tests; fixtures here are explicitly synthetic."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

S=Path(os.environ.get('SDFT_TEST_SKILL', str(Path(__file__).resolve().parents[1]/'skills/sparse-dft-driver')))/'scripts';sys.path.insert(0,str(S))
from source_catalog import source_catalog
from sparsity_workflow import frozen_profile,compare,prepare
SOURCE=Path(os.environ['SDFT_TEST_SOURCE']).resolve();sys.path.insert(0,str(SOURCE))

class SparseFeatures(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.p=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def put(self,name,obj):
        p=self.p/name;p.write_text(json.dumps(obj));return p
    def test_dos_handoff_requires_immutable_parent(self):
        from research_bridge import handoff
        parent=self.put('synthetic-bands.json',{'metadata':{'scf_converged':True},'fixture':'synthetic only'})
        from source_catalog import sha256
        dos=self.put('synthetic-dos.json',{'source':str(parent),'source_sha256':sha256(parent),'metadata':{'sampling':'uniform_mesh'},'fermi_energy_ev':0,'fixture':'synthetic only'})
        r=handoff(dos,self.p/'h1','S01','fixture','synthetic');self.assertEqual(r['state'],'converged')
        parent.write_text('{}')
        r=handoff(dos,self.p/'h2','S01','fixture','synthetic');self.assertEqual(r['state'],'failed')

    def test_api_indirect_spin_keys_and_omitted_controls(self):
        c=source_catalog(SOURCE)['api']
        for name in ('job.run','bands.run','dos.run','relax.ionic','relax.cell'):
            self.assertTrue({'spin_channels','initial_moments','hubbard'} <= set(c[name]['parameters']))
        self.assertNotIn('solver',c['job.run']['parameters']);self.assertNotIn('sparsity',c['relax.ionic']['parameters'])
        self.assertIn('sparsity',c['dos.run']['parameters'])
    def test_frozen_profile_removes_auto_and_forbidden_density_factor(self):
        p=self.put('synthetic.json',{'converged':True,'cutoff_ev':300,'density_cutoff_ev':1200,'atomic_units':'codata','projector_interpolation':'cubic','spin_channels':1,'provenance':{'sparsity':{'mode':'auto','density_cutoff_factor':4,'xc_padding':1,'auto_controls':{'level':'balanced'},'rule_ids':['fixture-only']}}})
        r=frozen_profile(p);self.assertEqual(r['sparsity'],{'mode':'custom','xc_padding':1});self.assertEqual(r['density_cutoff_ev'],1200)
        p.write_text('{"converged":false}')
        with self.assertRaises(ValueError):frozen_profile(p)
    def test_fp32_and_conflicts_fail_in_real_software(self):
        from sparse_dft.sparsity import coerce_profile,profile_environment
        from sparse_dft.errors import InputError,UnsupportedError
        with self.assertRaises(UnsupportedError):coerce_profile({'mode':'custom','search_precision':'fp32'})
        with self.assertRaises(InputError):profile_environment(coerce_profile({'mode':'custom','xc_padding':1}),{'SDFT_FFT_OVERSAMPLING':'2'})
    def test_real_profile_environment_and_parallel_validation(self):
        from sparse_dft.sparsity import coerce_profile,profile_environment
        from sparse_dft.parallel import coerce_layout
        from sparse_dft.errors import InputError
        p=coerce_profile({'mode':'custom','density_cutoff_factor':8,'xc_padding':1,'eigen_search_factor':1e-4,'block_ortho':True,'paw_compensation':'blocks','projector_space':'real-search','projector_radius_factor':1.5})
        e=profile_environment(p,{})
        self.assertEqual(e['SDFT_PROJECTOR_SPACE'],'real-search');self.assertEqual(e['SDFT_FFT_OVERSAMPLING'],'1');self.assertEqual(e['SDFT_PAW_COMPENSATION_SCOPE'],'blocks')
        with self.assertRaises(InputError):coerce_layout({'ranks':3,'kgroups':2})
    def test_prepare_writes_executable_python_calls(self):
        p=self.put('settings.json',{'paw_root':'/actual/local/paw','cutoff_ev':300,'scf_grid':[1,1,1],'solver':{'ritz_solver':'subset'}})
        v=self.put('variants.json',{'off':{'mode':'off'},'auto':{'mode':'auto'}})
        r=prepare(SOURCE/'examples/Si.POSCAR',p,v,self.p/'project');self.assertFalse(r['calculation_executed'])
        recipe=json.loads((self.p/'project/sparsity-recipe.json').read_text());self.assertEqual(recipe['steps'][1]['call'],'sparse_dft.scf:run_scf');self.assertIn('solver',recipe['steps'][1]['params']['kwargs'])
    def test_public_subpackage_python_and_tool_path(self):
        c=source_catalog(SOURCE)
        self.assertIn('sparse_dft.analysis.report:analyze_bands',c['python'])
        self.assertIn('scripts/install.py',c['tools']);self.assertIn('plot',c['cli']);self.assertIn('worker',c['cli'])
    def test_synthetic_comparison_checks_certificate_and_dimensions(self):
        data={'converged':True,'structure_hash':'synthetic','cutoff_ev':300,'density_cutoff_ev':4800,'atomic_units':'codata','projector_interpolation':'cubic','spin_channels':1,'electron_count':4,'scf_kpoints':[{'fractional':[0,0,0],'weight':1}],'hubbard_specs':[],'eigensolver_max_residual_hartree':1e-13,'hf_total_forces_ev_per_angstrom':[[0,0,0]],'hf_total_stress_ev_per_angstrom3':[[0,0,0]]*3,'total_energy_ev':-1,'provenance':{'sparsity':{'mode':'off'}},'fixture':'synthetic arithmetic only'}
        a=self.put('a.json',data);b=self.put('b.json',data)
        pts=self.put('points.json',{'points':[{'id':'off','result':str(a)},{'id':'custom','result':str(b)}]})
        r=compare(pts,self.p/'report.json');self.assertTrue(r['passed'])
        data['eigensolver_max_residual_hartree']=1e-5;b.write_text(json.dumps(data))
        with self.assertRaises(ValueError):compare(pts,self.p/'bad.json')

if __name__=='__main__':unittest.main()
