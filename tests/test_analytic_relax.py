import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sparse_dft.errors import InputError
from sparse_dft.structure import Structure
from sparse_dft.relax import ionic_relax, cell_relax
from sparse_dft.analytic_relax import _cell_deformation, _cell_gradient, _ionic_modes, _geometry_predictor_setting


class GeometryConventionTests(unittest.TestCase):
    def test_geometry_reuse_defaults_and_explicit_user_overrides(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_geometry_predictor_setting(), '1')
        for environment, expected in [({'SDFT_GEOMETRY_PREDICTOR':'0'}, '0'),
                ({'SDFT_ENABLE_PAW_CLOSURE_DIAGNOSTIC':'0'}, '0'),
                ({'SDFT_ENABLE_PAW_CLOSURE_DIAGNOSTIC':'1'}, '1'),
                ({'SDFT_ENABLE_PAW_CLOSURE_DIAGNOSTIC':'1','SDFT_GEOMETRY_PREDICTOR':'0'}, '0')]:
            with self.subTest(environment=environment), patch.dict(os.environ, environment, clear=True):
                self.assertEqual(_geometry_predictor_setting(), expected)
        for environment in [{'SDFT_ENABLE_PAW_CLOSURE_DIAGNOSTIC':'bad'},
                {'SDFT_GEOMETRY_PREDICTOR':'bad'},
                {'SDFT_ENABLE_PAW_CLOSURE_DIAGNOSTIC':'0','SDFT_GEOMETRY_PREDICTOR':'1'}]:
            with self.subTest(environment=environment), patch.dict(os.environ, environment, clear=True), self.assertRaises(InputError):
                _geometry_predictor_setting()

    def test_nonsymmetric_lattice_roundtrip_and_reciprocal_duality(self):
        a=((4.,.7,.2),(0.,3.,.4),(.1,.3,5.))
        fractional=(.17,.31,.43)
        s=Structure(['Si'],[fractional],a,True,(True,)*3)
        cart=s.cartesian_positions()[0]
        for source in (s,Structure(['Si'],[cart],a,False,(True,)*3)):
            normalized=source.normalized()
            for got,expected in zip(normalized.positions[0],fractional):
                self.assertAlmostEqual(got,expected,places=13)
            for got,expected in zip(normalized.cartesian_positions()[0],cart):
                self.assertAlmostEqual(got,expected,places=13)
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(sum(x*y for x,y in zip(a[i],s.reciprocal_lattice[j])),
                                       2*math.pi if i==j else 0.,places=13)

    def test_selective_constraints_are_direct_spans_and_are_hashed(self):
        a=((4.,1.,0.),(0.,3.,0.),(0.,0.,5.))
        s=Structure(['Si'],[(.2,.3,.4)],a,True,(True,)*3,[(True,False,False)])
        modes=_ionic_modes(s)
        self.assertEqual(len(modes),1)
        self.assertAlmostEqual(modes[0][1][1]/modes[0][1][0],.25)
        free=Structure(['Si'],s.positions,a,True,(True,)*3)
        self.assertNotEqual(s.content_hash(),free.content_hash())
        self.assertEqual(s.as_dict()['selective_dynamics'],s.selective_dynamics)

    def test_cell_gradient_matches_six_coordinate_derivatives(self):
        x=[.02,-.03,.01,.04,-.02,.03]
        f=_cell_deformation(x)
        # E(F)=1/2 sum(F-I)^2. Its spatial stress is P F^T / V.
        p=[[f[i][j]-(1. if i==j else 0.) for j in range(3)] for i in range(3)]
        volume=3.7
        stress=[[sum(p[i][k]*f[j][k] for k in range(3))/volume for j in range(3)] for i in range(3)]
        analytic=_cell_gradient(x,stress,volume)
        def energy(coords):
            fm=_cell_deformation(coords)
            return .5*sum((fm[i][j]-(1. if i==j else 0.))**2 for i in range(3) for j in range(3))
        for i,g in enumerate(analytic):
            plus=x.copy();minus=x.copy();plus[i]+=1e-6;minus[i]-=1e-6
            self.assertAlmostEqual(g,(energy(plus)-energy(minus))/2e-6,places=10)


class AnalyticRelaxationTests(unittest.TestCase):
    def structure(self):
        return Structure(['Si'],[(.05,0.,0.)],((4.,0.,0.),(0.,4.,0.),(0.,0.,4.)),
                         True,(True,)*3,[(True,False,False)])

    def quadratic(self,s,paw,workdir,**kwargs):
        # Deliberately stiff: the first trial must be rejected by the energy gate.
        x=s.cartesian_positions()[0][0]-.4
        energy=50*x*x
        result={'converged':True,'onecenter_closure_diagnostic':True,'iterations':1,
                'total_energy_ev':energy,'total_energy_valid':False,
                'hf_total_forces_ev_per_angstrom':[[-100*x,0.,0.]],
                'force_provenance':{'analytic':True,'validated':False}}
        return energy,result

    def test_native_line_search_restart_and_hash_guard(self):
        s=self.structure()
        with tempfile.TemporaryDirectory() as td,patch('sparse_dft.relax._energy',side_effect=self.quadratic) as energy:
            first=ionic_relax(s,'/paw',td,max_steps=1,max_displacement=1.,move_scale=.2,force_tolerance=1e-8)
            self.assertEqual(first['status'],'MAX_STEPS')
            self.assertGreater(first['history'][1]['line_search_trials'],1)
            second=ionic_relax(s,'/paw',td,max_steps=5,max_displacement=1.,move_scale=.2,force_tolerance=1e-8)
            self.assertEqual(second['status'],'CONVERGED')
            self.assertFalse(second['production_validated'])
            energies=[entry['energy_ev'] for entry in second['history']]
            self.assertTrue(all(b<a for a,b in zip(energies,energies[1:])))
            previous=energy.call_count
            again=ionic_relax(s,'/paw',td,max_steps=1,max_displacement=1.,move_scale=.2,force_tolerance=1e-8)
            self.assertEqual(energy.call_count,previous)
            self.assertEqual(again['status'],'CONVERGED')
            checkpoint=json.loads((Path(td)/'relax_checkpoint.json').read_text())
            Path(checkpoint['accepted_state']['scf_file']).write_text('corrupt')
            with self.assertRaisesRegex(InputError,'hash differs'):
                ionic_relax(s,'/paw',td,max_steps=1,max_displacement=1.,move_scale=.2,force_tolerance=1e-8)

    def test_incomplete_closure_is_rejected(self):
        def incomplete(*args,**kwargs):
            energy,result=self.quadratic(*args,**kwargs)
            result['onecenter_closure_diagnostic']=False
            return energy,result
        with tempfile.TemporaryDirectory() as td,patch('sparse_dft.relax._energy',side_effect=incomplete):
            with self.assertRaisesRegex(InputError,'complete PAW closure'):
                ionic_relax(self.structure(),'/paw',td)

    def test_symmetric_shear_finite_difference_has_two_components(self):
        s=self.structure()
        def sheared_energy(structure,*args,**kwargs):
            # A'=A*F^T; on this orthogonal lattice F_yz=A'_z,y/4.
            h=structure.lattice[2][1]/4
            return 2*s.volume*.03*h,{'converged':True}
        with tempfile.TemporaryDirectory() as td,patch('sparse_dft.relax._energy',side_effect=sheared_energy):
            result=cell_relax(s,'/paw',td,max_steps=1,stress_tolerance=.1,derivative_method='finite_difference')
            sigma=result['history'][0]['stress_ev_per_angstrom3']
            self.assertAlmostEqual(sigma[1][2],.03,places=12)


if __name__=='__main__': unittest.main()
