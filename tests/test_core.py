import json, tempfile, unittest
from pathlib import Path
from sparse_dft.structure import read_structure
from sparse_dft.kpoints import recommend_kpoints
from sparse_dft.paw import Paw54JsonArchive
from sparse_dft.resources import project_paw_archive

ROOT=project_paw_archive()
class CoreTests(unittest.TestCase):
  def test_structure_formats(self):
    ss=[read_structure('examples/Si.POSCAR'),read_structure('examples/Si.xyz'),read_structure('examples/Si.cif')]
    self.assertTrue(all(s.natoms==2 for s in ss))
    self.assertTrue(ss[0].volume and ss[1].volume and ss[2].volume)
    self.assertTrue(ss[0].content_hash()!=ss[1].content_hash() or ss[0].source_format!=ss[1].source_format)
    for s in ss:
      n=s.normalized()
      for frac, cart in zip(n.positions, n.cartesian_positions()):
        recovered=tuple(sum(s.lattice[i][j]*frac[i] for i in range(3)) for j in range(3))
        self.assertTrue(max(abs(a-b) for a,b in zip(recovered,cart))<1e-10)

  def test_kpoints(self):
    s=read_structure('examples/Si.POSCAR')
    k=recommend_kpoints(s)
    self.assertTrue(all(x>=1 for x in k.grid) and len(k.convergence_scan)==3)

  def test_archive(self):
    a=Paw54JsonArchive(ROOT); i=a.inventory()
    self.assertEqual(i['records'],327); self.assertEqual(len(i['elements']),97); self.assertEqual(i['empty_psctr'],['H:AE'])
    with tempfile.TemporaryDirectory() as d:
        ds=a.restore_dataset('Si:standard',Path(d)/'Si')
        self.assertEqual(ds.parsed_header['ZVAL'],4.0)
        self.assertEqual(Path(ds.potcar_path).stat().st_size,a.get_record('Si:standard')['files']['POTCAR']['size_bytes'])
