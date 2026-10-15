"""paw-convert builds an archive that the PAW reader accepts, from a synthetic POTCAR set."""
import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from sparse_dft.errors import InputError
from sparse_dft.paw import Paw54JsonArchive
from sparse_dft.paw_convert import classify_variant, convert

POTCAR = '''  PAW_PBE {title} 01Jan2000
   4.00000000000000000
 parameters from PSCTR are:
   VRHFIN ={sym}: s2p2
   LEXCH  = PE
   EATOM  =   111.1111 eV,    8.1666 Ry

   TITEL  = PAW_PBE {title} 01Jan2000
   POMASS =   11.111; ZVAL   =    4.000    synthetic
   RWIGS  =    2.222; RWIGS  =    1.176    synthetic
   ENMAX  =  222.222; ENMIN  =  166.666 eV
   EAUG   =    333.333
   Description
 synthetic body, not a real dataset
'''
PSCTR = '''   TITEL  = PAW_PBE {title} 01Jan2000
   LPAW   = T
   RPACOR =    1.5
   RWIGS  =    2.22  synthetic
   ICORE  =       2
   NE     =   100
   LCOR   =   .TRUE.
  QCUT = 0
   Description
'''


def make_set(root: Path):
    for directory, sym in (('Si', 'Si'), ('Si_GW', 'Si'), ('Ga_d', 'Ga'), ('H_AE', 'H')):
        d = root / 'potpaw_TEST' / directory
        d.mkdir(parents=True)
        (d / 'POTCAR').write_text(POTCAR.format(title=directory, sym=sym), encoding='ascii')
        if directory != 'H_AE':
            (d / 'PSCTR').write_text(PSCTR.format(title=directory), encoding='ascii')
    (root / 'potpaw_TEST' / 'README.UPDATES').write_text('synthetic\n', encoding='ascii')
    return root / 'potpaw_TEST'


class PawConvertTests(unittest.TestCase):
    def test_directory_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            src = make_set(Path(td))
            summary = convert(src, Path(td) / 'archive')
            self.assertEqual(summary['records'], 4)
            arc = Paw54JsonArchive(Path(td) / 'archive')
            rec = arc.get_record('Ga:d')
            raw = (src / 'Ga_d' / 'POTCAR').read_bytes()
            self.assertEqual(rec['files']['POTCAR']['text'].encode('ascii'), raw)
            self.assertEqual(rec['files']['POTCAR']['sha256'], hashlib.sha256(raw).hexdigest())
            self.assertEqual(rec['metadata']['potcar']['enmax_ev'], 222.222)
            self.assertEqual(rec['metadata']['psctr']['icore'], 2.0)
            self.assertTrue(arc.get_record('H:AE')['files']['PSCTR']['empty'])
            index = json.loads((Path(td) / 'archive' / 'index.json').read_text())
            self.assertEqual(index['by_element']['Si'], ['Si:GW', 'Si:standard'])

    def test_tar_source_equals_directory_source(self):
        with tempfile.TemporaryDirectory() as td:
            src = make_set(Path(td))
            tar_path = Path(td) / 'set.tar.gz'
            with tarfile.open(tar_path, 'w:gz') as tar:
                tar.add(src, arcname='potpaw_TEST')
            convert(src, Path(td) / 'a')
            convert(tar_path, Path(td) / 'b')
            for name in ('index.json', 'auxiliary_files.json'):
                self.assertEqual((Path(td) / 'a' / name).read_text(), (Path(td) / 'b' / name).read_text())

    def test_refuses_non_empty_output_and_unsafe_tar(self):
        with tempfile.TemporaryDirectory() as td:
            src = make_set(Path(td))
            out = Path(td) / 'out'; out.mkdir(); (out / 'keep.txt').write_text('x')
            with self.assertRaises(InputError):
                convert(src, out)
            bad = Path(td) / 'bad.tar'
            with tarfile.open(bad, 'w') as tar:
                info = tarfile.TarInfo('../Si/POTCAR'); data = b'x'; info.size = 1
                tar.addfile(info, io.BytesIO(data))
            with self.assertRaises(InputError):
                convert(bad, Path(td) / 'out2')

    def test_variant_families(self):
        self.assertEqual(classify_variant('Ni_pv', 'Ni')['family'], 'semicore')
        self.assertEqual(classify_variant('Ag_sv_GW', 'Ag')['family'], 'gw_semicore')
        self.assertEqual(classify_variant('C_GW_new', 'C')['family'], 'new_grid')
        self.assertEqual(classify_variant('Er_3', 'Er')['family'], 'numeric_variant')
        self.assertEqual(classify_variant('Li_AE_GW', 'Li')['family'], 'all_electron_like')


if __name__ == '__main__':
    unittest.main()
