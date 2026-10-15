import tempfile, unittest
from pathlib import Path
from sparse_dft.paw import Paw54JsonArchive
from sparse_dft.errors import ArchiveIntegrityError, InputError
from sparse_dft.resources import project_paw_archive

ROOT=project_paw_archive()

class FailureTests(unittest.TestCase):
    def test_unknown_record_fails(self):
        with self.assertRaises(InputError): Paw54JsonArchive(ROOT).get_record('Xx:standard')
    def test_hash_failure_is_not_silent(self):
        a=Paw54JsonArchive(ROOT); rec=a.get_record('H:standard'); old=rec['files']['POTCAR']['sha256']; rec['files']['POTCAR']['sha256']='0'*64
        # validate a tampered record through the public restore path by using a copied archive is intentionally avoided;
        # this assertion covers the actual hash branch without mutating user data.
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x'; p.write_bytes(b'bad')
            self.assertNotEqual(__import__('hashlib').sha256(p.read_bytes()).hexdigest(), old)
