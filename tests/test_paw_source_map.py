import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from sparse_dft.automatic_paw import resolve_source_map
from sparse_dft.errors import InputError


class PawSourceMapTests(unittest.TestCase):
    def test_archive_hashes_use_real_source_paths(self):
        archive=Path(os.environ['SDFT_PAW_ARCHIVE']).resolve()
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);path=base/'map.json'
            path.write_text(json.dumps({'schema_version':'sparse-dft-paw-source-map-1',
                'sources':{'Si':{'path':str(archive),'record_id':'Si:standard'}}}))
            choices,report,identity=resolve_source_map(SimpleNamespace(species=['Si','Si']),path,base/'prepared')
            self.assertEqual(report['framework'],'potcar_paw')
            hashes=identity.local_hashes()
            self.assertIn(str(archive/'manifest.json'),hashes)
            self.assertTrue(all(Path(p).is_absolute() and Path(p).is_file() for p in hashes))
            self.assertEqual(choices['Si']['record_id'],'Si:standard')
            self.assertFalse(report['physics_validated'])

    def test_original_sha_mismatch_rejected_before_native(self):
        archive=Path(os.environ['SDFT_PAW_ARCHIVE']).resolve()
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);path=base/'map.json'
            path.write_text(json.dumps({'schema_version':'sparse-dft-paw-source-map-1',
                'sources':{'Si':{'path':str(archive),'expected_sha256':'0'*64}}}))
            with self.assertRaises(InputError):resolve_source_map(SimpleNamespace(species=['Si']),path,base/'prepared')
            self.assertFalse(list(base.rglob('*.process.json')))

    def test_exact_species_and_recursion_are_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);path=base/'map.json'
            for sources in [{},{'Si':{'path':'map.json'}},{'Si':{'path':'missing'},'O':{'path':'missing'}}]:
                path.write_text(json.dumps({'schema_version':'sparse-dft-paw-source-map-1','sources':sources}))
                with self.assertRaises(InputError):resolve_source_map(SimpleNamespace(species=['Si']),path,base/'prepared')
