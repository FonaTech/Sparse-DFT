import hashlib
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
from sparse_dft.scf import PreparedPawSession
from sparse_dft.errors import InputError


class PreparedPawSessionTest(unittest.TestCase):
    def test_mutation_and_policy_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root/'archive'; archive.mkdir()
            (archive/'manifest.json').write_text('original')
            cache = root/'cache'; cache.mkdir()
            for name in ('POTCAR','PSCTR','numeric.bin','numeric.json','metadata.json'):
                (cache/name).write_text(name)
            structure = types.SimpleNamespace(species=['Si','Si'])
            provider = types.SimpleNamespace(local_hashes=lambda: {
                'manifest.json': hashlib.sha256((archive/'manifest.json').read_bytes()).hexdigest()})
            choices = {'Si': {'cache': str(cache/'numeric.bin'), 'zval': 4}}
            with patch('sparse_dft.scf._resolve_caches', return_value=(choices, [], provider)) as resolver:
                session = PreparedPawSession(structure, archive, root/'job')
                result, _, _ = session.resolve(structure, archive, 'standard')
                result['Si']['zval'] = 99
                self.assertEqual(session.resolve(structure, archive, 'standard')[0]['Si']['zval'], 4)
                self.assertEqual(resolver.call_count, 1)
                with self.assertRaises(InputError):
                    session.resolve(types.SimpleNamespace(species=['O']), archive, 'standard')
                with self.assertRaises(InputError):
                    session.resolve(structure, archive, 'semicore')
                (cache/'numeric.bin').write_text('changed')
                with self.assertRaises(InputError): session.resolve(structure, archive, 'standard')
                (cache/'numeric.bin').write_text('numeric.bin')
                (archive/'manifest.json').write_text('changed')
                with self.assertRaises(InputError): session.resolve(structure, archive, 'standard')


if __name__ == '__main__':
    unittest.main()
