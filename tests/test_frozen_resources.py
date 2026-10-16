"""Code paths a PyInstaller bundle takes: no .py sources on disk, data and
native helpers located through sparse_dft.resources."""
import ast
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sparse_dft import paw, prepared_paw_cache

ROOT = Path(__file__).resolve().parents[1]


class FrozenResourceTests(unittest.TestCase):
    def test_parser_identity_without_python_sources(self):
        self.assertEqual(prepared_paw_cache._parser_identity(), prepared_paw_cache.sha(paw.__file__))
        with tempfile.TemporaryDirectory() as td:
            exe = Path(td) / 'sparse-dft-worker'
            exe.write_bytes(b'frozen launcher')
            with mock.patch.object(paw, '__file__', str(Path(td) / 'absent' / 'paw.py')), \
                    mock.patch.object(sys, 'executable', str(exe)):
                identity = prepared_paw_cache._parser_identity()
            self.assertEqual(identity, 'frozen:' + prepared_paw_cache.sha(exe))

    def test_bundle_ships_every_package_json(self):
        # The spec collects sparse_dft/**/*.json; each file the code reads must be one of them.
        spec = (ROOT / 'packaging' / 'sparse_dft_gui.spec').read_text(encoding='utf-8')
        self.assertIn("(ROOT / 'sparse_dft').rglob('*.json')", spec)
        for name in ('sparsity_rules.json', 'sparsity_measured.json', 'data/elements.json'):
            self.assertTrue((ROOT / 'sparse_dft' / name).is_file(), name)

    def test_backend_uses_the_resource_locator(self):
        tree = ast.parse((ROOT / 'sparse_dft' / 'backend.py').read_text(encoding='utf-8'))
        literals = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
        self.assertFalse([s for s in literals if s.startswith('build/')], 'backend.py hard-codes build/ paths')
        from sparse_dft.backend import _native
        self.assertTrue(str(_native('sparse_dft_no_such_tool')).endswith('sparse_dft_no_such_tool'))

    def test_app_manifest_stays_outside_the_signed_bundle(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('make_manifest', ROOT / 'packaging' / 'make_manifest.py')
        mm = importlib.util.module_from_spec(spec); spec.loader.exec_module(mm)
        with tempfile.TemporaryDirectory() as td:
            app = Path(td) / 'Sparse-DFT.app'
            (app / 'Contents' / 'MacOS').mkdir(parents=True)
            (app / 'Contents' / 'MacOS' / 'launcher').write_bytes(b'x')
            path = mm.write(app, 'macos-arm64')
            self.assertEqual(path.parent, Path(td))
            self.assertEqual(sorted(p.name for p in app.iterdir()), ['Contents'])
            self.assertEqual(mm.check(app), [])
            (app / 'Contents' / 'MacOS' / 'launcher').write_bytes(b'y')
            self.assertEqual(mm.check(app), ['changed Contents/MacOS/launcher'])
            folder = Path(td) / 'Sparse-DFT'; folder.mkdir()
            self.assertEqual(mm.manifest_path(folder), folder / mm.MANIFEST)


if __name__ == '__main__':
    unittest.main()
