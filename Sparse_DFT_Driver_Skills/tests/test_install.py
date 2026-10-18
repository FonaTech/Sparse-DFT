import importlib.util
import tempfile
from pathlib import Path
import unittest

S=Path(__file__).resolve().parents[1]/'skills/sparse-dft-driver/scripts/install_skill.py'
spec=importlib.util.spec_from_file_location('installer',S);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

class Installer(unittest.TestCase):
    def test_migration_preserves_old_and_exact_new_copy(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);old=p/'skills/old-skill';old.mkdir(parents=True);(old/'SKILL.md').write_text('old real skill fixture')
            r=mod.install(p/'skills','old-skill',p/'backups')
            self.assertFalse(old.exists());self.assertEqual((p/'backups/old-skill/SKILL.md').read_text(),'old real skill fixture')
            self.assertEqual(mod.identity(Path(r['skill'])),mod.identity(mod.SOURCE))
            with self.assertRaises(ValueError):mod.install(p/'skills')
    def test_missing_external_backup_leaves_old_untouched(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);old=p/'skills/old-skill';old.mkdir(parents=True);(old/'SKILL.md').write_text('fixture')
            with self.assertRaises(ValueError):mod.install(p/'skills','old-skill')
            self.assertTrue(old.exists());self.assertFalse((p/'skills/sparse-dft-driver').exists())

if __name__=='__main__':unittest.main()
