"""sparse_dft.resources: executable lookup order, platform handling, frozen layout, user dirs."""
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sparse_dft import resources
from sparse_dft.errors import InputError

ENV_KEYS = ('SDFT_SCF_EXE', 'SDFT_RELAX_STEP_EXE', 'SDFT_FOO_TOOL_EXE', 'SDFT_USER_DATA_DIR', 'SDFT_PAW_ROOT',
            'APPDATA', 'XDG_DATA_HOME')


def _touch_exe(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('#!/bin/sh\nexit 0\n', encoding='utf-8')
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


class ResourceTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix='sparse_dft res 资源 ')
        self.root = Path(self._tmp.name)
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        clean = {k: v for k, v in os.environ.items() if k not in ENV_KEYS}
        clean['PATH'] = str(self.root / 'empty-path')
        self._env = patch.dict(os.environ, clean, clear=True)
        self._env.start()
        self._repo = patch.object(resources, '_REPO_ROOT', self.repo)
        self._repo.start()

    def tearDown(self):
        self._repo.stop()
        self._env.stop()
        self._tmp.cleanup()

    def _platform(self, name, machine='x86_64'):
        return patch.multiple('sparse_dft.resources', sys=_FakeSys(name), platform=_FakePlatform(machine))


class _FakeSys:
    """Proxy of ``sys`` with a different ``platform`` (and optional frozen attributes)."""

    def __init__(self, platform, **attrs):
        self.platform = platform
        for key, value in attrs.items():
            setattr(self, key, value)

    def __getattr__(self, name):
        return getattr(sys, name)


class _FakePlatform:
    def __init__(self, machine):
        self._machine = machine

    def machine(self):
        return self._machine


class FindExecutableTests(ResourceTestCase):
    def test_missing_lists_searched_locations(self):
        with self.assertRaises(InputError) as caught:
            resources.find_executable('sdft_scf')
        searched = caught.exception.details['searched']
        self.assertTrue(any('build' in s for s in searched))
        self.assertTrue(any(s.startswith('PATH') for s in searched))
        self.assertIn('SDFT_SCF_EXE', caught.exception.message)

    def test_order_build_release_debug_then_path(self):
        debug = _touch_exe(self.repo / 'build' / 'Debug' / 'sdft_scf')
        self.assertEqual(resources.find_executable('sdft_scf'), debug)
        rel = _touch_exe(self.repo / 'build' / 'RelWithDebInfo' / 'sdft_scf')
        self.assertEqual(resources.find_executable('sdft_scf'), rel)
        release = _touch_exe(self.repo / 'build' / 'Release' / 'sdft_scf')
        self.assertEqual(resources.find_executable('sdft_scf'), release)
        plain = _touch_exe(self.repo / 'build' / 'sdft_scf')
        self.assertEqual(resources.find_executable('sdft_scf'), plain)
        bundled = _touch_exe(self.repo / 'bin' / resources.platform_tag() / 'sdft_scf')
        self.assertEqual(resources.find_executable('sdft_scf'), bundled)

    def test_path_is_last_resort(self):
        on_path = _touch_exe(self.root / 'path dir' / 'sdft_scf')
        with patch.dict(os.environ, {'PATH': str(on_path.parent)}):
            self.assertEqual(Path(resources.find_executable('sdft_scf')), on_path)
            local = _touch_exe(self.repo / 'build' / 'sdft_scf')
            self.assertEqual(resources.find_executable('sdft_scf'), local)

    def test_env_override_wins_and_must_exist(self):
        _touch_exe(self.repo / 'build' / 'sdft_scf')
        custom = _touch_exe(self.root / 'custom 自定义' / 'my_scf')
        with patch.dict(os.environ, {'SDFT_SCF_EXE': str(custom)}):
            self.assertEqual(resources.find_executable('sdft_scf'), custom)
        with patch.dict(os.environ, {'SDFT_SCF_EXE': str(self.root / 'nope')}), self.assertRaises(InputError):
            resources.find_executable('sdft_scf')

    def test_relax_step_override_and_scf_sibling(self):
        scf = _touch_exe(self.root / 'tools' / 'sdft_scf')
        sibling = _touch_exe(self.root / 'tools' / 'sdft_relax_step')
        _touch_exe(self.repo / 'build' / 'sdft_relax_step')
        with patch.dict(os.environ, {'SDFT_SCF_EXE': str(scf)}):
            self.assertEqual(resources.find_executable('sdft_relax_step'), sibling)
        explicit = _touch_exe(self.root / 'other' / 'relax')
        with patch.dict(os.environ, {'SDFT_SCF_EXE': str(scf), 'SDFT_RELAX_STEP_EXE': str(explicit)}):
            self.assertEqual(resources.find_executable('sdft_relax_step'), explicit)

    def test_generic_override_name(self):
        tool = _touch_exe(self.root / 'foo')
        with patch.dict(os.environ, {'SDFT_FOO_TOOL_EXE': str(tool)}):
            self.assertEqual(resources.find_executable('foo-tool'), tool)

    def test_windows_appends_exe(self):
        with self._platform('win32', 'AMD64'):
            self.assertEqual(resources.platform_tag(), 'windows-x86_64')
            self.assertEqual(resources.executable_name('sdft_scf'), 'sdft_scf.exe')
            self.assertEqual(resources.executable_name('sdft_scf.exe'), 'sdft_scf.exe')
            _touch_exe(self.repo / 'build' / 'Release' / 'sdft_scf')  # POSIX name must be ignored
            exe = _touch_exe(self.repo / 'build' / 'Release' / 'sdft_scf.exe')
            self.assertEqual(resources.find_executable('sdft_scf'), exe)
            self.assertEqual(resources.find_executable('sdft_scf.exe'), exe)
            bare = self.root / 'bin' / 'sdft_scf'
            exe_override = _touch_exe(bare.with_suffix('.exe'))
            with patch.dict(os.environ, {'SDFT_SCF_EXE': str(bare)}):
                self.assertEqual(resources.find_executable('sdft_scf'), exe_override)
            relax = _touch_exe(self.repo / 'bin' / 'windows-x86_64' / 'sdft_relax_step.exe')
            self.assertEqual(resources.find_executable('sdft_relax_step'), relax)

    def test_platform_tags(self):
        for platform, machine, tag in (('darwin', 'arm64', 'macos-arm64'), ('darwin', 'x86_64', 'macos-x86_64'),
                                       ('linux', 'x86_64', 'linux-x86_64'), ('linux', 'aarch64', 'linux-arm64'),
                                       ('win32', 'ARM64', 'windows-arm64')):
            with self.subTest(platform=platform, machine=machine), self._platform(platform, machine):
                self.assertEqual(resources.platform_tag(), tag)


class FrozenLayoutTests(ResourceTestCase):
    def test_meipass_resolution(self):
        bundle = self.root / '_MEI 包'
        exe = _touch_exe(bundle / 'bin' / 'macos-arm64' / 'sdft_scf')
        _touch_exe(self.repo / 'build' / 'sdft_scf')
        fake = _FakeSys('darwin', frozen=True, _MEIPASS=str(bundle), executable=str(self.root / 'app' / 'sparse-dft-gui'))
        with patch.multiple('sparse_dft.resources', sys=fake, platform=_FakePlatform('arm64')):
            self.assertTrue(resources.is_frozen())
            self.assertEqual(resources.resource_root(), bundle)
            self.assertEqual(resources.find_executable('sdft_scf'), exe)
            self.assertEqual(resources.fonts_dir(), bundle / 'data' / 'fonts')
            self.assertEqual(resources.translations_dir(), bundle / 'sparse_dft_gui' / 'i18n')
            self.assertEqual(resources.paw_search_paths()[1], bundle / 'data' / 'paw')

    def test_frozen_without_meipass_uses_executable_dir(self):
        fake = _FakeSys('linux', frozen=True, executable=str(self.root / 'app' / 'sparse-dft-gui'))
        with patch.multiple('sparse_dft.resources', sys=fake, platform=_FakePlatform('x86_64')):
            self.assertEqual(resources.resource_root(), (self.root / 'app').resolve())
            self.assertEqual(resources.platform_tag(), 'linux-x86_64')

    def test_source_checkout_is_not_frozen(self):
        self.assertFalse(resources.is_frozen())
        self.assertEqual(resources.resource_root(), self.repo)


class UserDirTests(ResourceTestCase):
    def test_windows_appdata(self):
        with self._platform('win32'), patch.dict(os.environ, {'APPDATA': r'C:\Users\测试\AppData\Roaming'}):
            self.assertEqual(resources.user_data_dir(), Path(r'C:\Users\测试\AppData\Roaming') / 'Sparse-DFT')

    def test_macos_application_support(self):
        with self._platform('darwin', 'arm64'), patch.dict(os.environ, {'HOME': str(self.root)}):
            self.assertEqual(resources.user_data_dir(), self.root / 'Library' / 'Application Support' / 'Sparse-DFT')
            self.assertEqual(resources.platform_tag(), 'macos-arm64')

    def test_linux_xdg_and_default(self):
        with self._platform('linux'), patch.dict(os.environ, {'HOME': str(self.root)}):
            self.assertEqual(resources.user_data_dir(), self.root / '.local' / 'share' / 'sparse_dft')
            with patch.dict(os.environ, {'XDG_DATA_HOME': str(self.root / 'xdg')}):
                self.assertEqual(resources.user_data_dir(), self.root / 'xdg' / 'sparse_dft')

    def test_override_and_paw_search_order(self):
        with patch.dict(os.environ, {'SDFT_USER_DATA_DIR': str(self.root / 'ud'), 'SDFT_PAW_ROOT': str(self.root / 'paw')}):
            self.assertEqual(resources.paw_search_paths(),
                             [self.root / 'ud' / 'paw', self.repo / 'data' / 'paw', self.root / 'paw'])


class AnalyticRelaxLookupTests(ResourceTestCase):
    def test_optimizer_resolved_through_find_executable(self):
        from sparse_dft import analytic_relax
        exe = _touch_exe(self.root / 'opt' / 'sdft_relax_step')
        with patch.dict(os.environ, {'SDFT_RELAX_STEP_EXE': str(exe)}):
            self.assertEqual(analytic_relax._optimizer_executable(), exe)
        with self.assertRaises(InputError) as caught:
            analytic_relax._optimizer_executable()
        self.assertIn('native L-BFGS executable is missing', caught.exception.message)


class ProjectArchiveTests(unittest.TestCase):
    def test_explicit_override_must_be_valid(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {'SDFT_PAW_ARCHIVE': td}):
            with self.assertRaises(FileNotFoundError):
                resources.project_paw_archive()


if __name__ == '__main__':
    unittest.main()
