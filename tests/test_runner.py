"""sparse_dft.runner: job files, worker status, UTF-8 streaming, process-tree cancel, IterationTail."""
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from sparse_dft import runner

REPO = Path(__file__).resolve().parents[1]
GBK_ENV = {'PYTHONIOENCODING': 'gbk', 'LANG': 'zh_CN.GBK', 'LC_ALL': 'zh_CN.GBK'}
CHINESE = '你好 世界 · 繁體中文 · 日本語 Ünï'


def _wait_for(predicate, timeout=20.0, interval=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A zombie still answers kill(0); ask ps on POSIX.
    if os.name != 'nt':
        state = subprocess.run(['ps', '-o', 'stat=', '-p', str(pid)], capture_output=True, text=True,
                               encoding='utf-8', errors='replace').stdout.strip()
        return bool(state) and not state.startswith('Z')
    return True


class TempDirMixin:
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix='sparse_dft 测试 目录 Ünï ')
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()


class JobValidationTests(TempDirMixin, unittest.TestCase):
    def test_make_and_load_round_trip(self):
        job = runner.make_job(['inspect', 'examples/Si.POSCAR'], env={'A': '1', 'B': None}, cwd=REPO, label='标签')
        path = runner.write_job(self.tmp / '任务.json', job)
        loaded = runner.load_job(path)
        self.assertEqual(loaded['argv'], ['inspect', 'examples/Si.POSCAR'])
        self.assertEqual(loaded['env'], {'A': '1', 'B': None})
        self.assertEqual(loaded['label'], '标签')
        self.assertTrue(Path(loaded['cwd']).is_absolute())
        self.assertIn('标签', path.read_text(encoding='utf-8'))

    def test_invalid_jobs_are_rejected(self):
        bad = [None, [], {'schema_version': 'other', 'argv': ['x']},
               {'schema_version': runner.JOB_SCHEMA, 'kind': 'shell', 'argv': ['x']},
               {'schema_version': runner.JOB_SCHEMA, 'kind': 'cli', 'argv': []},
               {'schema_version': runner.JOB_SCHEMA, 'kind': 'cli', 'argv': 'scf a b'},
               {'schema_version': runner.JOB_SCHEMA, 'kind': 'cli', 'argv': ['scf', 3]},
               {'schema_version': runner.JOB_SCHEMA, 'kind': 'api'},
               {'schema_version': runner.JOB_SCHEMA, 'kind': 'api', 'method': 'x', 'params': []},
               {'schema_version': runner.JOB_SCHEMA, 'argv': ['x'], 'env': {'A': 1}},
               {'schema_version': runner.JOB_SCHEMA, 'argv': ['x'], 'cwd': 5}]
        for payload in bad:
            with self.subTest(payload=payload), self.assertRaises(runner.JobError):
                runner.validate_job(payload)

    def test_load_job_reports_unreadable_files(self):
        path = self.tmp / 'broken.json'
        path.write_text('{not json', encoding='utf-8')
        with self.assertRaises(runner.JobError):
            runner.load_job(path)
        with self.assertRaises(runner.JobError):
            runner.load_job(self.tmp / 'missing.json')

    def test_secret_options_are_masked(self):
        self.assertEqual(runner.mask_argv(['mp-fetch', '--api-key', 'SECRET', '--count', '2']),
                         ['mp-fetch', '--api-key', '***', '--count', '2'])
        self.assertEqual(runner.mask_argv(['mp-fetch', '--api-key=SECRET']), ['mp-fetch', '--api-key=***'])

    def test_output_dir_resolution(self):
        job = runner.make_job(['scf', 'a.POSCAR', '--paw-root', 'p', '--out', 'out dir 输出'], cwd=self.tmp)
        self.assertEqual(Path(runner.output_dir_for(job)), self.tmp.resolve() / 'out dir 输出')
        job = runner.make_job(['scf', 'a.POSCAR', '--paw-root', 'p'], cwd=self.tmp)
        self.assertEqual(Path(runner.output_dir_for(job)), self.tmp.resolve() / 'job')  # argparse default
        job = runner.make_job(['doctor'], cwd=self.tmp)
        self.assertIsNone(runner.output_dir_for(job))

    def test_atomic_write_replaces_whole_document(self):
        path = self.tmp / 'status.json'
        for i in range(20):
            runner.atomic_write_json(path, {'i': i, 'text': CHINESE})
        self.assertEqual(json.loads(path.read_text(encoding='utf-8')), {'i': 19, 'text': CHINESE})
        self.assertEqual([p.name for p in self.tmp.iterdir()], ['status.json'])

    def test_worker_command_frozen_and_source(self):
        self.assertEqual(runner.worker_command('j.json')[1:4], ['-m', 'sparse_dft.cli', 'worker'])
        frozen = getattr(sys, 'frozen', None)
        sys.frozen = True
        try:
            self.assertEqual(runner.worker_command('j.json', 's.json'),
                             [sys.executable, '--worker', 'j.json', '--status', 's.json'])
            from unittest.mock import patch
            worker = self.tmp / ('sparse-dft-worker.exe' if os.name == 'nt' else 'sparse-dft-worker')
            worker.write_text('', encoding='utf-8')
            with patch.object(sys, 'executable', str(self.tmp / 'sparse-dft-gui')):
                self.assertEqual(runner.worker_command('j.json'), [str(worker.resolve()), '--worker', 'j.json'])
        finally:
            if frozen is None:
                del sys.frozen
            else:
                sys.frozen = frozen


class WorkerMainTests(TempDirMixin, unittest.TestCase):
    """``sparse-dft worker`` exit codes and status documents, run as a real subprocess."""

    def _run_worker(self, job, env=None):
        path = runner.write_job(self.tmp / 'job.json', job)
        status = self.tmp / 'status 状态.json'
        environment = dict(os.environ, PYTHONPATH=str(REPO), **(env or {}))
        proc = subprocess.run([sys.executable, '-m', 'sparse_dft.cli', 'worker', str(path), '--status', str(status)],
                              capture_output=True, env=environment, cwd=str(self.tmp), timeout=120)
        return proc, json.loads(status.read_text(encoding='utf-8'))

    def test_success_exit_code_and_status(self):
        job = runner.make_job(['inspect', str(REPO / 'examples' / 'Si.POSCAR')], label='检查 Si')
        proc, status = self._run_worker(job)
        self.assertEqual(proc.returncode, 0, proc.stderr.decode('utf-8', 'replace'))
        self.assertEqual(status['state'], 'finished')
        self.assertEqual(status['returncode'], 0)
        self.assertEqual(status['label'], '检查 Si')
        self.assertIsInstance(status['pid'], int)
        self.assertTrue(status['start_time'] and status['end_time'])
        self.assertIn('"species"', proc.stdout.decode('utf-8'))

    def test_failure_exit_code_and_status(self):
        job = runner.make_job(['inspect', str(self.tmp / 'missing.POSCAR')])
        proc, status = self._run_worker(job)
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(status['state'], 'failed')
        self.assertEqual(status['returncode'], 2)

    def test_exec_job_propagates_return_code(self):
        job = runner.make_job([sys.executable, '-c', 'raise SystemExit(7)'], kind='exec')
        proc, status = self._run_worker(job)
        self.assertEqual((proc.returncode, status['state'], status['returncode']), (7, 'failed', 7))

    def test_api_job(self):
        job = runner.make_job(kind='api', method='structure.validate',
                              params={'path': str(REPO / 'examples' / 'Si.POSCAR')})
        proc, status = self._run_worker(job)
        self.assertEqual(proc.returncode, 0)
        self.assertEqual(json.loads(proc.stdout.decode('utf-8'))['status'], 'PASS')
        self.assertEqual(status['state'], 'finished')

    def test_invalid_job_file_fails_cleanly(self):
        path = self.tmp / 'job.json'
        path.write_text('{"schema_version": "nope"}', encoding='utf-8')
        status = self.tmp / 'status.json'
        proc = subprocess.run([sys.executable, '-m', 'sparse_dft.cli', 'worker', str(path), '--status', str(status)],
                              capture_output=True, env=dict(os.environ, PYTHONPATH=str(REPO)), timeout=60)
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(json.loads(status.read_text(encoding='utf-8'))['state'], 'failed')

    def test_frozen_entry_and_gui_entry_dispatch_to_worker(self):
        # ``sparse-dft-gui --worker JOB.json`` must run the worker without starting Qt.
        job = runner.make_job([sys.executable, '-c', "import sys; print('PyQt6' in sys.modules)"], kind='exec')
        path = runner.write_job(self.tmp / 'job.json', job)
        env = dict(os.environ, PYTHONPATH=str(REPO))
        for command in ([sys.executable, str(REPO / 'packaging' / 'sparse_dft_gui_entry.py'), '--worker', str(path)],
                        [sys.executable, '-m', 'sparse_dft_gui', '--worker', str(path)]):
            with self.subTest(command=command[1]):
                proc = subprocess.run(command, capture_output=True, env=env, timeout=60)
                self.assertEqual(proc.returncode, 0, proc.stderr.decode('utf-8', 'replace'))
                self.assertEqual(runner.read_status(runner.default_status_path(path))['state'], 'finished')

    def test_main_in_process_returns_code(self):
        job = runner.make_job([sys.executable, '-c', 'pass'], kind='exec')
        path = runner.write_job(self.tmp / 'job.json', job)
        saved = (sys.stdout, sys.stderr, os.getcwd(), dict(os.environ))
        try:
            rc = runner.main([str(path)])
        finally:
            sys.stdout, sys.stderr = saved[0], saved[1]
            os.chdir(saved[2]); os.environ.clear(); os.environ.update(saved[3])
        self.assertEqual(rc, 0)
        self.assertEqual(runner.read_status(runner.default_status_path(path))['state'], 'finished')


class StreamingTests(TempDirMixin, unittest.TestCase):
    def _collect(self, job, env=None, timeout=60):
        out, err = [], []
        process = runner.JobProcess(job, self.tmp / 'work 工作', on_stdout=out.append, on_stderr=err.append, env=env)
        process.start()
        rc = process.wait(timeout=timeout)
        return rc, out, err, process

    def test_chinese_output_survives_gbk_console_environment(self):
        # The worker prints in-process (api job echoes the Chinese path) and a Python
        # grandchild prints Chinese; both must arrive intact although the parent
        # environment advertises a GBK console.
        poscar = self.tmp / '硅 晶体 Si.POSCAR'
        poscar.write_text((REPO / 'examples' / 'Si.POSCAR').read_text(encoding='utf-8'), encoding='utf-8')
        job = runner.make_job(kind='api', method='structure.parse', params={'path': str(poscar)}, label='解析 结构')
        rc, out, err, _ = self._collect(job, env=GBK_ENV)
        self.assertEqual(rc, 0, err)
        self.assertTrue(any('硅 晶体 Si.POSCAR' in line for line in out), out)
        self.assertTrue(any('解析 结构' in line for line in err), err)
        job = runner.make_job([sys.executable, '-c', f'print({CHINESE!r}); import sys; print("错误 " + {CHINESE!r}, file=sys.stderr)'],
                              kind='exec')
        rc, out, err, _ = self._collect(job, env=GBK_ENV)
        self.assertEqual(rc, 0, err)
        self.assertIn(CHINESE, out)
        self.assertIn('错误 ' + CHINESE, err)

    def test_invalid_utf8_is_replaced_not_fatal(self):
        job = runner.make_job([sys.executable, '-c', r'import sys; sys.stdout.buffer.write(b"ok \xff\xfe end\n")'],
                              kind='exec')
        rc, out, _err, _ = self._collect(job)
        self.assertEqual(rc, 0)
        self.assertTrue(any(line.startswith('ok ') and line.endswith(' end') and '�' in line for line in out), out)

    def test_status_reflects_failure(self):
        job = runner.make_job([sys.executable, '-c', 'raise SystemExit(5)'], kind='exec')
        rc, _out, _err, process = self._collect(job)
        self.assertEqual(rc, 5)
        self.assertEqual(process.status()['state'], 'failed')


GRANDCHILD = ("import subprocess, sys\n"
              "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])\n"
              "print('grandchild', p.pid, flush=True)\n"
              "p.wait()\n")


class CancelTests(TempDirMixin, unittest.TestCase):
    def _start(self, job):
        lines = []
        process = runner.JobProcess(job, self.tmp / 'cancel', on_stdout=lines.append, on_stderr=lines.append, grace=2.0)
        process.start()
        self.assertTrue(_wait_for(lambda: any(l.startswith('grandchild') for l in lines), 30), lines)
        pid = int(next(l for l in lines if l.startswith('grandchild')).split()[1])
        return process, pid, lines

    def test_cancel_kills_process_tree(self):
        job = runner.make_job([sys.executable, '-c', GRANDCHILD], kind='exec')
        process, grandchild, _ = self._start(job)
        started = time.monotonic()
        process.cancel()
        rc = process.wait(timeout=15)
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 8.0)
        self.assertNotEqual(rc, 0)
        self.assertTrue(_wait_for(lambda: not _alive(grandchild), 8), f'grandchild {grandchild} survived')
        self.assertEqual(process.status()['state'], 'cancelled')

    def test_cancel_cli_job_blocked_in_subprocess_run(self):
        # Same shape as scf.py: the worker thread waits in subprocess.run on the native solver.
        code = ("import subprocess, sys\n"
                "print('grandchild', 0, flush=True)\n"
                "subprocess.run([sys.executable, '-c', 'import time; time.sleep(300)'])\n")
        job = runner.make_job([sys.executable, '-c', code], kind='exec')
        process, _pid, lines = self._start(job)
        process.cancel(wait=True)
        self.assertFalse(process.running())
        self.assertEqual(process.status()['state'], 'cancelled')

    @unittest.skipIf(os.name == 'nt', 'cooperative cancel file path is exercised by the GUI on Windows')
    def test_cancel_request_file_stops_worker(self):
        job = runner.make_job([sys.executable, '-c', GRANDCHILD], kind='exec')
        process, grandchild, _ = self._start(job)
        runner.request_cancel(process.status_path)
        self.assertEqual(process.wait(timeout=15), runner.CANCELLED_RETURNCODE)
        self.assertEqual(process.status()['state'], 'cancelled')
        self.assertTrue(_wait_for(lambda: not _alive(grandchild), 8))


class IterationTailTests(TempDirMixin, unittest.TestCase):
    def test_missing_file_then_incremental_records(self):
        path = self.tmp / '迭代 scf.json.iterations.jsonl'
        tail = runner.IterationTail(path)
        self.assertEqual(tail.read_new(), [])
        with open(path, 'w', encoding='utf-8') as stream:
            stream.write('{"iteration": 1, "density_residual": 0.1}\n{"iteration": 2, "dens')
            stream.flush()
            self.assertEqual([r['iteration'] for r in tail.read_new()], [1])
            self.assertEqual(tail.read_new(), [])  # partial line is held back
            stream.write('ity_residual": 0.01}\n')
            stream.flush()
            self.assertEqual(tail.read_new(), [{'iteration': 2, 'density_residual': 0.01}])
            stream.write('not json\n{"iteration": 3}\n')
        self.assertEqual([r['iteration'] for r in tail.read_new()], [3])
        self.assertEqual(tail.errors, 1)
        self.assertEqual(tail.count, 3)

    def test_truncation_and_replacement_restart(self):
        path = self.tmp / 'log.jsonl'
        path.write_text('{"iteration": 1}\n{"iteration": 2}\n', encoding='utf-8')
        tail = runner.IterationTail(path)
        self.assertEqual(len(tail.read_new()), 2)
        path.write_text('{"iteration": 1}\n', encoding='utf-8')  # truncated in place
        self.assertEqual(tail.read_new(), [{'iteration': 1}])
        self.assertEqual(tail.resets, 1)
        replacement = self.tmp / 'new.jsonl'
        replacement.write_text('{"iteration": 10}\n{"iteration": 11}\n{"iteration": 12}\n', encoding='utf-8')
        os.replace(replacement, path)  # rotated: new inode, larger file
        self.assertEqual([r['iteration'] for r in tail.read_new()], [10, 11, 12])
        self.assertEqual(tail.resets, 2)
        path.unlink()
        self.assertEqual(tail.read_new(), [])

    def test_final_read_accepts_unterminated_last_line(self):
        path = self.tmp / 'log.jsonl'
        path.write_text('{"iteration": 1}\n{"iteration": 2}', encoding='utf-8')
        tail = runner.IterationTail(path)
        self.assertEqual(len(tail.read_new()), 1)
        self.assertEqual(tail.read_new(final=True), [{'iteration': 2}])

    def test_latest_iteration_log_prefers_newest(self):
        first = self.tmp / 'step_0001' / 'scf.json.iterations.jsonl'
        second = self.tmp / 'step_0002' / 'scf.json.iterations.jsonl'
        for path in (first, second):
            path.parent.mkdir(parents=True)
            path.write_text('{"iteration": 1}\n', encoding='utf-8')
        os.utime(first, (time.time() - 100, time.time() - 100))
        self.assertEqual(runner.latest_iteration_log(self.tmp), second)
        self.assertIsNone(runner.latest_iteration_log(self.tmp / 'missing'))


if __name__ == '__main__':
    unittest.main()
