"""Job runner shared by the GUI, batch scripts and ``sparse-dft worker``.

A job is a JSON document (schema ``sparse-dft-job-1``)::

    {"schema_version": "sparse-dft-job-1", "kind": "cli",
     "argv": ["scf", "examples/Si.POSCAR", "--paw-root", "...", "--out", "..."],
     "env": {"OMP_NUM_THREADS": "2"}, "cwd": "/work", "label": "Si SCF"}

``kind`` is ``cli`` (``sparse_dft.cli.main(argv)``), ``api`` (``sparse_dft.api.dispatch``
with ``method`` and ``params``) or ``exec`` (an external command list, used
for tools and tests).  ``env`` values of ``null`` unset a variable.

``sparse-dft worker JOB.json [--status STATUS.json]`` runs one job in-process and
keeps an atomically written status document.  :class:`JobProcess` launches
such a worker as an independent process group (POSIX session or Windows Job
Object) so a cancel also stops the native solver started by the worker.
Nothing here depends on Qt.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

JOB_SCHEMA = 'sparse-dft-job-1'
STATUS_SCHEMA = 'sparse-dft-job-status-1'
JOB_KINDS = ('cli', 'api', 'exec')
STATES = ('queued', 'running', 'finished', 'failed', 'cancelled')
TERMINAL_STATES = ('finished', 'failed', 'cancelled')
CANCELLED_RETURNCODE = 143  # 128 + SIGTERM, also used for Windows TerminateJobObject
# Options whose values must never reach logs, job files or status documents.
SECRET_OPTIONS = ('--api-key',)
SECRET_ENV = ('MP_API_KEY', 'MAPI_KEY')
UTF8_ENV = {'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8'}


class JobError(ValueError):
    """Invalid job description."""


class JobCancelled(BaseException):
    """Raised in the worker's main thread when a termination signal arrives."""


# --------------------------------------------------------------------------- job files

def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec='seconds')


def validate_job(job) -> dict:
    """Return a normalized copy of ``job`` or raise :class:`JobError`."""
    if not isinstance(job, dict):
        raise JobError('job description must be a JSON object')
    if job.get('schema_version') != JOB_SCHEMA:
        raise JobError(f'job schema_version must be {JOB_SCHEMA!r}, got {job.get("schema_version")!r}')
    kind = job.get('kind', 'cli')
    if kind not in JOB_KINDS:
        raise JobError(f'job kind must be one of {JOB_KINDS}, got {kind!r}')
    out = dict(job)
    out['kind'] = kind
    if kind in ('cli', 'exec'):
        argv = job.get('argv')
        if not isinstance(argv, list) or not argv or not all(isinstance(a, str) for a in argv):
            raise JobError('job argv must be a non-empty list of strings')
        out['argv'] = list(argv)
    else:
        if not isinstance(job.get('method'), str) or not job['method']:
            raise JobError('api job needs a method string')
        params = job.get('params', {})
        if not isinstance(params, dict):
            raise JobError('api job params must be an object')
        out['params'] = params
    env = job.get('env') or {}
    if not isinstance(env, dict) or not all(isinstance(k, str) and (v is None or isinstance(v, str))
                                            for k, v in env.items()):
        raise JobError('job env must map strings to strings or null')
    out['env'] = dict(env)
    cwd = job.get('cwd')
    if cwd is not None and not isinstance(cwd, str):
        raise JobError('job cwd must be a string path')
    label = job.get('label')
    if label is not None and not isinstance(label, str):
        raise JobError('job label must be a string')
    return out


def make_job(argv=None, *, kind='cli', env=None, cwd=None, label=None, method=None, params=None) -> dict:
    job = {'schema_version': JOB_SCHEMA, 'kind': kind, 'env': dict(env or {})}
    if kind == 'api':
        job.update(method=method, params=dict(params or {}))
    else:
        job['argv'] = [str(a) for a in argv]
    if cwd is not None:
        # The worker runs in its own directory; a relative cwd would be ambiguous there.
        job['cwd'] = str(Path(cwd).expanduser().resolve())
    if label is not None:
        job['label'] = label
    return validate_job(job)


def load_job(path) -> dict:
    try:
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError) as error:
        raise JobError(f'cannot read job description {path}: {error}') from error
    return validate_job(payload)


def atomic_write_json(path, payload) -> Path:
    """Write JSON through a temporary file and ``os.replace`` (retried on Windows sharing errors)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False, default=str) + '\n'
    fd, tmp = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as stream:
            stream.write(text)
        for attempt in range(50):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                # A reader on Windows may hold the target open briefly.
                if attempt == 49:
                    raise
                time.sleep(0.02)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
    return path


def write_job(path, job) -> Path:
    return atomic_write_json(path, validate_job(job))


def read_status(path) -> dict | None:
    """Read a status document; ``None`` while missing or mid-replace."""
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def default_status_path(job_path) -> Path:
    job_path = Path(job_path)
    return job_path.with_name(job_path.stem + '.status.json')


def mask_argv(argv) -> list[str]:
    """Copy of ``argv`` with secret option values replaced by ``***``."""
    out, hide = [], False
    for item in argv:
        if hide:
            out.append('***'); hide = False; continue
        for option in SECRET_OPTIONS:
            if item == option:
                hide = True
                break
            if item.startswith(option + '='):
                item = option + '=***'
                break
        out.append(item)
    return out


def _parsed_cli(argv):
    """argparse namespace for a CLI argv, or ``None`` (never prints or exits)."""
    if not argv or argv[0] in ('plot', 'worker'):
        return None
    try:
        from .cli import build_parser
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            return build_parser().parse_args(list(argv))
    except (SystemExit, Exception):
        return None


def output_dir_for(job, cwd=None) -> str | None:
    """Output directory (or file) a job writes, resolved against its cwd."""
    base = Path(cwd or job.get('cwd') or os.getcwd())
    if job.get('kind') == 'api':
        value = (job.get('params') or {}).get('output')
    elif job.get('kind') == 'cli':
        namespace = _parsed_cli(job['argv'])
        value = getattr(namespace, 'out', None) if namespace is not None else None
        if value is None and job['argv'] and job['argv'][0] == 'plot':
            argv = job['argv']
            for i, item in enumerate(argv):
                if item == '--out' and i + 1 < len(argv):
                    value = argv[i + 1]
                elif item.startswith('--out='):
                    value = item.split('=', 1)[1]
    else:
        value = None
    if not value:
        return None
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else (base / path))


# --------------------------------------------------------------------------- iteration logs

def iteration_log_names(command: str) -> list[str]:
    """Native iteration-log file names written for ``command`` (``<result>.iterations.jsonl``)."""
    return {'scf': ['scf.json.iterations.jsonl'],
            'bands': ['bands.json.iterations.jsonl']}.get(command, [])


def latest_iteration_log(root) -> Path | None:
    """Most recently modified ``*.iterations.jsonl`` under ``root`` (relaxations nest SCF runs)."""
    root = Path(root)
    if root.is_file():
        return root if root.name.endswith('.jsonl') else None
    if not root.is_dir():
        return None
    best, best_time = None, -1.0
    for candidate in root.rglob('*.iterations.jsonl'):
        with contextlib.suppress(OSError):
            mtime = candidate.stat().st_mtime
            if mtime > best_time:
                best, best_time = candidate, mtime
    return best


class IterationTail:
    """Incremental reader of a growing JSON-lines file.

    :meth:`read_new` returns only records whose line is complete (terminated
    by a newline).  A missing file yields nothing; truncation or replacement
    (a different inode, or a size below the read offset) restarts from the
    beginning and increments :attr:`resets`.  Malformed lines are counted in
    :attr:`errors` and skipped.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.offset = 0
        self.resets = 0
        self.errors = 0
        self.count = 0
        self._buffer = b''
        self._identity = None

    def reset(self):
        self.offset, self._buffer, self._identity = 0, b'', None

    def read_new(self, *, final=False) -> list[dict]:
        try:
            info = os.stat(self.path)
        except OSError:
            if self._identity is not None:
                self.reset(); self.resets += 1
            return []
        identity = (info.st_dev, info.st_ino)
        if self._identity is not None and (identity != self._identity or info.st_size < self.offset):
            self.reset(); self.resets += 1
        self._identity = identity
        data = b''
        if info.st_size > self.offset:
            try:
                with open(self.path, 'rb') as stream:
                    stream.seek(self.offset)
                    data = stream.read(info.st_size - self.offset)
            except OSError:
                return []
            self.offset += len(data)
        buffer = self._buffer + data
        lines = buffer.split(b'\n')
        self._buffer = lines.pop()
        if final and self._buffer.strip():
            lines.append(self._buffer); self._buffer = b''
        records = []
        for raw in lines:
            raw = raw.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw.decode('utf-8', errors='replace'))
            except ValueError:
                self.errors += 1
                continue
            if isinstance(record, dict):
                records.append(record)
        self.count += len(records)
        return records

    def read_all(self) -> list[dict]:
        self.reset()
        return self.read_new(final=True)


# --------------------------------------------------------------------------- worker side

def _windows_std_fd(fd):
    """CRT descriptor for an inherited Win32 std handle (windowed apps start without one)."""
    import ctypes
    import msvcrt
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel32.GetStdHandle.restype = ctypes.c_void_p
    handle = kernel32.GetStdHandle(-11 if fd == 1 else -12)
    if not handle or handle == ctypes.c_void_p(-1).value:
        raise OSError('no inherited std handle')
    return msvcrt.open_osfhandle(handle, os.O_WRONLY)


def _reconfigure(stream, fd):
    if stream is None:
        # Windowed frozen apps may start without Python std streams although the
        # launcher passed pipes; recover them, else discard output (status still works).
        try:
            if os.name == 'nt':
                fd = _windows_std_fd(fd)
            return os.fdopen(fd, 'w', encoding='utf-8', errors='replace', buffering=1)
        except (OSError, ImportError, AttributeError, ValueError):
            return open(os.devnull, 'w', encoding='utf-8')
    with contextlib.suppress(AttributeError, ValueError, io.UnsupportedOperation):
        stream.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
    return stream


def configure_stdio():
    """UTF-8, line-buffered stdout/stderr regardless of the console code page."""
    sys.stdout = _reconfigure(sys.stdout, 1)
    sys.stderr = _reconfigure(sys.stderr, 2)


def _log(message):
    print(f'[sparse-dft worker] {message}', file=sys.stderr, flush=True)


def _install_signal_handlers():
    def handler(signum, _frame):
        # Further signals during cleanup must not interrupt the status write.
        for name in ('SIGTERM', 'SIGINT', 'SIGBREAK', 'SIGHUP'):
            sig = getattr(signal, name, None)
            if sig is not None:
                with contextlib.suppress(ValueError, OSError):
                    signal.signal(sig, signal.SIG_IGN)
        raise JobCancelled(signum)
    for name in ('SIGTERM', 'SIGINT', 'SIGBREAK', 'SIGHUP'):
        sig = getattr(signal, name, None)
        if sig is not None:
            with contextlib.suppress(ValueError, OSError):
                signal.signal(sig, handler)


def cancel_request_path(status_path) -> Path:
    """File whose appearance asks a running worker to cancel (works on every platform)."""
    status_path = Path(status_path)
    return status_path.with_name(status_path.name + '.cancel')


def _self_terminate():
    """Deliver the cancel to this worker's main thread.

    POSIX: SIGTERM to the worker only (never to a process group it does not
    lead); the handler raises :class:`JobCancelled`, which makes
    ``subprocess.run`` kill the native solver.  Windows: the caller writes the
    final status and exits; the worker's own Job Object (kill-on-close) then
    removes the solver.
    """
    if os.name == 'nt':
        return False
    os.kill(os.getpid(), signal.SIGTERM)
    return True


def _watch(status_path, status, parent_pid=None, interval=0.25):
    cancel_file = cancel_request_path(status_path)
    def loop():
        ticks = 0
        while True:
            time.sleep(interval)
            ticks += 1
            reason = None
            if cancel_file.exists():
                reason = 'cancel requested'
            elif parent_pid and ticks % 8 == 0 and not _pid_alive(parent_pid):
                reason = f'parent {parent_pid} exited'
            if reason is None:
                continue
            _log(f'{reason}; cancelling job')
            if not _self_terminate():
                status.update(state='cancelled', returncode=CANCELLED_RETURNCODE, end_time=now_iso(),
                              error=reason, updated_at=now_iso())
                with contextlib.suppress(Exception):
                    atomic_write_json(status_path, status)
                with contextlib.suppress(Exception):
                    sys.stdout.flush(); sys.stderr.flush()
                os._exit(CANCELLED_RETURNCODE)
            return
    threading.Thread(target=loop, name='sparse-dft-worker-watch', daemon=True).start()


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _windows_self_job():
    """Windows: place the worker in its own kill-on-close Job Object (nested jobs, Windows 8+)
    so the native solver dies with the worker even if nobody terminates it."""
    if os.name != 'nt':
        return None
    try:
        job = WindowsJobObject()
        job.assign_current()
        _SELF_JOB.append(job)  # keep the handle open for the worker's lifetime
        return job
    except Exception as error:  # noqa: BLE001 - best effort; the parent job still applies
        _log(f'could not create worker job object: {error}')
        return None


_SELF_JOB: list = []


def _apply_env(env):
    for key, value in env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _run_job(job) -> int:
    kind = job['kind']
    if kind == 'cli':
        from . import cli
        rc = cli.main(list(job['argv']))
        return int(rc or 0)
    if kind == 'api':
        from . import api
        response = api.dispatch(job['method'], job.get('params') or {})
        print(json.dumps(response, indent=2, ensure_ascii=False, default=str), flush=True)
        return 0 if response.get('status') == 'PASS' else 3
    # exec: an external command inheriting this worker's (UTF-8) streams.
    sys.stdout.flush(); sys.stderr.flush()
    return subprocess.run(list(job['argv']), stdin=subprocess.DEVNULL).returncode


def base_status(job, *, state, status_path=None) -> dict:
    return {'schema_version': STATUS_SCHEMA, 'state': state, 'label': job.get('label'),
            'kind': job.get('kind'), 'argv': mask_argv(job.get('argv') or []),
            'method': job.get('method'), 'cwd': job.get('cwd'),
            'output_dir': output_dir_for(job), 'pid': None, 'start_time': None,
            'end_time': None, 'duration_s': None, 'returncode': None, 'error': None,
            'updated_at': now_iso()}


def main(argv=None) -> int:
    """``sparse-dft worker JOB.json [--status STATUS.json]``; returns the job's exit code."""
    configure_stdio()
    parser = argparse.ArgumentParser(prog='sparse-dft worker', description='run one sparse-dft-job-1 description')
    parser.add_argument('job', help='job description JSON (schema sparse-dft-job-1)')
    parser.add_argument('--status', help='status JSON, rewritten atomically (default JOB.status.json)')
    try:
        args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    except SystemExit as exit_:
        return int(exit_.code or 0)
    status_path = Path(args.status) if args.status else default_status_path(args.job)
    try:
        job = load_job(args.job)
    except JobError as error:
        _log(f'invalid job: {error}')
        atomic_write_json(status_path, {'schema_version': STATUS_SCHEMA, 'state': 'failed',
                                        'error': str(error), 'returncode': 2, 'pid': os.getpid(),
                                        'end_time': now_iso(), 'updated_at': now_iso()})
        return 2
    # Every Python child (and the job's own output) speaks UTF-8; the job may override.
    _apply_env(UTF8_ENV)
    _apply_env(job['env'])
    if job.get('cwd'):
        os.chdir(job['cwd'])
    status = base_status(job, state='running')
    status.update(pid=os.getpid(), start_time=now_iso(), cwd=os.getcwd(),
                  output_dir=output_dir_for(job, os.getcwd()))
    started = time.monotonic()
    _install_signal_handlers()
    watch = os.environ.get('SDFT_WORKER_PARENT_PID', '')
    _windows_self_job()
    with contextlib.suppress(FileNotFoundError):
        cancel_request_path(status_path).unlink()
    atomic_write_json(status_path, status)
    _watch(status_path, status, int(watch) if watch.isdigit() and os.name != 'nt' else None)
    shown = ' '.join(mask_argv(job.get('argv') or [job.get('method') or '']))
    _log(f'start pid={os.getpid()} label={job.get("label") or "-"} :: {shown}')
    rc, state, error = 1, 'failed', None
    try:
        rc = _run_job(job)
        state = 'finished' if rc == 0 else 'failed'
    except JobCancelled:
        rc, state = CANCELLED_RETURNCODE, 'cancelled'
    except SystemExit as exit_:
        code = exit_.code
        rc = code if isinstance(code, int) else (0 if code is None else 2)
        state = 'finished' if rc == 0 else 'failed'
        if code is not None and not isinstance(code, int):
            error = str(code)
    except BaseException as exc:  # noqa: BLE001 - the status must record every failure
        traceback.print_exc()
        rc, state, error = 1, 'failed', f'{type(exc).__name__}: {exc}'
    finally:
        with contextlib.suppress(Exception):
            sys.stdout.flush()
        status.update(state=state, returncode=rc, error=error, end_time=now_iso(),
                      duration_s=round(time.monotonic() - started, 3), updated_at=now_iso())
        with contextlib.suppress(Exception):
            atomic_write_json(status_path, status)
        if state == 'cancelled':
            _terminate_own_group()
        _log(f'{state} returncode={rc} duration={status["duration_s"]:.1f}s')
    return rc


def _terminate_own_group():
    """After a cancel, SIGTERM every remaining descendant when this worker leads its group.

    Only a group leader (started by :class:`JobProcess` or the GUI with a new
    session) does this, so a worker run by hand never signals its shell.
    """
    if os.name == 'nt' or not hasattr(os, 'killpg'):
        return
    try:
        if os.getpgrp() != os.getpid():
            return
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        os.killpg(os.getpgrp(), signal.SIGTERM)
    except (OSError, ValueError):
        pass


# --------------------------------------------------------------------------- process trees

class WindowsJobObject:
    """Minimal ctypes wrapper of a Windows Job Object with KILL_ON_JOB_CLOSE (no pywin32)."""

    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    JOB_OBJECT_LIMIT_BREAKAWAY_OK = 0x00000800
    JobObjectExtendedLimitInformation = 9
    PROCESS_TERMINATE = 0x0001
    PROCESS_SET_QUOTA = 0x0100

    def __init__(self):
        import ctypes
        from ctypes import wintypes
        self._ctypes = ctypes
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        self._kernel32 = kernel32

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [(n, ctypes.c_ulonglong) for n in (
                'ReadOperationCount', 'WriteOperationCount', 'OtherOperationCount',
                'ReadTransferCount', 'WriteTransferCount', 'OtherTransferCount')]

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [('PerProcessUserTimeLimit', ctypes.c_longlong),
                        ('PerJobUserTimeLimit', ctypes.c_longlong),
                        ('LimitFlags', wintypes.DWORD),
                        ('MinimumWorkingSetSize', ctypes.c_size_t),
                        ('MaximumWorkingSetSize', ctypes.c_size_t),
                        ('ActiveProcessLimit', wintypes.DWORD),
                        ('Affinity', ctypes.c_size_t),
                        ('PriorityClass', wintypes.DWORD),
                        ('SchedulingClass', wintypes.DWORD)]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [('BasicLimitInformation', JOBOBJECT_BASIC_LIMIT_INFORMATION),
                        ('IoInfo', IO_COUNTERS),
                        ('ProcessMemoryLimit', ctypes.c_size_t),
                        ('JobMemoryLimit', ctypes.c_size_t),
                        ('PeakProcessMemoryUsed', ctypes.c_size_t),
                        ('PeakJobMemoryUsed', ctypes.c_size_t)]

        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.TerminateJobObject.restype = wintypes.BOOL
        kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        self.handle = kernel32.CreateJobObjectW(None, None)
        if not self.handle:
            raise OSError(ctypes.get_last_error(), 'CreateJobObjectW failed')
        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = self.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(self.handle, self.JobObjectExtendedLimitInformation,
                                                ctypes.byref(info), ctypes.sizeof(info)):
            error = ctypes.get_last_error(); self.close()
            raise OSError(error, 'SetInformationJobObject failed')

    def assign_pid(self, pid: int):
        kernel32 = self._kernel32
        process = kernel32.OpenProcess(self.PROCESS_TERMINATE | self.PROCESS_SET_QUOTA, False, int(pid))
        if not process:
            raise OSError(self._ctypes.get_last_error(), 'OpenProcess failed')
        try:
            if not kernel32.AssignProcessToJobObject(self.handle, process):
                raise OSError(self._ctypes.get_last_error(), 'AssignProcessToJobObject failed')
        finally:
            kernel32.CloseHandle(process)

    def assign_current(self):
        if not self._kernel32.AssignProcessToJobObject(self.handle, self._kernel32.GetCurrentProcess()):
            raise OSError(self._ctypes.get_last_error(), 'AssignProcessToJobObject failed')

    def terminate(self, code: int = CANCELLED_RETURNCODE) -> bool:
        return bool(self.handle) and bool(self._kernel32.TerminateJobObject(self.handle, code))

    def close(self):
        if self.handle:
            self._kernel32.CloseHandle(self.handle)
            self.handle = None


WORKER_EXE = 'sparse-dft-worker'


def frozen_worker_executable() -> Path | None:
    """Console-subsystem worker bundled next to a frozen GUI (``sparse-dft-worker[.exe]``), if any.

    A windowed GUI executable has no console, so native console programs it
    starts would open visible windows on Windows; the console worker gives
    them an inherited, hidden console (QProcess uses CREATE_NO_WINDOW).
    """
    name = WORKER_EXE + ('.exe' if os.name == 'nt' else '')
    candidate = Path(sys.executable).resolve().with_name(name)
    return candidate if candidate.is_file() and candidate != Path(sys.executable).resolve() else None


def worker_command(job_path, status_path=None) -> list[str]:
    """Command line that runs ``job_path`` in a fresh worker process."""
    extra = ['--status', str(status_path)] if status_path else []
    if getattr(sys, 'frozen', False):
        worker = frozen_worker_executable()
        return [str(worker or sys.executable), '--worker', str(job_path), *extra]
    return [sys.executable, '-m', 'sparse_dft.cli', 'worker', str(job_path), *extra]


def worker_environment(base=None, extra=None) -> dict:
    """Environment for a worker: UTF-8 Python IO, the repo on ``PYTHONPATH`` for source runs."""
    env = dict(os.environ if base is None else base)
    env.update(UTF8_ENV)
    if not getattr(sys, 'frozen', False):
        root = str(Path(__file__).resolve().parents[1])
        parts = [p for p in env.get('PYTHONPATH', '').split(os.pathsep) if p]
        if root not in parts:
            env['PYTHONPATH'] = os.pathsep.join([root, *parts])
    env['SDFT_WORKER_PARENT_PID'] = str(os.getpid())
    for key, value in (extra or {}).items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = str(value)
    return env


def popen_platform_kwargs() -> dict:
    """``subprocess.Popen`` options that start a worker as its own process group."""
    if os.name == 'nt':
        flags = getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0x200)
        flags |= getattr(subprocess, 'CREATE_NO_WINDOW', 0x08000000)
        return {'creationflags': flags}
    return {'start_new_session': True}


def request_cancel(status_path) -> None:
    """Ask a worker to cancel cooperatively (it writes a final ``cancelled`` status)."""
    path = cancel_request_path(status_path)
    with contextlib.suppress(OSError):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(now_iso() + '\n', encoding='utf-8')


def signal_tree(pid: int) -> None:
    """Soft stop of a worker process group: SIGTERM on POSIX (Windows relies on the cancel file)."""
    if os.name == 'nt' or not pid:
        return
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.killpg(pid, signal.SIGTERM)


def kill_tree(pid: int, job_object=None) -> None:
    """Hard stop of a worker and every descendant (SIGKILL group / TerminateJobObject)."""
    if not pid:
        return
    if os.name == 'nt':
        if job_object is not None and job_object.terminate():
            return
        subprocess.run(['taskkill', '/PID', str(pid), '/T', '/F'], capture_output=True,
                       encoding='utf-8', errors='replace')
        return
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.killpg(pid, signal.SIGKILL)


def group_alive(pgid: int) -> bool:
    """True while any member of the POSIX process group exists (zombies included)."""
    if os.name == 'nt' or not pgid:
        return False
    try:
        os.killpg(pgid, 0)
    except OSError:
        return False
    return True


def terminate_tree(pid: int, *, grace: float = 3.0, job_object=None, poll=None, status_path=None) -> None:
    """Blocking stop: cancel file + SIGTERM, then the hard kill after ``grace`` seconds.

    ``poll`` (e.g. ``Popen.poll``) reaps the worker so a zombie is not taken
    for a live group member.
    """
    if status_path is not None:
        request_cancel(status_path)
    signal_tree(pid)
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        exited = poll() is not None if poll is not None else False
        if os.name == 'nt':
            if exited:
                break
        elif not group_alive(pid):
            return
        time.sleep(0.05)
    # The worker may be gone while an orphaned solver keeps the group (or job) alive.
    kill_tree(pid, job_object)


def finalize_status(status_path, job, returncode, *, cancelled=False) -> dict:
    """Make the status terminal after the worker exited (e.g. it was SIGKILLed)."""
    status = read_status(status_path) or base_status(job, state='queued')
    if status.get('state') not in TERMINAL_STATES:
        state = 'cancelled' if cancelled else ('finished' if returncode == 0 else 'failed')
        status.update(state=state, returncode=returncode, end_time=now_iso(), updated_at=now_iso())
        if status.get('start_time') and status.get('duration_s') is None:
            with contextlib.suppress(Exception):
                start = _dt.datetime.fromisoformat(status['start_time'])
                status['duration_s'] = round((_dt.datetime.now(_dt.timezone.utc) - start).total_seconds(), 3)
        with contextlib.suppress(Exception):
            atomic_write_json(status_path, status)
    return status


class JobProcess:
    """One job in an independent worker process; output lines arrive on reader threads.

    Callbacks (``on_stdout(line)``, ``on_stderr(line)``, ``on_exit(returncode)``)
    run on background threads; GUI code must marshal them to its own thread.
    """

    def __init__(self, job, workdir=None, *, on_stdout=None, on_stderr=None, on_exit=None,
                 env=None, grace: float = 3.0):
        self.job = validate_job(job)
        self.workdir = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix='sparse-dft-job-'))
        self.job_path = self.workdir / 'job.json'
        self.status_path = self.workdir / 'status.json'
        self.on_stdout, self.on_stderr, self.on_exit = on_stdout, on_stderr, on_exit
        self.extra_env = dict(env or {})
        self.grace = grace
        self.process: subprocess.Popen | None = None
        self.cancel_requested = False
        self.started_at = None
        self.ended_at = None
        self._job_object = None
        self._threads: list[threading.Thread] = []
        self._exit_lock = threading.Lock()
        self._exit_reported = False

    # ---- lifecycle
    def start(self):
        self.workdir.mkdir(parents=True, exist_ok=True)
        write_job(self.job_path, self.job)
        atomic_write_json(self.status_path, base_status(self.job, state='queued'))
        command = worker_command(self.job_path, self.status_path)
        self.process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, env=worker_environment(extra=self.extra_env),
                                        cwd=str(self.workdir), **popen_platform_kwargs())
        self.started_at = time.time()
        if os.name == 'nt':
            try:
                self._job_object = WindowsJobObject()
                self._job_object.assign_pid(self.process.pid)
            except Exception:  # noqa: BLE001 - taskkill /T remains as fallback
                self._job_object = None
        for stream, callback, name in ((self.process.stdout, lambda line: self._emit(self.on_stdout, line), 'stdout'),
                                       (self.process.stderr, lambda line: self._emit(self.on_stderr, line), 'stderr')):
            thread = threading.Thread(target=self._pump, args=(stream, callback), name=f'sparse-dft-job-{name}', daemon=True)
            thread.start(); self._threads.append(thread)
        waiter = threading.Thread(target=self._wait_exit, name='sparse-dft-job-wait', daemon=True)
        waiter.start(); self._threads.append(waiter)
        return self

    @staticmethod
    def _emit(callback, line):
        if callback is not None:
            with contextlib.suppress(Exception):
                callback(line)

    @staticmethod
    def _pump(stream, callback):
        try:
            for raw in iter(stream.readline, b''):
                callback(raw.decode('utf-8', errors='replace').rstrip('\r\n'))
        except (OSError, ValueError):
            pass
        finally:
            with contextlib.suppress(Exception):
                stream.close()

    def _wait_exit(self):
        returncode = self.process.wait()
        # An orphaned descendant may hold the pipes open; do not wait on it forever.
        for thread in self._threads[:2]:
            thread.join(timeout=2)
        self.ended_at = time.time()
        finalize_status(self.status_path, self.job, returncode, cancelled=self.cancel_requested)
        if self._job_object is not None:
            self._job_object.close()
        with self._exit_lock:
            if self._exit_reported:
                return
            self._exit_reported = True
        self._emit(self.on_exit, returncode)

    def cancel(self, *, wait: bool = False):
        """Stop the worker and its descendants; non-blocking unless ``wait``."""
        if self.process is None or self.process.poll() is not None:
            return
        self.cancel_requested = True
        def run():
            terminate_tree(self.process.pid, grace=self.grace, job_object=self._job_object,
                           poll=self.process.poll, status_path=self.status_path)
        if wait:
            run()
            self.wait(timeout=self.grace + 5)
        else:
            threading.Thread(target=run, name='sparse-dft-job-cancel', daemon=True).start()

    # ---- queries
    @property
    def pid(self):
        return self.process.pid if self.process else None

    @property
    def returncode(self):
        return self.process.poll() if self.process else None

    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def wait(self, timeout=None):
        if self.process is None:
            return None
        rc = self.process.wait(timeout=timeout)
        deadline = time.monotonic() + 5
        while not self._exit_reported and time.monotonic() < deadline:
            time.sleep(0.01)
        return rc

    def status(self) -> dict | None:
        return read_status(self.status_path)

    def output_dir(self):
        return output_dir_for(self.job)
