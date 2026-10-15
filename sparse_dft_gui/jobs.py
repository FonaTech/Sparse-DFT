"""Job queue: each job is an independent ``sparse-dft worker`` process driven by ``QProcess``."""
from __future__ import annotations

import datetime as _dt
import os
import time
import uuid
from pathlib import Path

from PyQt6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, pyqtSignal

from sparse_dft import runner


class Job:
    def __init__(self, job_id, spec, workdir, *, command, display):
        self.id = job_id
        self.spec = spec              # sparse-dft-job-1 description (no secrets)
        self.workdir = Path(workdir)
        self.job_path = self.workdir / 'job.json'
        self.status_path = self.workdir / 'status.json'
        self.command = command        # e.g. 'scf'
        self.display = display        # copyable command line
        self.label = spec.get('label') or command
        self.state = 'queued'
        self.returncode = None
        self.started = None
        self.ended = None
        self.process: QProcess | None = None
        self.pid = None
        self.secret_env: dict = {}
        self.extra_env: dict = {}
        self.cancel_requested = False
        self.lines: list[tuple[str, str]] = []  # (stream, text)
        self.output_dir = runner.output_dir_for(spec)
        self._partial = {'stdout': b'', 'stderr': b''}

    @property
    def duration(self):
        if self.started is None:
            return None
        return (self.ended or time.time()) - self.started

    def finished(self) -> bool:
        return self.state in runner.TERMINAL_STATES


def format_duration(seconds) -> str:
    if seconds is None:
        return '—'
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f'{h:d}:{m:02d}:{s:02d}' if h else f'{m:d}:{s:02d}'


def jobs_root() -> Path:
    from sparse_dft.resources import user_data_dir
    return user_data_dir() / 'jobs'


class JobManager(QObject):
    """Starts, tracks and cancels jobs; all signals are emitted on the GUI thread."""

    job_added = pyqtSignal(object)
    job_changed = pyqtSignal(object)
    job_output = pyqtSignal(object, str, str)  # job, stream, line
    job_finished = pyqtSignal(object)

    def __init__(self, parent=None, *, root: Path | None = None, grace_ms: int = 3000):
        super().__init__(parent)
        self.root = Path(root) if root else jobs_root()
        self.grace_ms = grace_ms
        self.jobs: list[Job] = []
        self.env_provider = None  # callable -> dict of extra env for every job
        self._status_timer = QTimer(self)
        self._status_timer.setInterval(1000)
        self._status_timer.timeout.connect(self._tick)
        self._status_timer.start()

    # ---- submission
    def submit(self, argv=None, *, label=None, kind='cli', env=None, secret_env=None, cwd=None,
               method=None, params=None, command=None) -> Job:
        env = dict(env or {})
        if self.env_provider is not None:
            merged = dict(self.env_provider() or {})
            merged.update(env)
            env = merged
        spec = runner.make_job(argv, kind=kind, env={k: str(v) for k, v in env.items()},
                               cwd=cwd or os.getcwd(), label=label, method=method, params=params)
        stamp = _dt.datetime.now().strftime('%Y%m%d-%H%M%S')
        job_id = f'{stamp}-{uuid.uuid4().hex[:6]}'
        workdir = self.root / job_id
        from .argspec import display_command
        display = display_command(argv) if kind == 'cli' else (
            ' '.join(argv) if kind == 'exec' else f'api {method}')
        job = Job(job_id, spec, workdir, command=command or (argv[0] if argv else method or kind), display=display)
        job.secret_env = dict(secret_env or {})
        self.jobs.append(job)
        self.job_added.emit(job)
        self._start(job)
        return job

    def rerun(self, job: Job) -> Job:
        spec = job.spec
        new = self.submit(spec.get('argv'), label=job.label, kind=spec.get('kind', 'cli'),
                          env={k: v for k, v in spec.get('env', {}).items() if v is not None},
                          secret_env=job.secret_env, cwd=spec.get('cwd'), method=spec.get('method'),
                          params=spec.get('params'), command=job.command)
        return new

    def _start(self, job: Job):
        try:
            job.workdir.mkdir(parents=True, exist_ok=True)
            runner.write_job(job.job_path, job.spec)
            runner.atomic_write_json(job.status_path, runner.base_status(job.spec, state='queued'))
        except OSError as error:
            self._fail_start(job, str(error))
            return
        command = runner.worker_command(job.job_path, job.status_path)
        process = QProcess(self)
        environment = QProcessEnvironment.systemEnvironment()
        for key, value in runner.worker_environment(extra={**job.secret_env}).items():
            environment.insert(key, value)
        process.setProcessEnvironment(environment)
        process.setWorkingDirectory(str(job.workdir))
        process.setProgram(command[0])
        process.setArguments(command[1:])
        if os.name != 'nt':
            process.setUnixProcessParameters(QProcess.UnixProcessFlag.CreateNewSession)
        else:
            _windows_new_group(process)
        process.readyReadStandardOutput.connect(lambda j=job: self._read(j, 'stdout'))
        process.readyReadStandardError.connect(lambda j=job: self._read(j, 'stderr'))
        process.started.connect(lambda j=job: self._on_started(j))
        process.finished.connect(lambda code, status, j=job: self._on_finished(j, code, status))
        process.errorOccurred.connect(lambda error, j=job: self._on_error(j, error))
        job.process = process
        job.started = time.time()
        job.state = 'running'
        process.start()  # asynchronous; never blocks the event loop
        self.job_changed.emit(job)

    def _on_started(self, job: Job):
        job.pid = int(job.process.processId())
        job.process.closeWriteChannel()
        if os.name == 'nt':
            job.job_object = _windows_job_object(job.pid)
        self.job_changed.emit(job)

    def _fail_start(self, job: Job, message: str):
        job.state, job.returncode, job.ended = 'failed', -1, time.time()
        self._line(job, 'stderr', f'[sparse-dft gui] could not start worker: {message}')
        self.job_changed.emit(job)
        self.job_finished.emit(job)

    # ---- process events
    def _read(self, job: Job, stream: str):
        process = job.process
        if process is None:
            return
        data = bytes(process.readAllStandardOutput() if stream == 'stdout' else process.readAllStandardError())
        buffer = job._partial[stream] + data
        *complete, job._partial[stream] = buffer.split(b'\n')
        for raw in complete:
            self._line(job, stream, raw.decode('utf-8', errors='replace').rstrip('\r'))

    def _flush(self, job: Job):
        for stream in ('stdout', 'stderr'):
            if job._partial[stream]:
                self._line(job, stream, job._partial[stream].decode('utf-8', errors='replace').rstrip('\r'))
                job._partial[stream] = b''

    def _line(self, job: Job, stream: str, text: str):
        job.lines.append((stream, text))
        if len(job.lines) > 20000:
            del job.lines[:5000]
        self.job_output.emit(job, stream, text)

    def _on_error(self, job: Job, error):
        if error == QProcess.ProcessError.FailedToStart:
            self._fail_start(job, job.process.errorString() if job.process else str(error))

    def _on_finished(self, job: Job, code: int, exit_status):
        if job.finished():
            return
        self._read(job, 'stdout'); self._read(job, 'stderr'); self._flush(job)
        crashed = exit_status == QProcess.ExitStatus.CrashExit
        returncode = code if not crashed else (-15 if job.cancel_requested else -1)
        status = runner.finalize_status(job.status_path, job.spec, returncode, cancelled=job.cancel_requested)
        job.returncode = status.get('returncode', returncode)
        job.state = status.get('state') or ('finished' if returncode == 0 else 'failed')
        if job.cancel_requested and job.state != 'finished':
            job.state = 'cancelled'
        job.ended = time.time()
        job.output_dir = status.get('output_dir') or job.output_dir
        if getattr(job, 'job_object', None) is not None:
            job.job_object.close(); job.job_object = None
        if job.process is not None:
            job.process.deleteLater()
        job.process = None
        self.job_changed.emit(job)
        self.job_finished.emit(job)

    # ---- control
    def cancel(self, job: Job):
        """Cooperative cancel (cancel file + SIGTERM to the group), hard kill after the grace period."""
        if job.process is None or job.finished():
            return
        job.cancel_requested = True
        runner.request_cancel(job.status_path)
        if job.pid:
            runner.signal_tree(job.pid)
        QTimer.singleShot(self.grace_ms, lambda j=job: self._hard_kill(j))
        self.job_changed.emit(job)

    def _hard_kill(self, job: Job):
        if job.pid and (not job.finished() or runner.group_alive(job.pid)):
            runner.kill_tree(job.pid, getattr(job, 'job_object', None))
            if job.process is not None and not job.finished():
                job.process.kill()

    def cancel_all(self, *, wait_ms: int = 0):
        for job in self.jobs:
            self.cancel(job)
        if wait_ms:
            deadline = time.monotonic() + wait_ms / 1000
            for job in self.jobs:
                if job.process is not None:
                    job.process.waitForFinished(max(1, int((deadline - time.monotonic()) * 1000)))
            for job in self.jobs:
                self._hard_kill(job)

    def remove_finished(self):
        self.jobs = [j for j in self.jobs if not j.finished()]

    def running(self) -> list[Job]:
        return [j for j in self.jobs if not j.finished()]

    def _tick(self):
        for job in self.jobs:
            if not job.finished():
                self.job_changed.emit(job)


def _windows_new_group(process: QProcess):
    """CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW on Windows (QProcess modifier hook)."""
    modifier = getattr(process, 'setCreateProcessArgumentsModifier', None)
    if modifier is None:
        return
    def apply(args):
        try:
            args.flags |= 0x00000200 | 0x08000000
        except Exception:  # noqa: BLE001
            pass
    modifier(apply)


def _windows_job_object(pid: int):
    try:
        job_object = runner.WindowsJobObject()
        job_object.assign_pid(pid)
        return job_object
    except Exception:  # noqa: BLE001 - taskkill /T is the fallback in kill_tree
        return None
