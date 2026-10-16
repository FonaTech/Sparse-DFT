"""Content-addressed PAW preparation; every reuse verifies original bytes.

This cache stores parsed numerical data only, never a geometry or SCF state.
A damaged entry is an error rather than a silent change of pseudopotential.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from .errors import InputError

FILES = ('POTCAR', 'PSCTR', 'numeric.bin', 'metadata.json', 'numeric.json')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): h.update(block)
    return h.hexdigest()


@contextmanager
def locked(path):
    with Path(path).open('a+b') as stream:
        if os.name == 'nt':
            import msvcrt
            if stream.tell() == 0: stream.write(b'0'); stream.flush()
            stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX)
        try: yield
        finally:
            if os.name == 'nt':
                stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else: fcntl.flock(stream, fcntl.LOCK_UN)


def _parser_identity():
    """Hash of the PAW parser source; a frozen app ships no .py files, so its
    executable stands in (a rebuilt app gets a fresh cache key)."""
    from . import paw
    source = Path(paw.__file__)
    if source.is_file():
        return sha(source)
    import sys
    return 'frozen:' + sha(sys.executable)


def prepare(archive, record_id, destination):
    enabled = os.environ.get('SDFT_PERSISTENT_PAW_CACHE', '1')
    if enabled not in ('0', '1'): raise InputError('SDFT_PERSISTENT_PAW_CACHE must be 0 or 1')
    if enabled == '0':
        archive.restore_dataset(record_id, destination)
        return {'reused': False}
    identity = {'schema': 'sparse-dft-prepared-paw-1', 'archive': archive.local_hashes(),
                'record_id': record_id, 'parser_source_sha256': _parser_identity()}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    root = Path(os.environ.get('SDFT_PAW_PREPARED_CACHE',
        str(Path(tempfile.gettempdir()) / ('sparse-dft-paw-cache-' + str(getattr(os, 'getuid', lambda: 0)())))))
    root.mkdir(parents=True, exist_ok=True)
    entry = root / key
    with locked(root / (key + '.lock')):
        reused = entry.exists()
        if not reused:
            stage = Path(tempfile.mkdtemp(prefix=key+'.', dir=root))
            try:
                archive.restore_dataset(record_id, stage)
                manifest = {'identity': identity, 'files': {name: sha(stage/name) for name in FILES}}
                (stage/'cache_manifest.json').write_text(json.dumps(manifest, sort_keys=True)+'\n', encoding='utf-8')
                stage.rename(entry)
            finally:
                if stage.exists(): shutil.rmtree(stage)
        try:
            manifest = json.loads((entry/'cache_manifest.json').read_text(encoding='utf-8'))
            if manifest['identity'] != identity or set(manifest['files']) != set(FILES):
                raise ValueError('identity or file list mismatch')
            for name in FILES:
                if sha(entry/name) != manifest['files'][name]: raise ValueError('hash differs: '+name)
            metadata = json.loads((entry/'metadata.json').read_text(encoding='utf-8'))
            if metadata['record_id'] != record_id or metadata['source_hashes']['archive'] != identity['archive']:
                raise ValueError('PAW metadata source mismatch')
            if manifest['files']['numeric.bin'] != metadata['numeric_cache']['binary_sha256']:
                raise ValueError('numeric binary hash mismatch')
            if archive.local_hashes() != identity['archive']: raise ValueError('archive changed')
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise InputError('prepared PAW cache integrity check failed', details={'entry': str(entry)}) from error
        dest = Path(destination); dest.mkdir(parents=True, exist_ok=True)
        for name in FILES:
            shutil.copy2(entry/name, dest/name)
            if sha(dest/name) != manifest['files'][name]: raise InputError('PAW cache copy hash differs')
        (dest/'prepared_cache_provenance.json').write_text(json.dumps({
            'key':key,'reused':reused,'identity':identity,'files':manifest['files']},indent=2)+'\n', encoding='utf-8')
    return {'key':key,'reused':reused}
