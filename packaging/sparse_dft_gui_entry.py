"""PyInstaller entry point of the frozen Sparse-DFT GUI.

``sparse-dft-gui --worker JOB.json [--status STATUS.json]`` runs one job in this
process (the GUI launches its workers this way); anything else starts Qt.
The worker branch must not import Qt, so it is checked before the GUI import.
"""
import multiprocessing
import sys
from pathlib import Path


def main():
    multiprocessing.freeze_support()
    argv = sys.argv[1:]
    worker_binary = Path(sys.executable).stem.lower() == 'sparse-dft-worker'
    if worker_binary and (not argv or argv[0] != '--worker'):
        print('usage: sparse-dft-worker --worker JOB.json [--status STATUS.json]', file=sys.stderr)
        return 2
    if argv and argv[0] == '--worker':
        from sparse_dft.runner import main as worker_main
        return worker_main(argv[1:])
    from sparse_dft_gui.app import main as gui_main
    return gui_main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
