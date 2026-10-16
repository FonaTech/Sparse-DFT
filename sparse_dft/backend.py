"""Runtime backend and performance diagnostics.

The numerical core keeps one deterministic algorithm and reports optional
accelerators explicitly.  A detected device is never silently treated as an
active GPU backend; ``active`` is true only when the corresponding C++ build
feature was compiled and a runtime is available.
"""
from __future__ import annotations

import os
import platform
import json
import shutil
import subprocess
from pathlib import Path


def _version(command):
    try:
        return subprocess.check_output(command, stderr=subprocess.STDOUT, text=True,
                                       timeout=3, encoding='utf-8', errors='replace').splitlines()[0]
    except Exception:
        return None


def _native(name):
    """Bundled, configured or built executable; a non-existent path when absent."""
    from .errors import InputError
    from .resources import find_executable
    try:
        return find_executable(name)
    except InputError as exc:
        searched = (exc.details or {}).get('searched') or [name]
        return Path(searched[0].split(' (')[0])


def _openmp_probe():
    configured = os.environ.get("SDFT_OPENMP_PROBE")
    executable = Path(configured) if configured else _native("sdft_openmp_probe")
    if not executable.is_file():
        return {"compiled": False, "runtime": False, "active": False,
                "team_threads": 1, "max_threads": 1, "probe": str(executable),
                "probe_status": "executable_missing"}
    try:
        output = subprocess.check_output([str(executable)], stderr=subprocess.STDOUT,
                                         text=True, timeout=10, encoding='utf-8', errors='replace')
        result = json.loads(output)
        compiled = bool(result.get("compiled"))
        team = int(result.get("team_threads", 1))
        return {"compiled": compiled, "runtime": compiled,
                "active": compiled and team > 1,
                "team_threads": team, "max_threads": int(result.get("max_threads", team)),
                "requested_threads": int(result.get("requested_threads", team)),
                "probe": str(executable), "probe_status": "passed" if compiled else "serial_build"}
    except Exception as exc:
        return {"compiled": False, "runtime": False, "active": False,
                "team_threads": 1, "max_threads": 1, "probe": str(executable),
                "probe_status": "failed", "probe_error": str(exc)}


def backend_info():
    machine = platform.machine().lower()
    system = platform.system().lower()
    nvcc = shutil.which("nvcc")
    omp = _openmp_probe()
    requested = os.environ.get("OMP_NUM_THREADS")
    omp["threads_requested_by_environment"] = int(requested) if requested and requested.isdigit() else None
    executable = _native('sdft_scf')
    capabilities = {}
    if executable.is_file():
        try:
            capabilities = json.loads(subprocess.check_output([str(executable), '--backend-info'],
                text=True, stderr=subprocess.DEVNULL, timeout=3, encoding='utf-8', errors='replace'))
        except (OSError, subprocess.SubprocessError, ValueError):
            pass  # Older release: missing native capability protocol is explicit below.
    # Platform eligibility is separate from actual device visibility/dispatch.
    mps_available = system == "darwin" and machine in {"arm64", "aarch64"}
    return {
        "cpu": {"machine": machine, "system": system, "logical_threads": os.cpu_count() or 1},
        "openmp": {"requested": os.environ.get("SDFT_ENABLE_OPENMP", "1") not in ("0", "OFF", "false"),
                   **omp},
        "cuda": {"available": bool(nvcc), "nvcc": nvcc, "version": _version([nvcc, "--version"]) if nvcc else None,
                 "compiled": capabilities.get('cuda_compiled', False), "active": False},
        "mps": {"available": mps_available, "compiled": capabilities.get('metal_compiled', False),
                "active": False},
        "native_capabilities": capabilities,
        "algorithm": {
            "default_solver": "matrix-free FFT block Davidson",
            "dominant_scf_cost": "H applications O(Ngrid log Ngrid + NPW*Nproj + Nproj^2); subspace orthogonalization O(NPW*m^2) and Rayleigh-Ritz O(m^3)",
            "reduced_cost_paths": ["low-rank PAW projector application O(NPW*Nproj + Nproj^2)",
                                    "FFT Hartree/XC O(Ngrid log Ngrid)",
                                    "dense reference assembly has an OpenMP static loop when the runtime probe passes",
                                    "independent finite-difference displacements can run concurrently"],
            "reference_solver": "SDFT_OPERATOR=dense explicitly retains O(NPW^3) full eigensolve",
            "gpu_status": "native Metal compensated GEMM and CUDA FP64 source; execution and precision gates are per-run in OUTPUT.performance.json",
        },
    }


def benchmark_summary(samples=None):
    """Return a stable schema for C++/Python performance reports."""
    samples = samples or {}
    return {"schema_version": "sparse-dft-backend-benchmark-1", "backend": backend_info(),
            "samples": samples, "status": "DIAGNOSTIC_ONLY"}
