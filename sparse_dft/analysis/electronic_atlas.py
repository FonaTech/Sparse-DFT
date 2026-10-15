"""Quantitative analysis behind the electronic-structure atlas figures.

Every number an atlas panel prints is computed here from the loaded band,
projection and DOS data and returned with its definition and conditions, so
the figure and its provenance record carry the same values.

Conventions
-----------
* k vectors are converted to Cartesian Å⁻¹ with the reciprocal basis of the
  run (``reciprocal_basis`` includes 2π for Sparse-DFT and for vasprun.xml).
* effective mass: E(k) = E0 + ħ²k²/(2 m* m0), so m*/m0 = (ħ²/2m0) / c2 with
  c2 the quadratic coefficient (eV Å²) of a least-squares fit in |k| along one
  straight path direction; ħ²/(2 m0) = 3.80998 eV Å² (CODATA 2018).
* engines are compared only at k points present in both runs (fractional
  coordinates equal modulo a reciprocal lattice vector, tolerance 1e-5) and
  only when structure, cutoff, PAW datasets and XC agree; the energy zero of
  each code is kept (no shift) unless the caller asks otherwise.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..errors import InputError
from .electronic import (BandEdges, KPath, ProjectionSet, band_edges, canonical_orbital, channels_of, kpath,
                         orbital_l)

HBAR2_OVER_2M0 = 3.80998212          # eV Å², ħ²/(2 m_e), CODATA 2018
K_MATCH_TOL = 1e-5                   # fractional coordinates
DEGENERATE_EV = 1e-3                 # bands closer than this at the extremum are degenerate (heavy/light)
NEAR_DEGENERATE_EV = 0.040           # projection-parity pairs (anchor: 7-39 meV pairs exchange weight)


# --- geometry ------------------------------------------------------------------
def reciprocal_matrix(bands) -> np.ndarray | None:
    """3x3 reciprocal basis (rows b1, b2, b3) in Å⁻¹ including 2π, or None."""
    md = bands.metadata or {}
    basis = md.get('reciprocal_basis')
    if not basis or len(basis) != 3:
        return None
    b = np.asarray([[float(x) for x in row[:3]] for row in basis], dtype=float)
    if not md.get('reciprocal_basis_includes_2pi', True):
        b = b * 2.0 * math.pi
    return b


def cartesian_k(bands) -> np.ndarray | None:
    b = reciprocal_matrix(bands)
    if b is None:
        return None
    return np.asarray(bands.kpoints, dtype=float)[:, :3] @ b


def _kkey(k):
    return tuple(int(round(((float(c) % 1.0) + 1.0) % 1.0 * 1e5)) % 100000 for c in k[:3])


def match_kpoints(a, b, tol=K_MATCH_TOL) -> list[tuple[int, int]]:
    """Index pairs (i in a, j in b) with equal fractional k modulo G (|Δ| < tol)."""
    index = {}
    for j, k in enumerate(b.kpoints):
        index.setdefault(_kkey(k), j)
    pairs = []
    kb = np.asarray(b.kpoints, dtype=float)[:, :3]
    for i, k in enumerate(a.kpoints):
        j = index.get(_kkey(k))
        if j is None:
            continue
        d = np.asarray(k[:3], dtype=float) - kb[j]
        if np.max(np.abs(d - np.round(d))) < tol:
            pairs.append((i, j))
    return pairs


def same_path(a, b, tol=K_MATCH_TOL) -> bool:
    if a.nkpoints != b.nkpoints:
        return False
    d = np.asarray(a.kpoints, dtype=float)[:, :3] - np.asarray(b.kpoints, dtype=float)[:, :3]
    return bool(np.max(np.abs(d - np.round(d))) < tol)


# --- path analysis -----------------------------------------------------------------
@dataclass
class EdgeFit:
    """Parabolic fit of one band at one extremum along one path direction."""
    carrier: str                 # 'electron' | 'hole'
    band: int                    # 0-based band index
    kindex: int                  # path index of the extremum
    direction: str               # e.g. 'Γ→X' (towards the label the segment runs to)
    c2_ev_a2: float              # quadratic coefficient (eV Å²)
    mass: float                  # m*/m0 (signed: negative for holes at a maximum)
    r2: float
    k_window_inv_a: float        # |k| range used for the fit (Å⁻¹)
    npoints: int
    kx: list = field(default_factory=list)       # fitted |k| samples (signed, Å⁻¹)
    ey: list = field(default_factory=list)       # energies relative to the extremum (eV)

    uncertainty: float = 0.0     # |m*(order 2) - m*(order 1)| of the Richardson extrapolation (m0)
    slope: float = 0.0           # one-sided fits: linear coefficient c1 (eV Å), ~0 at an extremum
    window_ev: float = 0.0       # max |E - E0| of the samples used

    def curve(self, k):
        """Band-edge parabola c2 k² (eV) at signed |k| (Å⁻¹)."""
        k = np.asarray(k, dtype=float)
        return self.c2_ev_a2 * k * k

    def as_dict(self):
        return {'carrier': self.carrier, 'band': self.band + 1, 'kindex': self.kindex, 'direction': self.direction,
                'c2_ev_a2': self.c2_ev_a2, 'mass_m0': self.mass, 'uncertainty_m0': self.uncertainty,
                'k_window_inv_a': self.k_window_inv_a, 'energy_window_ev': self.window_ev,
                'samples_per_side': self.npoints, 'linear_slope_ev_a': self.slope,
                'method': 'Richardson extrapolation of (E-E0)/q^2 in q^2 (order 2, 3 nearest path points); '
                          'uncertainty = |order 2 - order 1|; m*/m0 = 3.80998 eV A^2 / c2'}


def _segment_of(kp: KPath, k: int):
    return [(s, e) for s, e in kp.segments if s <= k <= e]


def _tick_label_at(kp: KPath, x: float) -> str | None:
    for tx, name in kp.ticks:
        if abs(tx - x) < 1e-9:
            return name
    return None


def _point_label(bands, k):
    from .electronic import pretty_kpoint_label
    labels = bands.labels or []
    if 0 <= k < len(labels) and labels[k]:
        return _plain(pretty_kpoint_label(labels[k]))
    if all(abs(float(c)) < 1e-8 for c in bands.kpoints[k][:3]):
        return 'Γ'
    return None


def interior_label(bands, kp, k) -> str:
    """'0.85 Γ→X' for an unlabelled path point: its fraction along the labelled stretch it lies on."""
    name = _point_label(bands, k)
    if name:
        return name
    seg = _segment_of(kp, k)
    if not seg:
        return f'k{k}'
    s, e = seg[0]
    a = next((j for j in range(k, s - 1, -1) if _point_label(bands, j)), None)
    b = next((j for j in range(k, e + 1) if _point_label(bands, j)), None)
    if a is None or b is None or abs(kp.x[b] - kp.x[a]) < 1e-12:
        return f'k{k}'
    frac = (kp.x[k] - kp.x[a]) / (kp.x[b] - kp.x[a])
    return f'{frac:.2f} {_point_label(bands, a)}→{_point_label(bands, b)}'


def _direction_label(bands, kp, k0, step, s, e):
    """'A→B': A labels k0, B is the next labelled point along the path in ``step`` direction."""
    start = _point_label(bands, k0) or 'k₀'
    j = k0 + step
    while s <= j <= e:
        name = _point_label(bands, j)
        if name and abs(kp.x[j] - kp.x[k0]) > 1e-9:
            return f'{start}→{name}'
        j += step
    return f'{start}→path'


def _line_label(bands, kp, k0, s, e):
    """'M–K–Γ' through a labelled k0, 'along Γ–X' for an interior k0."""
    left = _direction_label(bands, kp, k0, -1, s, e).split('→')[-1]
    right = _direction_label(bands, kp, k0, 1, s, e).split('→')[-1]
    here = _point_label(bands, k0)
    return f'{left}–{here}–{right}' if here else f'along {left}–{right}'


def _plain(label):
    if not label:
        return None
    text = label.replace('$', '').replace(r'\mathrm', '').replace('{', '').replace('}', '')
    for a, b in ((r'\Gamma', 'Γ'), (r'\Delta', 'Δ'), (r'\Lambda', 'Λ'), (r'\Sigma', 'Σ')):
        text = text.replace(a, b)
    return text


def _is_trim(k) -> bool:
    """Time-reversal-invariant point (2k = G): dE/dk = 0 there for a non-degenerate band."""
    return all(abs(2.0 * float(c) - round(2.0 * float(c))) < 1e-6 for c in k[:3])


def _richardson_c2(q, s):
    """c2 = lim_{q->0} s(q)/q² from samples s(q) = c2 q² + c4 q⁴ + ...: polynomial in q², orders 1 and 2."""
    f, x = s / q ** 2, q ** 2
    one = np.polyfit(x[:2], f[:2], 1)
    two = np.polyfit(x[:3], f[:3], 2)
    return float(np.polyval(two, 0.0)), float(np.polyval(one, 0.0))


def _two_sided_fit(bands, kp, rows, band, k0, carrier, a, b, qa, qb, s, e, e0, max_window_ev, max_rel):
    """Signed-q fit E - E0 = c1 q + c2 q² + c3 q³ + c4 q⁴ through 3 points on each side (unequal steps).

    The uncertainty compares with the 2-points-per-side cubic fit.  The point is
    accepted as an extremum when the fitted stationary point -c1/(2 c2) lies within
    half the first step of k0.
    """
    q = np.concatenate([-qa[::-1], qb])
    vals = np.concatenate([rows[a[::-1], band], rows[b, band]]) - e0
    if float(np.max(np.abs(vals))) > max_window_ev:
        return None
    qq, vv = np.concatenate([q, [0.0]]), np.concatenate([vals, [0.0]])
    hi = np.polyfit(qq, vv, 4)                     # c4 c3 c2 c1 c0
    inner = np.abs(qq) <= max(qa[1], qb[1]) + 1e-12
    lo = np.polyfit(qq[inner], vv[inner], 3)       # c3 c2 c1 c0
    c2, c1, c2_low = float(hi[2]), float(hi[3]), float(lo[1])
    if (c2 > 0) != (carrier == 'electron') or abs(c2) < 1e-6:
        return None
    if abs(c1 / (2.0 * c2)) > 0.5 * min(qa[0], qb[0]):
        return None
    rel = abs(c2 - c2_low) / abs(c2)
    if rel > max_rel:
        return None
    fit = EdgeFit(carrier, band, k0, _line_label(bands, kp, k0, s, e), c2, HBAR2_OVER_2M0 / c2, 1.0 - rel,
                  float(max(qa[-1], qb[-1])), 3, list(q), list(vals))
    fit.uncertainty = abs(HBAR2_OVER_2M0 / c2 - HBAR2_OVER_2M0 / c2_low)
    fit.window_ev = float(np.max(np.abs(vals)))
    fit.slope = c1
    return fit


def _one_sided_fit(bands, kp, rows, band, k0, carrier, side, step, s, e, kc, x0, e0, max_window_ev, max_rel):
    """(E-E0)/q = c1 + c2 q (+ c3 q²) from 2 and 3 points; accepted only for an extremum (|c1| q1 < 10 % of E-E0)."""
    q = np.array([np.linalg.norm(kc[i] - x0) for i in side])
    vals = rows[side, band] - e0
    if float(np.max(np.abs(vals))) > max_window_ev:
        return None
    f = vals / q
    low = np.polyfit(q[:2], f[:2], 1)          # c2, c1
    high = np.polyfit(q, f, 2)                 # c3, c2, c1
    c2, c1, c2_low = float(high[1]), float(high[2]), float(low[0])
    if (c2 > 0) != (carrier == 'electron') or abs(c2) < 1e-6:
        return None
    if abs(c1) * q[0] > 0.10 * abs(vals[0]):  # a sizeable slope: k0 is not an extremum along this line
        return None
    rel = abs(c2 - c2_low) / abs(c2)
    if rel > max_rel:
        return None
    fit = EdgeFit(carrier, band, k0, _direction_label(bands, kp, k0, step, s, e), c2, HBAR2_OVER_2M0 / c2,
                  1.0 - rel, float(q[-1]), 3, list(q * step), list(vals))
    fit.uncertainty = abs(HBAR2_OVER_2M0 / c2 - HBAR2_OVER_2M0 / c2_low)
    fit.window_ev = float(np.max(np.abs(vals)))
    fit.slope = c1
    return fit


def fit_edge(bands, kp: KPath, rows: np.ndarray, band: int, k0: int, carrier: str, *,
             max_window_ev=0.60, max_rel_uncertainty=0.10) -> list[EdgeFit]:
    """Band-edge effective masses of ``band`` at path point ``k0``.

    Method: the curvature c2 = lim_{q->0} [E(k0+q) - E0]/q² is Richardson-
    extrapolated in q² from the three nearest path points (polynomial in q² of
    order 2; the order-1 value gives the uncertainty), |q| Cartesian in 1/Å.
    At a time-reversal-invariant k0 the band is even along any line through it,
    so each direction leaving k0 gives one mass; elsewhere the even part
    [E(k0+q) + E(k0-q)]/2 - E0 along the segment is used (one mass per line).
    m*/m0 = (ħ²/2m0)/c2.  A fit is rejected when the samples reach
    |E - E0| > ``max_window_ev``, the curvature has the wrong sign, or the
    relative uncertainty exceeds ``max_rel_uncertainty``.
    """
    kc = cartesian_k(bands)
    if kc is None:
        return []
    x0, e0 = kc[k0], rows[k0, band]
    trim = _is_trim(bands.kpoints[k0])

    def walk(step, s, e):
        idx, j = [], k0 + step
        while s <= j <= e and len(idx) < 3:
            if np.linalg.norm(kc[j] - (kc[idx[-1]] if idx else x0)) > 1e-10:
                idx.append(j)
            j += step
        return idx

    fits = []
    for s, e in _segment_of(kp, k0):
        lines = [(-1,), (1,)] if trim else [(-1, 1)]
        if not trim:
            a, b = walk(-1, s, e), walk(1, s, e)
            qa = np.array([np.linalg.norm(kc[i] - x0) for i in a]) if len(a) == 3 else None
            qb = np.array([np.linalg.norm(kc[i] - x0) for i in b]) if len(b) == 3 else None
            if qa is not None and qb is not None and np.max(np.abs(qa - qb)) > 1e-6 * max(1.0, qb.max()):
                fit = _two_sided_fit(bands, kp, rows, band, k0, carrier, a, b, qa, qb, s, e, e0,
                                     max_window_ev, max_rel_uncertainty)
                if fit is not None:
                    fits.append(fit)
                continue
            if qa is None or qb is None:
                # one-sided (path end): fit c1 q + c2 q^2 + c3 q^3 per direction
                for step, side in ((-1, a), (1, b)):
                    if len(side) < 3:
                        continue
                    fit = _one_sided_fit(bands, kp, rows, band, k0, carrier, side, step, s, e, kc, x0, e0,
                                         max_window_ev, max_rel_uncertainty)
                    if fit is not None:
                        fits.append(fit)
                continue
        for steps in lines:
            sides = [walk(st, s, e) for st in steps]
            if any(len(side) < 3 for side in sides):
                continue
            q = np.array([np.linalg.norm(kc[i] - x0) for i in sides[-1]])
            if len(sides) == 2:
                qa = np.array([np.linalg.norm(kc[i] - x0) for i in sides[0]])
                if np.max(np.abs(qa - q)) > 1e-6 * max(1.0, q.max()):
                    continue                       # unequal steps on the two sides
                vals = 0.5 * (rows[sides[0], band] + rows[sides[1], band]) - e0
                raw = np.concatenate([rows[sides[0], band], rows[sides[1], band]]) - e0
            else:
                vals = rows[sides[0], band] - e0
                raw = vals
            if float(np.max(np.abs(raw))) > max_window_ev:
                continue
            c2, c2_low = _richardson_c2(q, vals)
            sign_ok = (c2 > 0) if carrier == 'electron' else (c2 < 0)
            if not sign_ok or abs(c2) < 1e-6:
                continue
            rel = abs(c2 - c2_low) / abs(c2)
            if rel > max_rel_uncertainty:
                continue
            step = steps[-1]
            direction = (_direction_label(bands, kp, k0, step, s, e) if trim else _line_label(bands, kp, k0, s, e))
            ks = np.concatenate([-q[::-1], q]) if len(sides) == 2 else q * step
            es = (np.concatenate([rows[sides[0][::-1], band], rows[sides[1], band]]) - e0 if len(sides) == 2
                  else vals)
            fit = EdgeFit(carrier, band, k0, direction, c2, HBAR2_OVER_2M0 / c2, 1.0 - rel, float(q[-1]), 3,
                          list(ks), list(es))
            fit.uncertainty = abs(HBAR2_OVER_2M0 / c2 - HBAR2_OVER_2M0 / c2_low)
            fit.window_ev = float(np.max(np.abs(raw)))
            fits.append(fit)
    return fits


@dataclass
class Valley:
    kindex: int
    label: str
    energy_rel_ev: float


@dataclass
class PathAnalysis:
    edges: BandEdges
    kp: KPath
    channel: str
    rows: np.ndarray                # [k][band] of the analysed channel (eV, absolute)
    zero_ev: float                  # energy zero of the figures (VBM or E_F)
    zero_label: str                 # 'VBM' | 'F'
    vb: int | None                  # top valence band index (0-based), insulators only
    cb: int | None
    vbm_k: int | None
    cbm_k: int | None
    classification: str
    electron_fits: list = field(default_factory=list)
    hole_fits: list = field(default_factory=list)
    valleys: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    @property
    def is_metal(self):
        return self.vb is None

    def values(self) -> dict:
        out = {'classification': self.classification, 'energy_zero': self.zero_label,
               'energy_zero_ev': self.zero_ev, 'channel': self.channel,
               'gap_status_source': self.edges.status}
        if not self.is_metal:
            out.update({'gap_ev': self.edges.gap_ev, 'direct': self.edges.direct,
                        'vbm_ev': self.edges.vbm_ev, 'cbm_ev': self.edges.cbm_ev,
                        'vbm_band': self.vb + 1, 'cbm_band': self.cb + 1,
                        'vbm_k_label': self.k_label(self.vbm_k), 'cbm_k_label': self.k_label(self.cbm_k),
                        'direct_gap_ev': self.edges.direct_gap_ev,
                        'effective_mass_definition': 'band-edge m*/m0 = (hbar^2/2m0)/c2, c2 = lim (E-E0)/q^2 '
                                                     'by Richardson extrapolation along one path direction, '
                                                     '|q| Cartesian (1/Å)',
                        'electron_masses': [f.as_dict() for f in self.electron_fits],
                        'hole_masses': [f.as_dict() for f in self.hole_fits],
                        'conduction_valleys': [{'label': v.label, 'kindex': v.kindex, 'energy_rel_ev': v.energy_rel_ev}
                                               for v in self.valleys]})
        if self.notes:
            out['notes'] = list(self.notes)
        return out

    def k_label(self, k):
        if k is None:
            return None
        return _plain(_tick_label_at(self.kp, self.kp.x[k])) or f'k{k}'

    def k_name(self, bands, k):
        """Label of path point k itself (not the merged tick text)."""
        return interior_label(bands, self.kp, k) if k is not None else None


def _top_index(row, value, tol=1e-6):
    hits = [b for b, e in enumerate(row) if abs(e - value) <= tol]
    return hits[-1] if hits else None


def _degenerate(row, band, tol=DEGENERATE_EV) -> list[int]:
    return [b for b, e in enumerate(row) if abs(e - row[band]) <= tol]


def _fits_for_extremum(bands, kp, rows, k0, band, carrier) -> list[EdgeFit]:
    """Fits for every band degenerate with ``band`` at k0 (heavy/light holes at Γ...)."""
    out = []
    for b in _degenerate(rows[k0], band):
        out += fit_edge(bands, kp, rows, b, k0, carrier)
    # one fit per (band, direction); keep the best R²
    best = {}
    for f in out:
        key = (f.band, f.direction)
        if key not in best or f.r2 > best[key].r2:
            best[key] = f
    return sorted(best.values(), key=lambda f: (abs(f.mass), f.direction))


def analyse_path(bands, *, channel=None) -> PathAnalysis:
    """Gap, band edges, effective masses and conduction valleys of a band path.

    The classification follows :func:`band_edges` (occupations or E_F; a band
    spanning E_F makes it a metal).  Spin-polarised paths are analysed in the
    channel holding the VBM (``channel`` overrides).
    """
    edges = band_edges(bands)
    kp = kpath(bands)
    chans = channels_of(bands)
    name = channel or edges.vbm_channel or next(iter(chans))
    if name not in chans:
        raise InputError(f'spin channel {name!r} not in {list(chans)}')
    rows = np.asarray(chans[name], dtype=float)
    notes = []
    if edges.is_metal or edges.vbm_ev is None or edges.cbm_ev is None:
        ef = bands.fermi_energy
        cls = 'metal' if edges.is_metal else 'unclassified (no occupations or E_F)'
        if ef is None:
            notes.append('no Fermi energy: energies are absolute')
        return PathAnalysis(edges, kp, name, rows, float(ef) if ef is not None else 0.0, 'F', None, None, None, None,
                            cls, notes=notes)
    vbm_k = (edges.vbm_kindex or [int(np.argmax(rows.max(axis=1)))])[0]
    cbm_k = (edges.cbm_kindex or [0])[0]
    # pick the occurrence of the extremum k nearest the path centre for drawing; fits use every copy
    vb = _top_index(rows[vbm_k], edges.vbm_ev)
    cb = next((b for b, e in enumerate(rows[cbm_k]) if abs(e - edges.cbm_ev) <= 1e-6), None)
    if vb is None or cb is None:
        notes.append('band edges reported by the parser are not on this channel')
        return PathAnalysis(edges, kp, name, rows, float(edges.vbm_ev), 'VBM', None, None, None, None,
                            'semiconductor (edges off channel)', notes=notes)
    gap = float(edges.gap_ev)
    kind = 'direct' if edges.direct else 'indirect'
    cls = f'{kind}-gap ' + ('insulator' if gap >= 3.0 else 'semiconductor')
    electron_fits, hole_fits = [], []
    for k in edges.cbm_kindex or [cbm_k]:
        electron_fits += _fits_for_extremum(bands, kp, rows, k, cb, 'electron')
    for k in edges.vbm_kindex or [vbm_k]:
        hole_fits += _fits_for_extremum(bands, kp, rows, k, vb, 'hole')
    electron_fits, hole_fits = _unique_fits(electron_fits), _unique_fits(hole_fits)
    if not electron_fits:
        notes.append('no electron effective mass: < 3 path points per side, |E-E0| > 0.6 eV or uncertainty > 10 %')
    if not hole_fits:
        notes.append('no hole effective mass: < 3 path points per side, |E-E0| > 0.6 eV or uncertainty > 10 %')
    pa = PathAnalysis(edges, kp, name, rows, float(edges.vbm_ev), 'VBM', vb, cb, vbm_k, cbm_k, cls,
                      electron_fits, hole_fits, notes=notes)
    pa.valleys = conduction_valleys(pa)
    return pa


def _unique_fits(fits):
    """Equivalent copies of one extremum (Γ at both path ends) give identical fits: keep one per band+direction."""
    seen, out = set(), []
    for f in sorted(fits, key=lambda f: f.uncertainty):
        key = (f.band, f.direction, round(f.mass, 4))
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    return sorted(out, key=lambda f: (f.band, f.direction))


def conduction_valleys(pa: PathAnalysis, *, max_valleys=3) -> list[Valley]:
    """Local minima of the lowest conduction band at labelled path points, lowest first.

    Only points carrying a path label are named (no lattice-type guessing); the
    CBM itself is the first entry.
    """
    if pa.cb is None:
        return []
    band = pa.rows[:, pa.cb] - pa.zero_ev
    out, seen = [], set()
    for tx, name in pa.kp.ticks:
        idx = [i for i, x in enumerate(pa.kp.x) if abs(x - tx) < 1e-9]
        if not idx:
            continue
        label = _plain(name)
        e = min(float(band[i]) for i in idx)
        i0 = min(idx, key=lambda i: band[i])
        lo, hi = max(0, i0 - 3), min(len(band), i0 + 4)
        if e > float(np.min(band[lo:hi])) + 1e-6:        # not a local minimum along the path
            continue
        if label in seen:
            continue
        seen.add(label)
        out.append(Valley(i0, label, e))
    out.sort(key=lambda v: v.energy_rel_ev)
    return out[:max_valleys]


# --- orbital character -----------------------------------------------------------
def orbital_fraction(proj: ProjectionSet, channel: str, which='s') -> tuple[np.ndarray, np.ndarray]:
    """(fraction of ``which``-character, total sphere weight) per [k][band].

    The fraction is w_which / w_total of the PAW-sphere weights; the total is
    returned so the caller can fade states with little weight inside the spheres.
    """
    w = np.asarray(proj.channels[channel], dtype=float)            # k, band, atom, orbital
    cols = [o for o, orb in enumerate(proj.orbitals) if (orbital_l(orb) or canonical_orbital(orb)) == which]
    total = w.sum(axis=(2, 3))
    part = w[..., cols].sum(axis=(2, 3)) if cols else np.zeros_like(total)
    with np.errstate(invalid='ignore', divide='ignore'):
        frac = np.where(total > 1e-8, part / np.maximum(total, 1e-12), 0.0)
    return np.clip(frac, 0.0, 1.0), total


def element_l_channels(proj: ProjectionSet, channel: str, *, min_share=0.025):
    """[(label, element, l, weights[k][band])] for element+l groups holding >= ``min_share`` of the weight.

    Smaller groups are summed into one 'other' entry when it reaches 1 %.
    """
    w = np.asarray(proj.channels[channel], dtype=float)
    species = proj.species or [f'atom{a + 1}' for a in range(proj.nions)]
    groups = []
    for el in dict.fromkeys(species):
        atoms = [a for a, s in enumerate(species) if s == el]
        for l in ('s', 'p', 'd', 'f'):
            cols = [o for o, orb in enumerate(proj.orbitals) if (orbital_l(orb) or canonical_orbital(orb)) == l]
            if not cols:
                continue
            groups.append((el, l, w[:, :, atoms][:, :, :, cols].sum(axis=(2, 3))))
    total = sum(float(g[2].sum()) for g in groups)
    if total <= 0:
        return []
    out, other = [], None
    for el, l, arr in groups:
        share = float(arr.sum()) / total
        if share >= min_share:
            out.append((f'{el} ${l}$', el, l, arr))
        elif share > 1e-6:
            other = arr if other is None else other + arr
    if other is not None and float(other.sum()) / total >= 0.01:
        out.append(('other', None, None, other))
    return out


def dominant_character(proj: ProjectionSet, channel: str, rows: np.ndarray, zero_ev: float,
                       window=(-6.0, 6.0)) -> str:
    """The l channel whose state fraction varies most among states inside ``window`` (eV from the zero).

    The orbital-character colour map shows this fraction: s for sp semiconductors
    (s-like conduction minimum), d for transition metals, ...
    """
    w = np.asarray(proj.channels[channel], dtype=float)
    total = w.sum(axis=(2, 3))
    nb = min(rows.shape[1], total.shape[1])
    inside = ((rows[:, :nb] - zero_ev) >= window[0]) & ((rows[:, :nb] - zero_ev) <= window[1]) & (total[:, :nb] > 1e-6)
    spread = {}
    for l in ('s', 'p', 'd', 'f'):
        cols = [o for o, orb in enumerate(proj.orbitals) if (orbital_l(orb) or canonical_orbital(orb)) == l]
        if not cols:
            continue
        frac = w[:, :nb][..., cols].sum(axis=(2, 3)) / np.maximum(total[:, :nb], 1e-12)
        vals = frac[inside]
        if vals.size:
            spread[l] = float(vals.std())
    if not spread:
        return 's'
    top = max(spread.values())
    # s and p fractions of sp states are complementary (equal spread): keep the
    # lower l, which marks the s-like conduction edge of sp semiconductors
    return next(l for l in ('s', 'p', 'd', 'f') if l in spread and spread[l] >= 0.95 * top)


# --- run conditions and engine comparison -----------------------------------------
def run_conditions(source, bands=None) -> dict:
    """Engine, cutoff, PAW datasets, XC and species recorded by a band run (Sparse-DFT JSON or vasprun.xml)."""
    from pathlib import Path
    from .common import read_json
    from ..plotting.style import PRODUCT
    path = Path(source)
    if path.is_dir():
        from .electronic import find_band_file
        path = find_band_file(path)
    if path.suffix.lower() == '.xml':
        import xml.etree.ElementTree as ET
        root = ET.parse(path).getroot()
        incar = {i.get('name'): (i.text or '').strip() for i in root.find('incar').findall('i')} if root.find('incar') is not None else {}
        gen = {i.get('name'): (i.text or '').strip() for i in root.find('generator').findall('i')} if root.find('generator') is not None else {}
        types = root.find('.//atominfo/array[@name="atomtypes"]')
        titles = {}
        if types is not None:
            for rc in types.findall('.//rc'):
                c = [x.text.strip() for x in rc.findall('c')]
                if len(c) >= 5:
                    titles[c[1]] = c[4]
        species = [rc.find('c').text.strip() for rc in root.findall('.//atominfo/array[@name="atoms"]/set/rc')]
        gga = incar.get('GGA', '').strip().upper()
        xc = 'PBE' if titles and all(t.startswith('PAW_PBE') for t in titles.values()) and gga in ('', '--', 'PE') else (gga or 'unknown')
        encut = incar.get('ENCUT')
        return {'engine': 'VASP', 'version': gen.get('version'), 'file': str(path), 'species': species,
                'cutoff_ev': float(encut) if encut else None, 'paw_titles': titles, 'xc': xc,
                'nbands': int(incar['NBANDS']) if incar.get('NBANDS') else None,
                'ispin': int(incar.get('ISPIN', '1') or 1), 'lorbit': incar.get('LORBIT'),
                'smearing': {'ismear': incar.get('ISMEAR'), 'sigma': incar.get('SIGMA')}}
    data = read_json(path)
    prov = data.get('provenance') or {}
    titles = {}
    for entry in prov.get('paw') or []:
        sel = entry.get('selected') or {}
        if entry.get('element') and sel.get('title'):
            titles[entry['element']] = sel['title']
    return {'engine': PRODUCT, 'file': str(path), 'species': list(data.get('projection_atom_symbols') or []),
            'cutoff_ev': float(prov['cutoff_ev']) if prov.get('cutoff_ev') is not None else None,
            'paw_titles': titles, 'xc': 'PBE', 'atomic_units': prov.get('atomic_units'),
            'density_cutoff_ev': prov.get('density_cutoff_ev') or (data.get('metadata') or {}).get('density_cutoff_ev'),
            'nbands': len(data['eigenvalues'][0]) if data.get('eigenvalues') else None,
            'spin_channels': prov.get('spin_channels'), 'sparsity': (prov.get('sparsity') or {}).get('mode')}


def compare_conditions(cand: dict, ref: dict, cand_bands, ref_bands) -> dict:
    """Checks that two band runs describe the same calculation (see module docstring)."""
    checks = {}
    if cand.get('species') and ref.get('species'):
        checks['species'] = cand['species'] == ref['species']
    if cand.get('cutoff_ev') is not None and ref.get('cutoff_ev') is not None:
        checks['cutoff_ev'] = abs(cand['cutoff_ev'] - ref['cutoff_ev']) < 1e-6
    if cand.get('paw_titles') and ref.get('paw_titles'):
        checks['paw_titles'] = cand['paw_titles'] == ref['paw_titles']
    checks['xc'] = cand.get('xc') == ref.get('xc')
    pairs = match_kpoints(cand_bands, ref_bands)
    checks['kpoints'] = len(pairs) >= 0.9 * min(cand_bands.nkpoints, ref_bands.nkpoints) and len(pairs) > 0
    cb, rb = reciprocal_matrix(cand_bands), reciprocal_matrix(ref_bands)
    if cb is not None and rb is not None:
        checks['reciprocal_basis'] = bool(np.max(np.abs(cb - rb)) < 1e-5 * max(1.0, float(np.max(np.abs(rb)))))
    differences = [k for k, ok in checks.items() if not ok]
    return {'matched': not differences and checks.get('kpoints', False), 'checks': checks,
            'differences': differences, 'shared_kpoints': len(pairs),
            'candidate': {k: cand.get(k) for k in ('engine', 'cutoff_ev', 'paw_titles', 'xc', 'atomic_units', 'file')},
            'reference': {k: ref.get(k) for k in ('engine', 'version', 'cutoff_ev', 'paw_titles', 'xc', 'file')}}


def _channel_pairs(cand, ref):
    cc, rc = channels_of(cand), channels_of(ref)
    if len(cc) == 1 and len(rc) == 1:
        return [(next(iter(cc)), next(iter(rc)))]
    return [(n, n) for n in ('up', 'down') if n in cc and n in rc]


def compared_band_count(cand, ref, top_excluded=2) -> int:
    """Bands compared between two runs: all when the counts agree, else min - ``top_excluded`` (Davidson tail)."""
    if cand.nbands == ref.nbands:
        return cand.nbands
    return max(1, min(cand.nbands, ref.nbands) - top_excluded)


def engine_residuals(cand, ref, *, align='none', zero_ev=0.0) -> dict:
    """E_cand - E_ref (meV) at shared k points over the compared bands.

    ``align='none'`` keeps both energy zeros (PAW codes share the zero of the
    pseudopotential; the VASP NSCF path recomputes E_F from the path, so a
    Fermi alignment would add an artificial shift); 'vbm' removes the VBM offset.
    """
    pairs = match_kpoints(cand, ref)
    if not pairs:
        raise InputError('the two band runs share no k point')
    nb = compared_band_count(cand, ref)
    shift = 0.0
    if align == 'vbm':
        if cand.vbm is None or ref.vbm is None:
            raise InputError("align='vbm' needs a VBM in both runs")
        shift = float(cand.vbm) - float(ref.vbm)
    cc, rc = channels_of(cand), channels_of(ref)
    energy, delta, band_id, chan = [], [], [], []
    for cn, rn in _channel_pairs(cand, ref):
        a, b = np.asarray(cc[cn], dtype=float), np.asarray(rc[rn], dtype=float)
        for i, j in pairs:
            for n in range(nb):
                energy.append(b[j, n] - zero_ev)
                delta.append((a[i, n] - b[j, n] - shift) * 1000.0)
                band_id.append(n + 1)
                chan.append(cn)
    d = np.asarray(delta)
    return {'energy_rel_ev': energy, 'delta_mev': delta, 'band': band_id, 'channel': chan,
            'n_kpoints': len(pairs), 'n_bands': nb, 'n_channels': len(_channel_pairs(cand, ref)),
            'mae_mev': float(np.mean(np.abs(d))), 'rmse_mev': float(np.sqrt(np.mean(d ** 2))),
            'max_abs_mev': float(np.max(np.abs(d))), 'mean_mev': float(np.mean(d)), 'align': align,
            'shift_ev': shift,
            'definition': 'Delta E = E_candidate - E_reference at k points equal modulo G (1e-5), bands 1..n_bands, '
                          + ('no energy shift (shared PAW energy zero)' if align == 'none' else 'VBM offset removed')}


def projection_parity(cand, cand_proj: ProjectionSet, ref, ref_proj: ProjectionSet, *,
                      near_ev=NEAR_DEGENERATE_EV) -> dict:
    """Per-(k, band, atom, orbital) lm weights of two runs at shared k points.

    R² is measured against the identity line y = x (not a regression).  States
    within ``near_ev`` of a neighbouring band of the reference are flagged: a
    near-degenerate pair can exchange weight while the pair sum agrees, so the
    maximum is reported for single states, for non-degenerate states and for
    pair-summed clusters separately.
    """
    if cand_proj.nions != ref_proj.nions:
        raise InputError('projection parity needs the same number of atoms in both runs')
    shared = [o for o in cand_proj.orbitals if o in ref_proj.orbitals]
    if not shared:
        raise InputError('the two runs share no projection orbital')
    ci = [cand_proj.orbitals.index(o) for o in shared]
    ri = [ref_proj.orbitals.index(o) for o in shared]
    pairs = match_kpoints(cand, ref)
    nb = compared_band_count(cand, ref)
    l_of = [orbital_l(o) or canonical_orbital(o) for o in shared]
    xs, ys, ls, near_flags = [], [], [], []
    cluster_dev = []
    rc = channels_of(ref)
    chan_pairs = [(c, r) for c, r in _channel_pairs(cand, ref) if c in cand_proj.channels and r in ref_proj.channels]
    if not chan_pairs and len(cand_proj.channels) == 1 and len(ref_proj.channels) == 1:
        chan_pairs = [(next(iter(cand_proj.channels)), next(iter(ref_proj.channels)))]
    for cn, rn in chan_pairs:
        A = np.asarray(cand_proj.channels[cn], dtype=float)[..., ci]
        B = np.asarray(ref_proj.channels[rn], dtype=float)[..., ri]
        E = np.asarray(rc.get(rn) or next(iter(rc.values())), dtype=float)
        # neighbours are looked up in every band both runs solved, so a state at the
        # top of the window whose partner lies just above it is still flagged
        nall = min(A.shape[1], B.shape[1], E.shape[1])
        for i, j in pairs:
            e = E[j, :nall]
            close = np.abs(np.diff(e)) < near_ev
            near = np.zeros(nall, dtype=bool)
            near[1:] |= close
            near[:-1] |= close
            for n in range(nb):
                xs.append(B[j, n].ravel()); ys.append(A[i, n].ravel())
                ls.append(np.tile(l_of, cand_proj.nions)); near_flags.append(np.full(A[i, n].size, near[n]))
            # clusters of consecutive near-degenerate bands touching the window: summed weights
            n = 0
            while n < nb:
                m = n
                while m + 1 < nall and abs(e[m + 1] - e[m]) < near_ev:
                    m += 1
                if m > n:
                    cluster_dev.append(float(np.max(np.abs(A[i, n:m + 1].sum(axis=0) - B[j, n:m + 1].sum(axis=0)))))
                n = m + 1
    x, y = np.concatenate(xs), np.concatenate(ys)
    lab, near = np.concatenate(ls), np.concatenate(near_flags)
    dev = np.abs(y - x)
    ss_tot = float(np.sum((x - x.mean()) ** 2))
    r2 = 1.0 - float(np.sum((y - x) ** 2)) / ss_tot if ss_tot > 0 else float('nan')
    single = dev[~near]
    return {'reference': x.tolist(), 'candidate': y.tolist(), 'l': lab.tolist(), 'near_degenerate': near.tolist(),
            'n_values': int(x.size), 'n_kpoints': len(pairs), 'n_bands': nb, 'orbitals': shared,
            'r2_identity': r2, 'max_abs_dw': float(dev.max()), 'mae_dw': float(dev.mean()),
            'max_abs_dw_non_degenerate': float(single.max()) if single.size else None,
            'max_abs_dw_pair_summed': max(cluster_dev) if cluster_dev else None,
            'near_degenerate_ev': near_ev, 'near_degenerate_values': int(near.sum()),
            'definition': 'lm PAW-sphere weights per (k, band, atom, orbital); R^2 = 1 - sum (w_c - w_r)^2 / '
                          'sum (w_r - mean w_r)^2 against y = x; near-degenerate: |E_n - E_n±1| < '
                          f'{near_ev * 1000:.0f} meV in the reference'}


# --- uniform-mesh analyses -----------------------------------------------------------
def _grid_axis(values, tol=1e-5):
    vals = sorted({round(((float(v) + 0.5) % 1.0) - 0.5, 6) for v in values})
    return [v for i, v in enumerate(vals) if i == 0 or v - vals[i - 1] > tol]


def bz_plane(mesh, *, quantity='direct', vb=None, cb=None, zero_ev=0.0, channel=None, upsample=8) -> dict:
    """A quantity on the plane k3 = 0 of a uniform Γ-containing mesh.

    ``quantity`` 'direct': E_cb - E_vb per k (eV); 'fermi': E_n - E_F for the
    bands crossing E_F on the plane (for metals).  Values are the mesh values;
    a periodic trigonometric interpolation (zero-padded FFT, factor
    ``upsample``) is returned separately for drawing and labelled as such.
    The reported minimum is a sampled value, never an interpolated one.
    """
    k = np.asarray(mesh.kpoints, dtype=float)[:, :3]
    folded = ((k + 0.5) % 1.0) - 0.5
    on = np.abs(folded[:, 2]) < 1e-5
    if on.sum() < 9:
        raise InputError('the mesh has fewer than 9 points on the k3 = 0 plane')
    a1, a2 = _grid_axis(folded[on, 0]), _grid_axis(folded[on, 1])
    n1, n2 = len(a1), len(a2)
    if n1 * n2 != int(on.sum()):
        raise InputError('the k3 = 0 points do not form a full uniform grid (symmetry-reduced mesh?)')
    step1, step2 = 1.0 / n1, 1.0 / n2
    if max(abs(np.diff(a1) - step1).max(initial=0), abs(np.diff(a2) - step2).max(initial=0)) > 1e-4:
        raise InputError('the k3 = 0 points are not equally spaced')
    chans = channels_of(mesh)
    name = channel or next(iter(chans))
    E = np.asarray(chans[name], dtype=float)
    idx = np.where(on)[0]
    i1 = np.rint((folded[idx, 0] - a1[0]) / step1).astype(int)
    i2 = np.rint((folded[idx, 1] - a2[0]) / step2).astype(int)
    fields, labels = [], []
    if quantity == 'direct':
        if vb is None or cb is None:
            raise InputError('a direct-transition map needs the valence and conduction band indices')
        g = np.empty((n1, n2)); g[i1, i2] = E[idx, cb] - E[idx, vb]
        fields.append(g); labels.append(f'E{cb + 1} - E{vb + 1}')
    else:
        ef = float(mesh.fermi_energy) if mesh.fermi_energy is not None else zero_ev
        for b in range(E.shape[1]):
            col = E[idx, b] - ef
            if col.min() < 0 < col.max():
                g = np.empty((n1, n2)); g[i1, i2] = col
                fields.append(g); labels.append(f'band {b + 1}')
    if not fields:
        raise InputError('no band crosses E_F on the k3 = 0 plane')
    b = reciprocal_matrix(mesh)

    def fourier(g, f1, f2):
        # Trigonometric interpolation through the samples: frequencies m1, m2 in
        # [-n/2, n/2) with the Nyquist term split symmetrically (real, periodic).
        c = np.fft.fft2(g) / g.size
        m1 = np.fft.fftfreq(n1, 1.0 / n1)
        m2 = np.fft.fftfreq(n2, 1.0 / n2)
        w1 = np.where(np.abs(m1) == n1 / 2, 0.5, 1.0)
        w2 = np.where(np.abs(m2) == n2 / 2, 0.5, 1.0)
        x1 = (np.asarray(f1)[:, None] - a1[0]) * n1      # grid-index coordinate of the fine points
        x2 = (np.asarray(f2)[:, None] - a2[0]) * n2
        out = np.zeros((len(f1), len(f2)))
        for s1 in (1.0, -1.0):          # Nyquist mode: average of +n/2 and -n/2 phases
            for s2 in (1.0, -1.0):
                mm1 = np.where(np.abs(m1) == n1 / 2, s1 * np.abs(m1), m1)
                mm2 = np.where(np.abs(m2) == n2 / 2, s2 * np.abs(m2), m2)
                e1 = np.exp(2j * np.pi * x1 * mm1[None, :] / n1)          # fine1 x n1
                e2 = np.exp(2j * np.pi * x2 * mm2[None, :] / n2)          # fine2 x n2
                part = e1 @ c @ e2.T
                out += 0.25 * np.real(part)
        return out
    f1 = a1[0] + np.arange(n1 * upsample + 1) * step1 / upsample
    f2 = a2[0] + np.arange(n2 * upsample + 1) * step2 / upsample
    fine = [fourier(g, f1, f2) for g in fields]
    sampled_min = None
    if quantity == 'direct':
        g = fields[0]
        m = np.unravel_index(int(np.argmin(g)), g.shape)
        sampled_min = {'value_ev': float(g[m]), 'k1': a1[m[0]], 'k2': a2[m[1]]}
    return {'quantity': quantity, 'channel': name, 'k1': a1, 'k2': a2, 'values': [g.tolist() for g in fields],
            'labels': labels, 'fine_k1': f1.tolist(), 'fine_k2': f2.tolist(), 'fine': [g.tolist() for g in fine],
            'grid': [n1, n2], 'n_points': int(on.sum()), 'sampled_min': sampled_min,
            'reciprocal_basis': b.tolist() if b is not None else None,
            'interpolation': f'periodic trigonometric interpolation (x{upsample}) through {int(on.sum())} mesh points'}


def band_composition(proj: ProjectionSet, bands, channel: str, *, nbands=None) -> dict:
    """BZ-weighted element+l share (%) of the PAW-sphere weight per band.

    Uses the k weights of a uniform mesh (equal weights on a path, flagged as a
    path average).  The share is normalised to the weight inside the spheres;
    the sphere fraction itself (sum of lm weights, < 1 for delocalised states)
    is returned per band so the normalisation is visible.
    """
    groups = element_l_channels(proj, channel, min_share=0.0)
    w = np.asarray(bands.kweights, dtype=float) if bands.kweights else np.ones(bands.nkpoints)
    w = w / w.sum()
    nb = min(nbands or 10 ** 9, *(g[3].shape[1] for g in groups))
    totals = np.zeros(nb)
    per = []
    for label, el, l, arr in groups:
        v = (arr[:, :nb] * w[:, None]).sum(axis=0)
        per.append((label, el, l, v))
        totals += v
    keep = [(lab, el, l, v) for lab, el, l, v in per if v.sum() / max(totals.sum(), 1e-12) >= 0.01]
    rest = sum((v for lab, el, l, v in per if v.sum() / max(totals.sum(), 1e-12) < 0.01), np.zeros(nb))
    if rest.sum() > 0:
        keep.append(('other', None, None, rest))
    shares = {lab: (100.0 * v / np.maximum(totals, 1e-12)).tolist() for lab, el, l, v in keep}
    return {'bands': list(range(1, nb + 1)), 'shares_percent': shares, 'order': [k[0] for k in keep],
            'sphere_weight': totals.tolist(), 'k_average': 'mesh k weights' if bands.kweights else 'path average',
            'definition': 'share = sum_k w_k sum_(atoms of element, m of l) w_lm / sum over all groups; '
                          'w_lm = PAW-sphere lm weights'}


def fermi_crossings(pa: PathAnalysis) -> list[int]:
    """Bands of the analysed channel that run from below to above the energy zero."""
    out = []
    for b in range(pa.rows.shape[1]):
        col = pa.rows[:, b] - pa.zero_ev
        if col.min() < 0 < col.max():
            out.append(b)
    return out
