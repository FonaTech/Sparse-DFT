"""Automated layout checks for rendered figures.

``layout_report(fig)`` draws the figure with Agg and reports text that
overlaps other text, leaves the figure, sits on data marks, uses a font size
outside the journal range, or needs a glyph the chosen fonts lack.
Used by the tests and available to the GUI as a pre-export lint.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.text import Annotation, Text

from .export import _ensure_canvas
from .style import style_of, use_spec

FONT_RANGE_PT = (5.0, 8.0)
DATA_ZORDER = 2.0


def _drawn_tick_labels(axis):
    try:
        ticks = axis._update_ticks()          # ticks actually drawn (inside the view interval)
    except Exception:                          # pragma: no cover - private API fallback
        ticks = axis.get_major_ticks()
    out = []
    for t in ticks:
        for label in (t.label1, t.label2):
            if label.get_visible() and label.get_text().strip():
                out.append(label)
    return out


def _texts(fig):
    texts = []
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        if ax.axison:                          # axis-off axes (legend/header holders) draw no ticks
            texts += _drawn_tick_labels(ax.xaxis) + _drawn_tick_labels(ax.yaxis)
        for t in (ax.xaxis.label, ax.yaxis.label, ax.title, ax._left_title, ax._right_title):
            if t.get_visible() and t.get_text().strip():
                texts.append(t)
        for child in ax.get_children():
            if isinstance(child, Text) and child.get_visible() and child.get_text().strip():
                texts.append(child)
            if isinstance(child, Legend):
                texts += [t for t in child.get_texts() if t.get_visible() and t.get_text().strip()]
    for t in fig.texts:
        if t.get_visible() and t.get_text().strip():
            texts.append(t)
    for t in (getattr(fig, '_supxlabel', None), getattr(fig, '_supylabel', None), getattr(fig, '_suptitle', None)):
        if t is not None and t.get_visible() and t.get_text().strip():
            texts.append(t)
    seen, unique = set(), []
    for t in texts:
        if id(t) not in seen:
            seen.add(id(t)); unique.append(t)
    return unique


def _data_points(ax):
    """Display coordinates of data marks (zorder >= 2), segments densely sampled."""
    pts = []
    def sample(xy, closed=False):
        xy = np.asarray(xy, dtype=float)
        if xy.ndim != 2 or len(xy) == 0:
            return
        xy = xy[np.all(np.isfinite(xy), axis=1)]
        if len(xy) == 1:
            pts.append(xy); return
        for a, b in zip(xy[:-1], xy[1:]):
            n = int(min(60, max(2, np.hypot(*(b - a)) / 2.0)))
            pts.append(a + (b - a) * np.linspace(0, 1, n)[:, None])
    for line in ax.lines:
        if line.get_zorder() < DATA_ZORDER or not line.get_visible():
            continue
        disp = line.get_transform().transform(line.get_xydata())
        if line.get_linestyle() in ('None', 'none', '', ' '):
            pts.append(disp[np.all(np.isfinite(disp), axis=1)])
        else:
            sample(disp)
    for coll in ax.collections:
        if coll.get_zorder() < DATA_ZORDER or not coll.get_visible():
            continue
        if isinstance(coll, LineCollection):
            tr = coll.get_transform()
            for seg in coll.get_segments():
                sample(tr.transform(seg))
        elif isinstance(coll, PathCollection):
            offs = coll.get_offsets()
            if len(offs):
                pts.append(coll.get_offset_transform().transform(offs))
    if not pts:
        return np.zeros((0, 2))
    return np.concatenate([p for p in pts if len(p)])


def _inside(points, bbox, pad=0.5):
    if len(points) == 0:
        return 0
    x, y = points[:, 0], points[:, 1]
    m = (x > bbox.x0 - pad) & (x < bbox.x1 + pad) & (y > bbox.y0 - pad) & (y < bbox.y1 + pad)
    return int(m.sum())


def _label_box(t, renderer):
    """Extent of the text itself; an annotation's leader line is meant to reach the data it labels."""
    return Text.get_window_extent(t, renderer) if isinstance(t, Annotation) else t.get_window_extent(renderer)


def _name(t):
    text = t.get_text().replace('\n', ' ')
    return text if len(text) <= 40 else text[:37] + '...'


def layout_report(fig, *, overlap_px=1.5) -> dict:
    """Return {'issues': [...], 'texts': n, 'size_mm': (w, h)} for ``fig``."""
    _ensure_canvas(fig)
    issues = []
    with use_spec(style_of(fig)):
        # Missing glyphs are reported two ways: backend warnings for plain
        # text, and log records from matplotlib._mathtext for mixed labels.
        records = []
        handler = logging.Handler()
        handler.emit = records.append
        mpl_logger = logging.getLogger('matplotlib')
        mpl_logger.addHandler(handler)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                fig.canvas.draw()
        finally:
            mpl_logger.removeHandler(handler)
        renderer = fig.canvas.get_renderer()
        messages = [str(w.message) for w in caught] + [r.getMessage() for r in records]
        for msg in dict.fromkeys(messages):
            low = msg.lower()
            if 'glyph' in low and ('missing' in low or 'does not have' in low):
                issues.append(f'missing glyph: {msg}')
        texts = _texts(fig)
        boxes = [_label_box(t, renderer) for t in texts]
        fb = fig.bbox
        for t, b in zip(texts, boxes):
            if b.x0 < fb.x0 - 1 or b.y0 < fb.y0 - 1 or b.x1 > fb.x1 + 1 or b.y1 > fb.y1 + 1:
                issues.append(f'text outside figure: {_name(t)!r}')
            size = t.get_fontsize()
            if not FONT_RANGE_PT[0] - 1e-6 <= size <= FONT_RANGE_PT[1] + 1e-6:
                issues.append(f'font size {size:g} pt outside {FONT_RANGE_PT}: {_name(t)!r}')
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                a, b = boxes[i], boxes[j]
                w = min(a.x1, b.x1) - max(a.x0, b.x0)
                h = min(a.y1, b.y1) - max(a.y0, b.y0)
                if w > overlap_px and h > overlap_px:
                    issues.append(f'text overlap: {_name(texts[i])!r} / {_name(texts[j])!r}')
        for ax in fig.axes:
            if not ax.get_visible():
                continue
            pts = _data_points(ax)
            # Artists are clipped to the axes: samples outside it (bands beyond the
            # energy window) are not drawn and cannot collide with a legend above it.
            if len(pts):
                box = ax.bbox
                pts = pts[(pts[:, 0] >= box.x0 - 0.5) & (pts[:, 0] <= box.x1 + 0.5) &
                          (pts[:, 1] >= box.y0 - 0.5) & (pts[:, 1] <= box.y1 + 0.5)]
            if not len(pts):
                continue
            leg = ax.get_legend()
            if leg is not None and leg.get_visible():
                n = _inside(pts, leg.get_window_extent(renderer))
                if n:
                    issues.append(f'legend covers {n} data samples in axes {ax.get_label() or id(ax)}')
            axbox = ax.bbox
            for child in ax.texts:
                if not child.get_visible() or not child.get_text().strip():
                    continue
                b = _label_box(child, renderer)
                if b.x1 < axbox.x0 or b.x0 > axbox.x1 or b.y1 < axbox.y0 or b.y0 > axbox.y1:
                    continue                    # outside the plotting area
                n = _inside(pts, b, pad=0.0)
                if n > 2:
                    issues.append(f'note {_name(child)!r} covers {n} data samples')
    w, h = fig.get_size_inches() * 25.4
    return {'issues': issues, 'texts': len(texts), 'size_mm': (round(float(w), 1), round(float(h), 1))}


def describe(fig) -> dict:
    """Plain-text description of a figure's axes (labels, limits, ticks, legend)."""
    _ensure_canvas(fig)
    with use_spec(style_of(fig)):
        fig.canvas.draw()
    out = []
    for ax in fig.axes:
        leg = ax.get_legend()
        out.append({
            'label': ax.get_label(), 'xlabel': ax.get_xlabel(), 'ylabel': ax.get_ylabel(),
            'xlim': [round(float(v), 6) for v in ax.get_xlim()], 'ylim': [round(float(v), 6) for v in ax.get_ylim()],
            'xscale': ax.get_xscale(), 'yscale': ax.get_yscale(),
            'xticks': [t.get_text() for t in _drawn_tick_labels(ax.xaxis)],
            'yticks': [t.get_text() for t in _drawn_tick_labels(ax.yaxis)],
            'legend': [t.get_text() for t in leg.get_texts()] if leg else [],
            'texts': [t.get_text() for t in ax.texts if t.get_visible()],
            'lines': len(ax.lines), 'collections': len(ax.collections), 'patches': len(ax.patches)})
    return {'axes': out, 'figtexts': [t.get_text() for t in fig.texts]}
