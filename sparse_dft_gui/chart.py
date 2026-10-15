"""Live SCF convergence chart fed by ``*.iterations.jsonl`` records."""
from __future__ import annotations

import math
from pathlib import Path

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from sparse_dft.runner import IterationTail, latest_iteration_log

from . import fonts, theme
from .i18n import translator, tr

SERIES = (('density_residual', 'chart.density'), ('onecenter_density_residual', 'chart.onecenter'),
          ('eigensolver_max_residual_hartree', 'chart.eigen'))
MOMENT_KEYS = ('output_magnetic_moment_muB', 'magnetic_moment_muB', 'total_magnetic_moment_muB')


def _canvas_classes():
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    return FigureCanvasQTAgg, Figure


class ConvergenceChart(QWidget):
    """Log-scale residuals versus iteration; magnetic moment on a twin axis for spin runs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.records: list[dict] = []
        self.theme_name = 'light'
        self.source: Path | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        head = QHBoxLayout()
        self.title = QLabel(self)
        self.title.setProperty('role', 'section')
        self.caption = QLabel(self)
        self.caption.setProperty('role', 'muted')
        head.addWidget(self.title)
        head.addStretch(1)
        head.addWidget(self.caption)
        layout.addLayout(head)
        self.canvas = None
        try:
            Canvas, Figure = _canvas_classes()
            self.figure = Figure(figsize=(4.2, 2.6), dpi=100, layout='constrained')
            self.canvas = Canvas(self.figure)
            self.canvas.setMinimumHeight(190)
            layout.addWidget(self.canvas, 1)
        except Exception as error:  # noqa: BLE001 - matplotlib missing: keep a text summary
            self.figure = None
            self.fallback = QLabel(f'matplotlib unavailable: {error}', self)
            layout.addWidget(self.fallback, 1)
        self._redraw = QTimer(self)
        self._redraw.setSingleShot(True)
        self._redraw.setInterval(150)
        self._redraw.timeout.connect(self.draw)
        self.retranslate()

    # ---- data
    def clear(self, source=None):
        self.records = []
        self.source = Path(source) if source else None
        self.schedule()

    def add_records(self, records):
        if records:
            self.records.extend(records)
            self.schedule()

    def schedule(self):
        self._redraw.start()

    def retranslate(self):
        self.title.setText(tr('chart.title'))
        self._update_caption()
        self.schedule()

    def _update_caption(self):
        if self.records:
            self.caption.setText(tr('chart.records', n=len(self.records)))
        else:
            self.caption.setText(tr('chart.waiting'))

    def set_theme(self, name):
        self.theme_name = name
        self.schedule()

    # ---- drawing
    def draw(self):
        self._update_caption()
        if self.figure is None:
            return
        p = theme.palette(self.theme_name)
        families = fonts.mpl_families(translator().language)
        fig = self.figure
        fig.clear()
        fig.set_facecolor(p['plot_bg'])
        ax = fig.add_subplot(111)
        ax.set_facecolor(p['plot_bg'])
        for spine in ax.spines.values():
            spine.set_color(p['border_strong'])
        ax.tick_params(colors=p['muted'], labelsize=8)
        ax.grid(True, which='major', color=p['grid'], linewidth=0.6)
        font = {'family': families, 'size': 9, 'color': p['text']}
        ax.set_xlabel(tr('chart.iteration'), fontdict=font)
        ax.set_ylabel(tr('chart.residual'), fontdict=font)
        iterations = [r.get('iteration', i + 1) for i, r in enumerate(self.records)]
        plotted = False
        for index, (key, label_key) in enumerate(SERIES):
            pts = [(x, r.get(key)) for x, r in zip(iterations, self.records)
                   if isinstance(r.get(key), (int, float)) and r.get(key) > 0 and math.isfinite(r.get(key))]
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker='o', markersize=2.8, linewidth=1.5, color=p['lines'][index],
                    label=tr(label_key))
            plotted = True
        if plotted:
            ax.set_yscale('log')
        moments = []
        for x, r in zip(iterations, self.records):
            for key in MOMENT_KEYS:
                value = r.get(key)
                if isinstance(value, (int, float)) and math.isfinite(value):
                    moments.append((x, value)); break
        handles, labels = ax.get_legend_handles_labels()
        if moments and any(abs(m) > 1e-8 for _, m in moments):
            twin = ax.twinx()
            xs, ys = zip(*moments)
            line, = twin.plot(xs, ys, linestyle='--', linewidth=1.3, color=p['lines'][3], label=tr('chart.moment'))
            twin.set_ylabel(tr('chart.moment'), fontdict=font)
            twin.tick_params(colors=p['muted'], labelsize=8)
            for spine in twin.spines.values():
                spine.set_color(p['border_strong'])
            handles.append(line); labels.append(tr('chart.moment'))
        if handles:
            legend = ax.legend(handles, labels, loc='upper right', fontsize=7.5, frameon=True,
                               prop={'family': families, 'size': 7.5})
            legend.get_frame().set_facecolor(p['surface'])
            legend.get_frame().set_edgecolor(p['border'])
            for text in legend.get_texts():
                text.set_color(p['text'])
        else:
            ax.text(0.5, 0.5, tr('chart.waiting'), transform=ax.transAxes, ha='center', va='center',
                    fontdict={'family': families, 'size': 9, 'color': p['muted']})
            ax.set_xticks([]); ax.set_yticks([])
        self.canvas.draw_idle()


class IterationFollower:
    """Follows the newest ``*.iterations.jsonl`` under an output directory."""

    def __init__(self, chart: ConvergenceChart):
        self.chart = chart
        self.root: Path | None = None
        self.tail: IterationTail | None = None
        self.received = 0

    def follow(self, root):
        self.root = Path(root) if root else None
        self.tail = None
        self.received = 0
        self.chart.clear()

    def poll(self, *, final=False) -> int:
        if self.root is None:
            return 0
        latest = latest_iteration_log(self.root)
        if latest is None:
            return 0
        if self.tail is None or self.tail.path != latest:
            self.tail = IterationTail(latest)
            self.chart.clear(latest)
        before = self.tail.resets
        records = self.tail.read_new(final=final)
        if self.tail.resets != before:
            # The log was truncated or replaced: drop the stale series before adding new records.
            self.chart.clear(latest)
        self.received += len(records)
        self.chart.add_records(records)
        return len(records)
