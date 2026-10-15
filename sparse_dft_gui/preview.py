"""Aspect-preserving, HiDPI figure preview.

A matplotlib figure is laid out in millimetres; stretching it to a widget
squeezes fonts and panels.  :class:`FigurePreview` instead renders the figure
at its own aspect ratio to the device pixels it will occupy (widget size ×
devicePixelRatio), letterboxes it, and re-renders after resizes settle
(debounced, off the GUI thread).  Several figures (split panels) are shown
one at a time with previous/next navigation.
"""
from __future__ import annotations

import io

from PyQt6.QtCore import QRectF, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter
from PyQt6.QtWidgets import QSizePolicy, QWidget

from .widgets import run_async

DEBOUNCE_MS = 140
MAX_PIXELS = 3840 * 2400          # cap one preview bitmap (4K at 2x is still sharp)


def fit_rect(box_w: float, box_h: float, aspect: float) -> tuple[float, float, float, float]:
    """(x, y, w, h) of the largest rectangle with ``aspect`` = w/h centred in a box."""
    if box_w <= 0 or box_h <= 0 or aspect <= 0:
        return 0.0, 0.0, 0.0, 0.0
    if box_w / box_h > aspect:
        h = box_h
        w = h * aspect
    else:
        w = box_w
        h = w / aspect
    return (box_w - w) / 2.0, (box_h - h) / 2.0, w, h


def render_image(fig, width_px: int, height_px: int) -> QImage:
    """The figure rasterised to about ``width_px`` × ``height_px`` device pixels at its own aspect."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    w_in, h_in = fig.get_size_inches()
    dpi = max(20.0, min(width_px / w_in, height_px / h_in))
    if w_in * h_in * dpi * dpi > MAX_PIXELS:
        dpi = (MAX_PIXELS / (w_in * h_in)) ** 0.5
    from .plots import PLOT_LOCK       # one figure operation at a time across worker threads
    with PLOT_LOCK:
        if not isinstance(fig.canvas, FigureCanvasAgg):
            FigureCanvasAgg(fig)
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=dpi, facecolor='white')
    image = QImage()
    image.loadFromData(buffer.getvalue(), 'PNG')
    return image


class FigurePreview(QWidget):
    """Letterboxed preview of one of several figures; ``rendered`` fires after each new bitmap."""

    rendered = pyqtSignal(int)          # index of the figure just drawn

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figures: list = []
        self.names: list[str] = []
        self.index = 0
        self.image: QImage | None = None
        self.image_key = None
        self._busy = False
        self._pending = False
        self._generation = 0
        self.background = QColor('#F8FAFC')
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(QSize(240, 180))
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(DEBOUNCE_MS)
        self._timer.timeout.connect(self._render)

    # ---- content
    def set_figures(self, figures: dict | list):
        """Show ``figures`` ({name: Figure} or [Figure]); the first is displayed."""
        if isinstance(figures, dict):
            self.names, self.figures = list(figures), list(figures.values())
        else:
            self.figures = list(figures)
            self.names = [str(i + 1) for i in range(len(self.figures))]
        self.index = 0
        self.image, self.image_key = None, None
        self._generation += 1
        self.schedule(immediate=True)
        self.update()

    def clear(self):
        self.set_figures([])

    @property
    def figure(self):
        return self.figures[self.index] if self.figures else None

    def show_index(self, index: int):
        if not self.figures:
            return
        self.index = max(0, min(len(self.figures) - 1, int(index)))
        self.image, self.image_key = None, None
        self.schedule(immediate=True)
        self.update()

    def step(self, delta: int):
        if self.figures:
            self.show_index((self.index + delta) % len(self.figures))

    # ---- geometry
    def aspect(self) -> float:
        fig = self.figure
        if fig is None:
            return 1.0
        w, h = fig.get_size_inches()
        return float(w) / float(h)

    def target_rect(self) -> QRectF:
        x, y, w, h = fit_rect(self.width(), self.height(), self.aspect())
        return QRectF(x, y, w, h)

    def device_size(self) -> tuple[int, int]:
        rect = self.target_rect()
        dpr = self.devicePixelRatioF()
        return max(1, int(round(rect.width() * dpr))), max(1, int(round(rect.height() * dpr)))

    # ---- rendering
    def schedule(self, immediate=False):
        if not self.figures:
            return
        if immediate:
            self._timer.stop()
            self._render()
        else:
            self._timer.start()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.schedule()

    def _render(self):
        fig = self.figure
        if fig is None:
            return
        size = self.device_size()
        key = (id(fig), size)
        if key == self.image_key:
            return
        if self._busy:
            self._pending = True
            return
        self._busy = True
        generation = self._generation
        index = self.index
        run_async(render_image, fig, *size, on_done=lambda img: self._done(img, key, generation, index),
                  on_error=lambda text: self._done(None, key, generation, index))

    def _done(self, image, key, generation, index):
        self._busy = False
        if generation == self._generation and index == self.index and image is not None and not image.isNull():
            image.setDevicePixelRatio(self.devicePixelRatioF())
            self.image, self.image_key = image, key
            self.update()
            self.rendered.emit(index)
        failed = image is None or image.isNull()
        stale = self.figure is not None and self.image_key != (id(self.figure), self.device_size())
        if self._pending or (stale and not failed):     # a failed render is not retried in a loop
            self._pending = False
            self.schedule(immediate=True)

    def render_now(self) -> QImage | None:
        """Synchronous render at the current size (tests and screenshots)."""
        fig = self.figure
        if fig is None:
            return None
        size = self.device_size()
        image = render_image(fig, *size)
        image.setDevicePixelRatio(self.devicePixelRatioF())
        self.image, self.image_key = image, (id(fig), size)
        self.update()
        return image

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.background)
        if self.image is not None:
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            painter.drawImage(self.target_rect(), self.image)
        painter.end()
