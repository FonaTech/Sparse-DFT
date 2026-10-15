"""Line icons drawn from inline SVG (24x24 grid, stroke only, tinted per theme)."""
from __future__ import annotations

_S = 'fill="none" stroke="{c}" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"'
GLYPHS = {
    'structure': '<path d="M12 3 20 7.5v9L12 21 4 16.5v-9z"/><path d="M4 7.5 12 12l8-4.5M12 12v9"/>',
    'paw': '<circle cx="12" cy="12" r="2.2"/><ellipse cx="12" cy="12" rx="9" ry="3.6"/>'
           '<ellipse cx="12" cy="12" rx="9" ry="3.6" transform="rotate(60 12 12)"/>'
           '<ellipse cx="12" cy="12" rx="9" ry="3.6" transform="rotate(120 12 12)"/>',
    'kpoints': '<circle cx="6" cy="6" r="1.4"/><circle cx="12" cy="6" r="1.4"/><circle cx="18" cy="6" r="1.4"/>'
               '<circle cx="6" cy="12" r="1.4"/><circle cx="12" cy="12" r="1.4"/><circle cx="18" cy="12" r="1.4"/>'
               '<circle cx="6" cy="18" r="1.4"/><circle cx="12" cy="18" r="1.4"/><circle cx="18" cy="18" r="1.4"/>',
    'scf': '<path d="M19.5 9A8 8 0 0 0 5.2 7.6"/><path d="M4.5 15a8 8 0 0 0 14.3 1.4"/>'
           '<path d="M19.8 4.5V9h-4.5M4.2 19.5V15h4.5"/>',
    'bands': '<path d="M3 7c3-3 6 3 9 0s6 3 9 0"/><path d="M3 13c3 3 6-3 9 0s6-3 9 0"/><path d="M3 19h18"/>',
    'relax': '<path d="M4 12h6M14 12h6M7 9l3 3-3 3M17 9l-3 3 3 3"/><path d="M12 4v4M12 16v4"/>',
    'sparsity': '<circle cx="5" cy="6" r="1.3"/><circle cx="12" cy="4.5" r="1.3"/><circle cx="18.5" cy="8" r="1.3"/>'
                '<circle cx="8" cy="12.5" r="1.3"/><circle cx="16" cy="14" r="1.3"/><circle cx="5.5" cy="19" r="1.3"/>'
                '<circle cx="12.5" cy="19.5" r="1.3"/><circle cx="19.5" cy="19" r="1.3"/>',
    'mp': '<ellipse cx="12" cy="5.5" rx="7.5" ry="2.5"/><path d="M4.5 5.5v13c0 1.4 3.4 2.5 7.5 2.5s7.5-1.1 7.5-2.5v-13"/>'
          '<path d="M4.5 12c0 1.4 3.4 2.5 7.5 2.5s7.5-1.1 7.5-2.5"/>',
    'analysis': '<path d="M4 20h16"/><path d="M7 16v-5M12 16V6M17 16v-8"/>',
    'diagnostics': '<path d="M3 12h4l2-5 4 10 2-5h6"/>',
    'queue': '<rect x="4" y="4" width="16" height="4.5" rx="1.5"/><rect x="4" y="10" width="16" height="4.5" rx="1.5"/>'
             '<rect x="4" y="16" width="10" height="4" rx="1.5"/>',
    'settings': '<circle cx="12" cy="12" r="3"/><path d="M12 2.8v2.6M12 18.6v2.6M2.8 12h2.6M18.6 12h2.6'
                'M5.5 5.5l1.8 1.8M16.7 16.7l1.8 1.8M5.5 18.5l1.8-1.8M16.7 7.3l1.8-1.8"/>',
    'play': '<path d="M9.4 5.9 17.9 11a1.2 1.2 0 0 1 0 2l-8.5 5.1A1.2 1.2 0 0 1 7.6 17.1V6.9'
            'A1.2 1.2 0 0 1 9.4 5.9z"/>',
    'stop': '<rect x="6" y="6" width="12" height="12" rx="2"/>',
    'copy': '<rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V5.5A1.5 1.5 0 0 0 14.5 4h-9A1.5 1.5 0 0 0 4 5.5v9A1.5 1.5 0 0 0 5.5 16H8"/>',
    'folder': '<path d="M3.5 7.5A1.5 1.5 0 0 1 5 6h4l2 2h8a1.5 1.5 0 0 1 1.5 1.5v8A1.5 1.5 0 0 1 19 19H5a1.5 1.5 0 0 1-1.5-1.5z"/>',
    'file': '<path d="M14 3.5H7A1.5 1.5 0 0 0 5.5 5v14A1.5 1.5 0 0 0 7 20.5h10a1.5 1.5 0 0 0 1.5-1.5V8z"/><path d="M14 3.5V8h4.5"/>',
    'rerun': '<path d="M4.5 12a7.5 7.5 0 1 0 2.2-5.3"/><path d="M4.5 4.5v4h4"/>',
    'export': '<path d="M12 15V4M8 8l4-4 4 4"/><path d="M5 14v4.5A1.5 1.5 0 0 0 6.5 20h11a1.5 1.5 0 0 0 1.5-1.5V14"/>',
    'search': '<circle cx="11" cy="11" r="6"/><path d="M20 20l-4.6-4.6"/>',
    'close': '<path d="M6 6l12 12M18 6 6 18"/>',
    'info': '<circle cx="12" cy="12" r="9"/><path d="M12 11v5.5M12 7.6v.2"/>',
    'warning': '<path d="M12 3.5 21.5 20h-19z"/><path d="M12 10v4.5M12 17.4v.2"/>',
    'error': '<circle cx="12" cy="12" r="9"/><path d="M9 9l6 6M15 9l-6 6"/>',
    'ok': '<circle cx="12" cy="12" r="9"/><path d="M8 12.5l2.7 2.7L16.5 9.5"/>',
    'trash': '<path d="M5 7h14M10 7V5h4v2M7 7l1 13h8l1-13"/>',
    'moon': '<path d="M19.5 14.5A8 8 0 0 1 9.5 4.5a8 8 0 1 0 10 10z"/>',
    'sun': '<circle cx="12" cy="12" r="4"/><path d="M12 2.5v2M12 19.5v2M2.5 12h2M19.5 12h2M5.3 5.3l1.4 1.4M17.3 17.3l1.4 1.4M5.3 18.7l1.4-1.4M17.3 6.7l1.4-1.4"/>',
    'globe': '<circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c2.6 2.5 3.9 5.5 3.9 9s-1.3 6.5-3.9 9c-2.6-2.5-3.9-5.5-3.9-9S9.4 5.5 12 3z"/>',
}


def svg(name: str, colour: str) -> str:
    body = GLYPHS.get(name, GLYPHS['file'])
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24">'
            f'<g {_S.format(c=colour)}>{body}</g></svg>')


_CACHE: dict = {}


def icon(name: str, colour: str = '#2B2A33', size: int = 20):
    """``QIcon`` for ``name`` drawn in ``colour`` (rendered at 1x and 2x for high-DPI screens)."""
    key = (name, colour, size)
    if key in _CACHE:
        return _CACHE[key]
    from PyQt6.QtCore import QByteArray, QRectF, Qt
    from PyQt6.QtGui import QIcon, QImage, QPainter, QPixmap
    from PyQt6.QtSvg import QSvgRenderer
    renderer = QSvgRenderer(QByteArray(svg(name, colour).encode('utf-8')))
    result = QIcon()
    for scale in (1, 2, 3):
        image = QImage(size * scale, size * scale, QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        renderer.render(painter, QRectF(0, 0, size * scale, size * scale))
        painter.end()
        pixmap = QPixmap.fromImage(image)
        pixmap.setDevicePixelRatio(scale)
        result.addPixmap(pixmap)
    _CACHE[key] = result
    return result
