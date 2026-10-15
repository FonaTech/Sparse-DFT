"""Macaron theme: one pastel palette, light and dark variants, QSS over Fusion.

Body text on every surface meets WCAG AA (>= 4.5:1); chart line colours reach
>= 3.8:1 on the plot background (non-text graphics need 3:1).  The ratios
are checked in ``tests/test_gui.py`` with :func:`contrast_ratio`.
"""
from __future__ import annotations

LIGHT = {
    'name': 'light',
    'bg': '#F6F4F1', 'surface': '#FFFFFF', 'surface_alt': '#FAF8F5', 'nav': '#EFEBF5',
    'border': '#E3DEE8', 'border_strong': '#CFC8DA',
    'text': '#2B2A33', 'muted': '#5E5A6E', 'disabled': '#9C98A8',
    'accent': '#DCCBFA', 'accent_strong': '#5B45A8', 'accent_text': '#2B2A33',
    'pistachio': '#BEE3B0', 'lavender': '#DCCBFA', 'peach': '#FFD8BE', 'blue': '#C4E2F7',
    'rose': '#F8CAD6', 'lemon': '#FFF0B3',
    'ok_text': '#24361F', 'warn_text': '#5A3B12', 'err_bg': '#FDECF1', 'err_text': '#9C2B4B',
    'info_bg': '#E8F3FC', 'info_text': '#1F4567', 'ok_bg': '#EAF6E4', 'warn_bg': '#FFF2E6',
    'selection': '#E6DAFC', 'console_bg': '#FBFAF8', 'console_text': '#2B2A33',
    'lines': ['#7559C0', '#3E8E5A', '#C8673F', '#3D7EBF', '#C2507A'],
    'grid': '#E9E5EE', 'plot_bg': '#FFFFFF',
}
DARK = {
    'name': 'dark',
    'bg': '#1C1C24', 'surface': '#262632', 'surface_alt': '#2D2D3A', 'nav': '#21212B',
    'border': '#363645', 'border_strong': '#4A4A5C',
    'text': '#ECEAF4', 'muted': '#B3AFC6', 'disabled': '#77748A',
    'accent': '#4A4170', 'accent_strong': '#C7B2F2', 'accent_text': '#ECEAF4',
    'pistachio': '#A9D89A', 'lavender': '#C7B2F2', 'peach': '#F5C3A2', 'blue': '#A9D1F0',
    'rose': '#F2AFC2', 'lemon': '#F3E3A0',
    'ok_text': '#CDEFC2', 'warn_text': '#F8D9B8', 'err_bg': '#3A2430', 'err_text': '#F6B3C5',
    'info_bg': '#20303F', 'info_text': '#C4E2F7', 'ok_bg': '#24342A', 'warn_bg': '#3A2E22',
    'selection': '#4A4170', 'console_bg': '#1F1F28', 'console_text': '#E2E0EC',
    'lines': ['#C7B2F2', '#A9D89A', '#F5C3A2', '#A9D1F0', '#F2AFC2'],
    'grid': '#34343F', 'plot_bg': '#262632',
}
THEMES = {'light': LIGHT, 'dark': DARK}


def palette(name: str) -> dict:
    return THEMES.get(name, LIGHT)


def _luminance(colour: str) -> float:
    colour = colour.lstrip('#')
    r, g, b = (int(colour[i:i + 2], 16) / 255 for i in (0, 2, 4))
    f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4  # noqa: E731
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


def contrast_ratio(a: str, b: str) -> float:
    la, lb = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def stylesheet(name: str = 'light', *, font_px: int = 13) -> str:
    """Application QSS for the chosen variant."""
    p = palette(name)
    return QSS_TEMPLATE.format(fs=font_px, fs_small=max(font_px - 1, 10), fs_title=font_px + 7,
                               fs_sub=font_px + 1, **{k: v for k, v in p.items() if isinstance(v, str)})


def apply_palette(app, name: str = 'light'):
    """Set the Fusion QPalette so native-drawn parts match the QSS."""
    from PyQt6.QtGui import QColor, QPalette
    p = palette(name)
    pal = QPalette()
    role = QPalette.ColorRole
    for r, key in ((role.Window, 'bg'), (role.Base, 'surface'), (role.AlternateBase, 'surface_alt'),
                   (role.WindowText, 'text'), (role.Text, 'text'), (role.Button, 'surface'),
                   (role.ButtonText, 'text'), (role.ToolTipBase, 'surface'), (role.ToolTipText, 'text'),
                   (role.Highlight, 'selection'), (role.HighlightedText, 'text'),
                   (role.PlaceholderText, 'disabled'), (role.Link, 'accent_strong'), (role.Mid, 'border'),
                   (role.Light, 'surface'), (role.Midlight, 'surface_alt'), (role.Dark, 'border_strong'),
                   (role.Shadow, 'border_strong')):
        pal.setColor(r, QColor(p[key]))
    group = QPalette.ColorGroup.Disabled
    for r in (role.WindowText, role.Text, role.ButtonText):
        pal.setColor(group, r, QColor(p['disabled']))
    app.setPalette(pal)


QSS_TEMPLATE = """
* {{ font-size: {fs}px; }}
QWidget {{ color: {text}; background: transparent; }}
QMainWindow, QDialog, QWidget#Root {{ background: {bg}; }}
QToolTip {{ background: {surface}; color: {text}; border: 1px solid {border_strong};
            border-radius: 6px; padding: 6px 8px; }}

/* navigation */
QWidget#NavPanel {{ background: {nav}; border-right: 1px solid {border}; }}
QLabel#Brand {{ font-size: {fs_title}px; font-weight: 700; color: {text}; padding: 2px 4px; }}
QLabel#BrandSub {{ color: {muted}; font-size: {fs_small}px; padding: 0 4px; }}
QListWidget#Nav {{ background: transparent; border: none; outline: 0; }}
QListWidget#Nav::item {{ padding: 7px 10px; margin: 1px 0; border-radius: 9px; color: {text}; }}
QListWidget#Nav::item:hover {{ background: {surface_alt}; }}
QListWidget#Nav::item:selected {{ background: {accent}; color: {accent_text}; font-weight: 600; }}

/* page header and cards */
QLabel[role="title"] {{ font-size: {fs_title}px; font-weight: 700; }}
QLabel[role="subtitle"] {{ color: {muted}; }}
QLabel[role="muted"] {{ color: {muted}; }}
QLabel[role="section"] {{ font-weight: 700; font-size: {fs_sub}px; }}
QLabel[role="mono"] {{ font-family: "Menlo", "Consolas", "DejaVu Sans Mono", monospace; }}
QFrame[card="true"] {{ background: {surface}; border: 1px solid {border}; border-radius: 12px; }}
QWidget#FormBody, QWidget#PageBody {{ background: {bg}; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}

/* inputs */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {{
    background: {surface}; border: 1px solid {border_strong}; border-radius: 7px;
    padding: 4px 8px; selection-background-color: {selection}; selection-color: {text}; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{ min-height: 22px; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QPlainTextEdit:focus {{
    border: 1px solid {accent_strong}; }}
QLineEdit[invalid="true"], QSpinBox[invalid="true"], QComboBox[invalid="true"] {{
    border: 1px solid {err_text}; background: {err_bg}; }}
QLineEdit:disabled, QSpinBox:disabled, QComboBox:disabled {{ color: {disabled}; background: {surface_alt}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{ background: {surface}; border: 1px solid {border_strong};
    selection-background-color: {selection}; selection-color: {text}; outline: 0; }}
QSpinBox::up-button, QSpinBox::down-button {{ width: 16px; border: none; }}
QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 4px; border: 1px solid {border_strong};
    background: {surface}; }}
QCheckBox::indicator:checked {{ background: {accent_strong}; border: 1px solid {accent_strong}; }}

/* buttons */
QPushButton, QToolButton {{ background: {surface}; border: 1px solid {border_strong}; border-radius: 8px;
    padding: 5px 12px; }}
QPushButton:hover, QToolButton:hover {{ background: {surface_alt}; border-color: {accent_strong}; }}
QPushButton:pressed, QToolButton:pressed {{ background: {selection}; }}
QPushButton:disabled, QToolButton:disabled {{ color: {disabled}; border-color: {border}; }}
QPushButton[kind="primary"] {{ background: {pistachio}; color: #1C2A18; border: 1px solid {pistachio};
    font-weight: 700; padding: 6px 18px; }}
QPushButton[kind="primary"]:hover {{ border: 1px solid {ok_text}; }}
QPushButton[kind="accent"] {{ background: {lavender}; color: #24203A; border: 1px solid {lavender}; font-weight: 600; }}
QPushButton[kind="danger"] {{ background: {rose}; color: #3A1020; border: 1px solid {rose}; font-weight: 600; }}
QPushButton[kind="ghost"], QToolButton[kind="ghost"] {{ background: transparent; border: 1px solid transparent; }}
QPushButton[kind="ghost"]:hover, QToolButton[kind="ghost"]:hover {{ background: {surface_alt}; border-color: {border}; }}

/* collapsible groups */
QGroupBox {{ background: {surface}; border: 1px solid {border}; border-radius: 12px; margin-top: 14px;
    padding: 14px 12px 10px 12px; font-weight: 600; }}
QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left; left: 12px; padding: 1px 8px;
    background: {lavender}; color: #24203A; border-radius: 6px; }}
QGroupBox::indicator {{ width: 12px; height: 12px; border-radius: 3px; border: 1px solid #6E6488; background: {surface}; }}
QGroupBox::indicator:checked {{ background: #6E6488; }}
QGroupBox[section="group"]::title {{ background: {peach}; color: #3A2412; }}
QGroupBox[section="inputs"]::title {{ background: {pistachio}; color: #1C2A18; }}
QGroupBox[section="spin"]::title {{ background: {rose}; color: #3A1020; }}
QGroupBox[section="convergence"]::title {{ background: {blue}; color: #12304A; }}
QGroupBox[section="relax"]::title {{ background: {lemon}; color: #3A3210; }}
QGroupBox[section="parallel"]::title {{ background: {blue}; color: #12304A; }}
QGroupBox QLabel {{ font-weight: 400; }}

/* tabs, tables, splitters */
QTabWidget::pane {{ border: 1px solid {border}; border-radius: 10px; background: {surface}; top: -1px; }}
QTabBar::tab {{ background: transparent; padding: 6px 14px; margin-right: 2px; border-top-left-radius: 8px;
    border-top-right-radius: 8px; color: {muted}; }}
QTabBar::tab:selected {{ background: {surface}; color: {text}; font-weight: 600; border: 1px solid {border};
    border-bottom: 2px solid {accent_strong}; }}
QTableWidget, QTableView, QTreeWidget, QListWidget {{ background: {surface}; border: 1px solid {border};
    border-radius: 10px; gridline-color: {border}; alternate-background-color: {surface_alt};
    selection-background-color: {selection}; selection-color: {text}; }}
QHeaderView::section {{ background: {surface_alt}; color: {muted}; border: none; border-bottom: 1px solid {border};
    padding: 6px 8px; font-weight: 600; }}
QSplitter::handle {{ background: {bg}; }}
QSplitter::handle:horizontal {{ width: 6px; }}
QSplitter::handle:vertical {{ height: 6px; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {border_strong}; border-radius: 4px; min-height: 30px; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {border_strong}; border-radius: 4px; min-width: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QProgressBar {{ background: {surface_alt}; border: 1px solid {border}; border-radius: 6px; text-align: center;
    min-height: 10px; max-height: 10px; color: transparent; }}
QProgressBar::chunk {{ background: {pistachio}; border-radius: 5px; }}

/* console, preview, banners, chips */
QPlainTextEdit#Console {{ background: {console_bg}; color: {console_text}; border-radius: 10px;
    font-family: "Menlo", "Consolas", "DejaVu Sans Mono", monospace; font-size: {fs_small}px; }}
QLineEdit#CommandPreview {{ background: {surface_alt}; color: {text};
    font-family: "Menlo", "Consolas", "DejaVu Sans Mono", monospace; font-size: {fs_small}px; }}
QFrame#Banner {{ border-radius: 10px; border: 1px solid {border}; }}
QFrame#Banner[level="error"] {{ background: {err_bg}; border-color: {rose}; }}
QFrame#Banner[level="warning"] {{ background: {warn_bg}; border-color: {peach}; }}
QFrame#Banner[level="info"] {{ background: {info_bg}; border-color: {blue}; }}
QFrame#Banner[level="ok"] {{ background: {ok_bg}; border-color: {pistachio}; }}
QFrame#Banner QLabel[level="error"] {{ color: {err_text}; }}
QFrame#Banner QLabel[level="warning"] {{ color: {warn_text}; }}
QFrame#Banner QLabel[level="info"] {{ color: {info_text}; }}
QFrame#Banner QLabel[level="ok"] {{ color: {ok_text}; }}
QLabel[chip] {{ border-radius: 9px; padding: 2px 10px; font-size: {fs_small}px; font-weight: 600; color: #24203A; }}
QLabel[chip="running"], QFrame QLabel[chip="running"], QTableWidget QLabel[chip="running"] {{ background: {blue}; color: #12304A; }}
QLabel[chip="queued"], QFrame QLabel[chip="queued"], QTableWidget QLabel[chip="queued"] {{ background: {lemon}; color: #3A3210; }}
QLabel[chip="finished"], QFrame QLabel[chip="finished"], QTableWidget QLabel[chip="finished"] {{ background: {pistachio}; color: #1C2A18; }}
QLabel[chip="failed"], QFrame QLabel[chip="failed"], QTableWidget QLabel[chip="failed"] {{ background: {rose}; color: #3A1020; }}
QLabel[chip="cancelled"], QFrame QLabel[chip="cancelled"], QTableWidget QLabel[chip="cancelled"] {{ background: {peach}; color: #3A2412; }}
QLabel[chip="idle"], QFrame QLabel[chip="idle"], QTableWidget QLabel[chip="idle"] {{ background: {surface_alt}; color: {muted}; border: 1px solid {border}; }}
QStatusBar {{ background: {nav}; color: {muted}; border-top: 1px solid {border}; }}
QStatusBar QLabel {{ color: {muted}; }}
QMenu {{ background: {surface}; border: 1px solid {border_strong}; border-radius: 8px; padding: 4px; }}
QMenu::item {{ padding: 5px 18px; border-radius: 6px; }}
QMenu::item:selected {{ background: {selection}; }}
"""
