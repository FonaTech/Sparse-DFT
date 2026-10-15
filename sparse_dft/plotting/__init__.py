"""Publication-grade figures for Sparse-DFT outputs.

Quick use::

    from sparse_dft.plotting import make_figure, export
    fig = make_figure('bands', 'job/bands', width='single', lang='en')
    export(fig, 'figures/si_bands', formats=('pdf', 'png'))

    figs = make_figures('electronic_atlas', ['job/bands', 'job/dos'], panels='both', reference='vasp/bands')
    export_figures(figs, 'figures/gaas_atlas', formats=('pdf', 'png'))   # STEM.png + STEM_<panel>.png

Every figure is a :class:`matplotlib.figure.Figure` built without pyplot and
styled inside :func:`matplotlib.rc_context`; global rcParams are untouched.
"""
from __future__ import annotations

from .export import export, export_figures
from .figures import KINDS, MULTI_PANEL, available_kinds, make_figure, make_figures
from .style import PALETTES, WIDTHS_MM, figure_style, new_figure, panel_labels

__all__ = ['KINDS', 'MULTI_PANEL', 'PALETTES', 'WIDTHS_MM', 'available_kinds', 'export', 'export_figures',
           'figure_style', 'make_figure', 'make_figures', 'new_figure', 'panel_labels']
