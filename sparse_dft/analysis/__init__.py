"""Sparse-DFT analysis: typed loaders for every calculation output plus the
provenance-tracked band/DOS report used by ``sparse-dft bands-analysis``.

The loaders are pure Python; plotting lives in :mod:`sparse_dft.plotting`.
"""
from __future__ import annotations

from .report import analyze_bands, render_scientific_plots, _normalize_projections

__all__ = ['analyze_bands', 'render_scientific_plots', '_normalize_projections']
