"""Matplotlib theme aligned with the project's brand tokens.

Counterpart to :mod:`ark.plot.theme` (Lets-Plot) and :mod:`ark.plot.gt_style`
(tables) - same brand, different rendering backend. See :mod:`ark.plot.tokens`
for the underlying color/font constants.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any

from cycler import cycler
import matplotlib as mpl
from matplotlib import font_manager

from ark.plot.tokens import BODY_FONT_CANDIDATES, BRAND_PALETTE

log = logging.getLogger(__name__)


def resolve_font(candidates: list[str]) -> str:
    """Return the first font from candidates that matplotlib can find.

    CSS-style font stacks (e.g. "Libre Franklin, sans-serif") only work in a
    browser - matplotlib needs a single font name present in
    ``font_manager.fontManager.ttflist``. Falls back to ``'DejaVu Sans'``,
    matplotlib's guaranteed built-in, if none of the candidates are installed.

    Args:
        candidates: Ordered list of preferred font names.

    Returns:
        Name of the first available font, or "DejaVu Sans".
    """
    available = {f.name for f in font_manager.fontManager.ttflist}
    for font in candidates:
        if font in available:
            return font
    log.debug("None of %s installed - falling back to DejaVu Sans", candidates)
    return "DejaVu Sans"


def golden_height(width: float) -> float:
    """Calculate height maintaining the golden ratio (~0.618)."""
    return width * 0.618


def configure_matplotlib_style(
    style: list[str] | None = None,
    font_size: int = 10,
    fig_width: float | None = None,
    fig_height: float | None = None,
    dpi: int = 150,
    color_cycle: list[str] | None = None,
    custom_rc: dict[str, Any] | None = None,
) -> None:
    """Apply the project's house style to matplotlib's global rcParams.

    Brand-aligned counterpart to :func:`ark.plot.theme.modern_theme`: light
    axis strokes, muted tick/label colors, no legend frame, and the brand
    color cycle from :data:`ark.plot.tokens.BRAND_PALETTE`.

    Args:
        style: Matplotlib style sheet name(s) to layer underneath these
            settings (e.g. ``["seaborn-v0_8-whitegrid"]``). None applies no
            base style - just the rcParams below.
        font_size: Base font size in points.
        fig_width: Figure width in inches. None uses matplotlib's default.
        fig_height: Figure height in inches. None derives it from fig_width
            via the golden ratio if fig_width is given, else uses the default.
        dpi: Output resolution for both on-screen and saved figures.
        color_cycle: Hex color cycle for `axes.prop_cycle`. None uses
            BRAND_PALETTE.
        custom_rc: Additional rcParams to apply after the defaults below,
            for one-off overrides.

    Examples:
        >>> from ark.plot.matplot_theme import configure_matplotlib_style
        >>> configure_matplotlib_style(font_size=11, fig_width=6)
        >>> fig, ax = plt.subplots()
    """
    font_family = resolve_font(BODY_FONT_CANDIDATES)
    title_size = int(font_size * 1.5)
    small_font_size = int(font_size * 0.85)

    if fig_width is not None and fig_height is None:
        fig_height = golden_height(fig_width)

    plot_config = {
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "figure.figsize": (
            fig_width or mpl.rcParams["figure.figsize"][0],
            fig_height or mpl.rcParams["figure.figsize"][1],
        ),
        "font.size": font_size,
        "font.family": font_family,
        "axes.labelsize": font_size,
        "axes.titlesize": title_size,
        "axes.edgecolor": "#bbbbbb",
        "axes.facecolor": "white",
        "axes.labelcolor": "#333333",
        "axes.titlecolor": "#333333",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": small_font_size,
        "ytick.labelsize": small_font_size,
        "xtick.color": "#666666",
        "ytick.color": "#666666",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.fontsize": small_font_size,
        "axes.linewidth": 0.8,
        "grid.color": "#e8e8e8",
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.4,
        "lines.markeredgewidth": 0.8,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "figure.autolayout": True,
        "legend.frameon": False,
        "legend.handlelength": 1.4,
        "legend.labelspacing": 0.3,
        "axes.prop_cycle": cycler(color=color_cycle or BRAND_PALETTE),
    }

    if custom_rc:
        plot_config.update(custom_rc)

    if style:
        try:
            mpl.style.use(style)
        except OSError as e:
            log.warning("Could not apply style %s: %s", style, e)

    mpl.rcParams.update(plot_config)

    with contextlib.suppress(ImportError):
        from matplotlib_inline.backend_inline import set_matplotlib_formats

        set_matplotlib_formats("svg", "png")


@contextlib.contextmanager
def plotting_context(**kwargs: Any):
    """Apply :func:`configure_matplotlib_style` temporarily, then restore rcParams.

    Examples:
        >>> with plotting_context(font_size=14):
        ...     fig, ax = plt.subplots()
    """
    original_rc = mpl.rcParams.copy()
    configure_matplotlib_style(**kwargs)
    try:
        yield
    finally:
        mpl.rcParams.update(original_rc)
