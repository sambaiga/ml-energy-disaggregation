"""Lets-Plot theme and color palette aligned with the project's brand tokens.

See :mod:`ark.plot.tokens` for the underlying color/font constants, and
:mod:`ark.plot.matplot_theme` / :mod:`ark.plot.gt_style` for the matplotlib
and table equivalents of this theme.
"""

from __future__ import annotations

from typing import Any, Literal

from lets_plot import element_blank, element_line, element_rect, element_text, theme

from ark.plot.tokens import BODY_FONT_STACK, BRAND_PALETTE

# Backwards-compatible aliases - existing call sites (ark.plot.basic_plots,
# book/03-data-analysis) import these names directly.
pro_colors = BRAND_PALETTE


def modern_theme(
    show_x_axis: bool = True,
    font_size: int = 12,
    line_font_size: float = 1.0,
    x_axis_angle: int = 0,
    legend_pos: Literal["top", "bottom", "left", "right", "none"] = "top",
    grid: bool = False,
    legend_direction: Literal["horizontal", "vertical"] | None = None,
    legend_key_size: int | None = None,
    legend_border: bool = False,
) -> Any:
    """Create the project's house theme for Lets-Plot visualizations.

    Produces clean, publication-quality figures aligned with the project
    brand (see :mod:`ark.plot.tokens`): minimal gridlines, a left-aligned
    title, styled facet strips, and a Libre Franklin / system-sans font
    stack matching the personal site's typography.

    Args:
        show_x_axis: Whether to render x-axis labels, ticks, and title.
        font_size: Base font size in points (axis labels, legend text).
        line_font_size: Base stroke width for axis lines and ticks.
        x_axis_angle: Rotation angle for x-axis tick labels.
        legend_pos: Legend anchor - "top", "bottom", "left", "right", or "none".
        grid: If True, draw subtle horizontal reference lines.
        legend_direction: Arrange legend keys "horizontal" or "vertical".
            When None (default), lets-plot chooses based on legend_pos.
        legend_key_size: Size in pixels of the colour swatch in each legend key.
        legend_border: If True, draw a thin grey border box around the legend.

    Returns:
        A Lets-Plot theme object ready to be added to any ggplot.

    Examples:
        >>> from ark.plot.theme import modern_theme
        >>> p = ggplot(df, aes("x", "y")) + geom_line() + modern_theme()
    """
    title_size = int(font_size * 1.6)
    subtitle_size = int(font_size * 1.4)
    axis_title_size = int(font_size * 1.2)

    legend_bg = element_rect(fill="#ffffff", color="#dddddd", size=0.5) if legend_border else element_blank()

    base = theme(
        # ── Legend ──────────────────────────────────────────────────────────
        legend_position=legend_pos,
        legend_background=legend_bg,
        legend_key=element_blank(),
        legend_spacing=10,
        # ── Backgrounds & whitespace ─────────────────────────────────────────
        plot_background=element_blank(),
        panel_background=element_blank(),
        panel_border=element_blank(),
        plot_margin=[16, 16, 10, 10],  # top, right, bottom, left
        strip_background=element_blank(),
        strip_text=element_text(family=BODY_FONT_STACK, face="bold", size=font_size, hjust=0),
        # ── Grid lines ──────────────────────────────────────────────────────
        panel_grid_minor=element_blank(),
        panel_grid_major_x=element_blank(),
        panel_grid_major_y=(element_line(color="#e8e8e8", size=0.4) if grid else element_blank()),
        # ── Axes - light strokes let data dominate ───────────────────────────
        axis_line=element_line(size=line_font_size, color="#bbbbbb"),
        axis_ticks=element_line(size=line_font_size * 0.75, color="#bbbbbb"),
        # ── Global text ─────────────────────────────────────────────────────
        text=element_text(family=BODY_FONT_STACK, size=font_size, color="#2d2d2d"),
        axis_text_x=element_text(angle=x_axis_angle, hjust=0.5, margin=[8, 0, 0, 0], color="#666666"),
        axis_text_y=element_text(margin=[0, 0, 0, 8], color="#666666"),
        axis_title_x=element_text(size=axis_title_size, face="bold", margin=[10, 0, 0, 0]),
        axis_title_y=element_text(size=axis_title_size, face="bold", margin=[0, 10, 0, 0]),
        # ── Title / subtitle / caption ───────────────────────────────────────
        plot_title=element_text(size=title_size, face="bold", hjust=0, margin=[0, 0, 6, 0], color="#333333"),
        plot_subtitle=element_text(size=subtitle_size, hjust=0, margin=[0, 0, 18, 0], color="#6b6b6b"),
        plot_caption=element_text(
            size=int(font_size * 0.82), face="italic", hjust=1.0, color="#999999", margin=[12, 0, 0, 0]
        ),
        # ── Legend text ──────────────────────────────────────────────────────
        legend_title=element_text(size=axis_title_size, face="bold"),
        legend_text=element_text(size=font_size),
    )

    if not show_x_axis:
        base += theme(
            axis_title_x=element_blank(),
            axis_text_x=element_blank(),
            axis_ticks_x=element_blank(),
            axis_line_x=element_blank(),
        )

    legend_extra: dict[str, Any] = {}
    if legend_direction is not None:
        legend_extra["legend_direction"] = legend_direction
    if legend_key_size is not None:
        legend_extra["legend_key_size"] = legend_key_size
    if legend_extra:
        base += theme(**legend_extra)

    return base
