"""Great Tables styling aligned with the project's brand tokens.

Counterpart to :mod:`ark.plot.theme` (Lets-Plot) and
:mod:`ark.plot.matplot_theme` (matplotlib) - same brand, applied to tables
instead of figures. See :mod:`ark.plot.tokens` for the underlying
color/font constants.
"""

from __future__ import annotations

from great_tables import GT, loc, style
import pandas as pd

from ark.plot.tokens import GRAY_300, PRIMARY, SURFACE_MUTED, TEXT_DARK, TEXT_MUTED, WHITE

_FONT_NAMES = ["Libre Franklin", "Segoe UI", "Roboto", "Helvetica Neue", "sans-serif"]


def themed_gt(
    gt: GT,
    *,
    width: str = "100%",
    striped: bool = True,
    n_rows: int | None = None,
) -> GT:
    """Apply the project's brand theme to any Great Tables `GT` object.

    Call this last, after `.tab_header()`, `.cols_label()`,
    `.tab_source_note()`, etc.

    Args:
        gt: Any GT object to style.
        width: CSS width for the table container.
        striped: If True, apply alternating muted-fill row stripes.
            Requires n_rows to take effect.
        n_rows: Number of data rows in the table (needed for striping).

    Returns:
        The same GT object with brand styling applied.

    Examples:
        >>> from great_tables import GT, md
        >>> from ark.plot.gt_style import themed_gt
        >>> table = themed_gt(
        ...     GT(df).tab_header(title=md("**My Table**")),
        ...     n_rows=len(df),
        ... )
        >>> table
    """
    gt = (
        gt.tab_options(
            table_width=width,
            container_width=width,
            table_font_names=_FONT_NAMES,
            table_font_size="13px",
            table_font_color=TEXT_DARK,
            heading_background_color=WHITE,
            heading_title_font_size="16px",
            heading_title_font_weight="bold",
            heading_subtitle_font_size="13px",
            heading_padding="10px",
            column_labels_background_color=PRIMARY,
            column_labels_font_weight="bold",
            column_labels_font_size="13px",
            column_labels_padding="9px",
            column_labels_border_bottom_width="1px",
            column_labels_border_bottom_style="solid",
            data_row_padding="7px",
            table_body_border_bottom_color=GRAY_300,
            table_body_border_bottom_width="1px",
            table_body_border_bottom_style="solid",
            table_border_top_color=PRIMARY,
            table_border_top_width="2px",
            table_border_top_style="solid",
            table_border_bottom_color=PRIMARY,
            table_border_bottom_width="1px",
            table_border_bottom_style="solid",
            source_notes_font_size="11px",
            source_notes_padding="6px",
        )
        .tab_style(style=style.text(color=TEXT_DARK, weight="bold"), locations=loc.title())
        .tab_style(style=style.text(color=TEXT_MUTED), locations=loc.subtitle())
        .tab_style(style=style.text(color=WHITE), locations=loc.column_labels())
        .tab_style(style=style.text(color=TEXT_MUTED, style="italic"), locations=loc.source_notes())
    )

    if striped and n_rows is not None:
        even_rows = list(range(1, n_rows, 2))
        if even_rows:
            gt = gt.tab_style(style=style.fill(color=SURFACE_MUTED), locations=loc.body(rows=even_rows))

    return gt


def metrics_report(
    res: pd.DataFrame,
    metrics: list[str],
    minimize_cols: list[str] | None = None,
    maximize_cols: list[str] | None = None,
    title: str = "Model Performance",
    subtitle: str | None = None,
    source_note: str = "DS-MLOps Path",
    decimals: int = 3,
) -> GT:
    """Create a brand-styled model/metric comparison table.

    Renders metrics in a clean, publication-ready table. Best values are
    highlighted; alternating row stripes improve readability.

    Args:
        res: DataFrame with one row per model/run and metric columns.
        metrics: Column names to format as numbers.
        minimize_cols: Metric columns where a lower value is better
            (e.g. MAE, RMSE). The minimum value is highlighted.
        maximize_cols: Metric columns where a higher value is better
            (e.g. accuracy, R^2). The maximum value is highlighted.
        title: Table title.
        subtitle: Optional subtitle shown below the title.
        source_note: Attribution line shown at the bottom of the table.
        decimals: Decimal places for numeric formatting.

    Returns:
        A GT object ready for display in a notebook.

    Examples:
        >>> from ark.plot.gt_style import metrics_report
        >>> table = metrics_report(
        ...     metrics_df,
        ...     metrics=["MAE", "RMSE", "R2"],
        ...     minimize_cols=["MAE", "RMSE"],
        ...     maximize_cols=["R2"],
        ... )
        >>> table
    """
    gt = (
        GT(res)
        .tab_header(title=title, subtitle=subtitle or "Metric comparison across all evaluated models")
        .fmt_number(columns=metrics, decimals=decimals)
        .tab_source_note(source_note)
    )
    gt = themed_gt(gt, n_rows=len(res))

    for col in minimize_cols or []:
        if col in res.columns:
            best_idx = int(res[col].idxmin())
            gt = gt.tab_style(
                style=[style.fill(color=SURFACE_MUTED), style.text(weight="bold", color=PRIMARY)],
                locations=loc.body(rows=best_idx, columns=col),
            )

    for col in maximize_cols or []:
        if col in res.columns:
            best_idx = int(res[col].idxmax())
            gt = gt.tab_style(
                style=[style.fill(color=SURFACE_MUTED), style.text(weight="bold", color=PRIMARY)],
                locations=loc.body(rows=best_idx, columns=col),
            )

    return gt
