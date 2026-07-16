"""Hand-drawn concept diagrams for the book, using matplotlib directly.

Unlike everything else in :mod:`ark.plot`, these are not data visualisations:
each function draws a fixed, illustrative diagram for one specific concept
(matplotlib's Figure/Axes model, NumPy broadcasting, ...). They are matplotlib
code on purpose, not Mermaid or a static image asset, so they render as a real
image everywhere a tutorial notebook is opened, including plain Jupyter and
VS Code, not just when processed by Quarto.
"""

from __future__ import annotations

from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import matplotlib.pyplot as plt
import numpy as np

from ark.plot.icons import ICONS, icon_font
from ark.plot.tokens import DANGER, INFO, PRIMARY, SUCCESS, TEXT_MUTED, WARNING


def figure_axes_diagram() -> plt.Figure:
    """Draw a labelled diagram of one Figure containing two Axes.

    Illustrates the distinction Part 5 opens with: a Figure is the whole
    canvas, an Axes is one plot inside it, and a single Figure can hold
    more than one. The two panels are real Axes with real plotted data,
    annotated rather than drawn as abstract boxes.

    Returns:
        The matplotlib Figure, ready to display in a notebook cell.
    """
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))

    for ax, title in zip(axes, ["Axes object 1", "Axes object 2"], strict=True):
        x = np.linspace(0, 10, 60)
        ax.plot(x, np.sin(x) + rng.normal(0, 0.08, 60), color=PRIMARY, linewidth=1.4)
        ax.set_xlabel("x label")
        ax.set_ylabel("y label")
        ax.set_title(title, fontsize=10, color=INFO, fontweight="bold")

    fig.subplots_adjust(top=0.72, bottom=0.18, left=0.12, right=0.97, wspace=0.35)

    # Draw the Figure's own boundary explicitly - this is the one line in
    # the whole diagram that has no equivalent in the rendered chart itself.
    # Its top edge sits below the label text so the two never overlap.
    boundary = plt.Rectangle(
        (0.01, 0.01),
        0.98,
        0.85,
        transform=fig.transFigure,
        fill=False,
        edgecolor=WARNING,
        linewidth=2,
        linestyle="dashed",
        clip_on=False,
    )
    fig.add_artist(boundary)
    fig.text(
        0.5,
        0.93,
        "One Figure (the dashed boundary - the whole canvas, what fig.savefig() writes)",
        ha="center",
        fontsize=10,
        color=WARNING,
        fontweight="bold",
    )
    return fig


def broadcasting_diagram() -> plt.Figure:
    """Draw a grid diagram of a (3,) array broadcasting against a (5, 3) matrix.

    The single row at the top represents the smaller array; the dashed
    arrows show it being virtually repeated for every row of the larger
    array below, which is the mental model Part 4's broadcasting rule
    depends on.

    Returns:
        The matplotlib Figure, ready to display in a notebook cell.
    """
    n_rows, n_cols = 5, 3
    fig, ax = plt.subplots(figsize=(5, 4.5))

    # The (5, 3) matrix: a grid of cells, one row per sample.
    for r in range(n_rows):
        for c in range(n_cols):
            ax.add_patch(plt.Rectangle((c, -r), 0.92, 0.92, facecolor="#EAF3FA", edgecolor=INFO, linewidth=1))
    ax.text(n_cols / 2, -n_rows - 0.3, "shape (5, 3)", ha="center", color=INFO, fontweight="bold")

    # The (3,) array: one row, drawn above the matrix, with dashed arrows
    # showing it "stretching" down to every row it gets broadcast against.
    for c in range(n_cols):
        ax.add_patch(plt.Rectangle((c, 1.6), 0.92, 0.92, facecolor="#FFF1E6", edgecolor=WARNING, linewidth=1))
        ax.annotate(
            "",
            xy=(c + 0.46, -n_rows + 1.1),
            xytext=(c + 0.46, 1.55),
            arrowprops={"arrowstyle": "->", "color": WARNING, "linestyle": "dashed", "alpha": 0.6},
        )
    ax.text(n_cols / 2, 2.75, "shape (3,)", ha="center", color=WARNING, fontweight="bold")
    ax.text(
        n_cols / 2,
        -n_rows / 2 + 0.5,
        "broadcast\n(no data copied)",
        ha="center",
        va="center",
        fontsize=9,
        color=TEXT_MUTED,
        style="italic",
    )

    ax.set_xlim(-0.3, n_cols + 0.3)
    ax.set_ylim(-n_rows - 0.8, 3.1)
    ax.set_title("X - feature_mean: (5, 3) and (3,)", fontsize=11, color=PRIMARY, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    return fig


def merge_join_types_diagram() -> plt.Figure:
    """Draw the four `pd.merge` join types as overlapping-circle diagrams.

    Each panel shows the same two circles, "left" keys and "right" keys,
    with the kept region shaded: only the overlap for inner, all of left
    plus the overlap for left, all of right plus the overlap for right, and
    everything for outer. The circles never change, only what gets shaded
    does, which is the point: a join type is a choice about which rows to
    keep, not a different way of matching them.

    Returns:
        The matplotlib Figure, ready to display in a notebook cell.
    """
    joins = [
        ("inner", "intersection only", False, True, False),
        ("left", "all of left, matched right", True, True, False),
        ("right", "all of right, matched left", False, True, True),
        ("outer", "everything, matched or not", True, True, True),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(10, 3))

    for ax, (name, subtitle, keep_left_only, keep_overlap, keep_right_only) in zip(axes, joins, strict=True):
        left = Circle((-0.45, 0), 1.0, facecolor="none", edgecolor=INFO, linewidth=1.6)
        right = Circle((0.45, 0), 1.0, facecolor="none", edgecolor=WARNING, linewidth=1.6)

        if keep_left_only:
            ax.add_patch(Circle((-0.45, 0), 1.0, facecolor=INFO, alpha=0.25, edgecolor="none"))
        if keep_right_only:
            ax.add_patch(Circle((0.45, 0), 1.0, facecolor=WARNING, alpha=0.25, edgecolor="none"))
        if keep_overlap:
            overlap = Circle((0, 0), 0.55, facecolor=SUCCESS, alpha=0.45, edgecolor="none")
            ax.add_patch(overlap)

        ax.add_patch(left)
        ax.add_patch(right)
        ax.text(-0.95, 1.15, "left", color=INFO, fontsize=9, fontweight="bold", ha="center")
        ax.text(0.95, 1.15, "right", color=WARNING, fontsize=9, fontweight="bold", ha="center")
        ax.set_title(f'how="{name}"', fontsize=11, color=PRIMARY, fontweight="bold")
        ax.text(0, -1.4, subtitle, ha="center", fontsize=8.5, color=TEXT_MUTED, style="italic")
        ax.set_xlim(-1.7, 1.7)
        ax.set_ylim(-1.8, 1.5)
        ax.set_aspect("equal")
        ax.axis("off")

    fig.suptitle("pd.merge(left, right, on=key, how=...) - shaded rows are kept", fontsize=11, color=PRIMARY)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def groupby_split_apply_combine_diagram() -> plt.Figure:
    """Draw groupby as three stages: split, apply, combine.

    A small table of rows is split into groups by key, the same function
    is applied independently to each group, and the per-group results are
    combined back into a single table, one row per group. This is the
    mental model behind every `df.groupby(...)` call, regardless of which
    aggregation function ends up inside `.apply()`.

    Returns:
        The matplotlib Figure, ready to display in a notebook cell.
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    group_colors = [INFO, WARNING, SUCCESS]
    group_labels = ["program = CS", "program = DS", "program = IT"]

    # Stage 1: split - one small stack of rows per group.
    split_x = 0.5
    for i, color in enumerate(group_colors):
        y0 = 2.6 - i * 1.3
        for r in range(3):
            ax.add_patch(
                plt.Rectangle((split_x, y0 - r * 0.32), 1.1, 0.26, facecolor=color, alpha=0.25, edgecolor=color)
            )
        ax.text(split_x + 0.55, y0 + 0.5, group_labels[i], ha="center", fontsize=8.5, color=color, fontweight="bold")
    ax.text(split_x + 0.55, 4.1, "split", ha="center", fontsize=11, color=PRIMARY, fontweight="bold")

    # Stage 2: apply - the same function applied to each group independently.
    apply_x = 3.3
    for i, color in enumerate(group_colors):
        y0 = 2.6 - i * 1.3
        ax.add_patch(plt.Rectangle((apply_x, y0 - 0.55), 1.1, 0.75, facecolor="none", edgecolor=color, linewidth=1.6))
        ax.text(apply_x + 0.55, y0 - 0.17, ".mean()", ha="center", fontsize=8.5, color=color)
        ax.annotate(
            "",
            xy=(apply_x - 0.1, y0 - 0.17),
            xytext=(split_x + 1.2, y0 - 0.17),
            arrowprops={"arrowstyle": "->", "color": TEXT_MUTED, "alpha": 0.7},
        )
    ax.text(apply_x + 0.55, 4.1, "apply", ha="center", fontsize=11, color=PRIMARY, fontweight="bold")

    # Stage 3: combine - one result row per group, back in a single table.
    combine_x = 6.0
    combine_values = ["0.61", "0.58", "0.74"]
    for i, color in enumerate(group_colors):
        y0 = 2.6 - i * 1.3
        ax.add_patch(plt.Rectangle((combine_x, y0 - 0.55), 1.6, 0.34, facecolor=color, alpha=0.3, edgecolor=color))
        ax.text(
            combine_x + 0.8,
            y0 - 0.38,
            f"{group_labels[i].split(' = ')[1]}: {combine_values[i]}",
            ha="center",
            fontsize=8,
            color=PRIMARY,
        )
        ax.annotate(
            "",
            xy=(combine_x - 0.1, y0 - 0.38),
            xytext=(apply_x + 1.2, y0 - 0.17),
            arrowprops={"arrowstyle": "->", "color": TEXT_MUTED, "alpha": 0.7},
        )
    ax.text(combine_x + 0.8, 4.1, "combine", ha="center", fontsize=11, color=PRIMARY, fontweight="bold")

    ax.set_xlim(0, 8)
    ax.set_ylim(-1.3, 4.5)
    ax.axis("off")
    ax.set_title(
        'df.groupby("program")["final_score"].mean() - one function, run once per group',
        fontsize=10.5,
        color=PRIMARY,
    )
    fig.tight_layout()
    return fig


def _curved_flow_arrow(
    ax: plt.Axes,
    xy_start: tuple[float, float],
    xy_end: tuple[float, float],
    color: str,
    rad: float = 0.15,
    linewidth: float = 1.6,
) -> None:
    """A single curved, single-headed arrow, the shared idiom every panel below uses for power flow."""
    ax.add_patch(
        FancyArrowPatch(
            xy_start,
            xy_end,
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>",
            mutation_scale=14,
            color=color,
            linewidth=linewidth,
            shrinkA=6,
            shrinkB=6,
        )
    )


def _grid_tier_zone(ax: plt.Axes, xy: tuple[float, float], width: float, height: float, label: str) -> None:
    """A soft, dashed rounded-rectangle zone with its label sitting above it, not fighting the icons inside."""
    ax.add_patch(
        FancyBboxPatch(
            xy,
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.1,
            linestyle=(0, (4, 3)),
            edgecolor=TEXT_MUTED,
            facecolor="none",
            alpha=0.55,
        )
    )
    ax.text(
        xy[0] + width / 2,
        xy[1] + height + 0.1,
        label,
        fontsize=8,
        color=TEXT_MUTED,
        style="italic",
        ha="center",
        va="bottom",
        alpha=0.85,
    )


def _grid_panel(ax: plt.Axes, title: str, decentralized: bool) -> None:
    """One side of the centralized/decentralized comparison; see `centralized_vs_decentralized_grid_diagram`."""
    # Deliberately not a repeated template: the two panels use their own
    # hand-picked coordinates, box sizes, and arrow curvatures rather than
    # one geometry shared by both and recolored, the thing that made the
    # first version of this diagram read as auto-generated despite the
    # curved arrows. Every number below was chosen for this specific panel.
    if decentralized:
        plant_xy = (0.53, 5.62)
        sub_xy = (0.49, 3.7)
        house_xys = [(0.11, 1.1), (0.52, 0.86), (0.88, 1.16)]
        zones = [(0.0, 4.9, 1.0, 1.02), (0.01, 2.7, 0.99, 1.5), (0.03, 0.02, 0.95, 2.08)]
        node_r, sub_r = 0.175, 0.235
        icon_sizes = (26, 17, 12)
    else:
        plant_xy = (0.47, 5.5)
        sub_xy = (0.45, 3.62)
        house_xys = [(0.16, 1.0), (0.48, 0.83), (0.83, 1.04)]
        zones = [(0.03, 4.82, 0.94, 1.08), (0.0, 2.62, 1.0, 1.58), (0.01, 0.08, 0.98, 2.0)]
        node_r, sub_r = 0.165, 0.245
        icon_sizes = (25, 18, 12)

    _grid_tier_zone(ax, zones[0][:2], zones[0][2], zones[0][3], "HV transmission")
    _grid_tier_zone(ax, zones[1][:2], zones[1][2], zones[1][3], "MV substation")
    _grid_tier_zone(ax, zones[2][:2], zones[2][2], zones[2][3], "LV feeder")

    # Hand-picked per panel too: a real illustrator wouldn't re-derive one
    # arrow curvature from a loop index for six different arrows, they'd
    # nudge each one until it looked right. These numbers are that nudging.
    if decentralized:
        plant_sub_rad = 0.13
        down_rads = [-0.19, -0.04, 0.09]
        up_rads = [-0.06, 0.07, 0.19]
    else:
        plant_sub_rad = 0.06
        feed_rads = [-0.16, 0.01, 0.13]

    ax.text(
        plant_xy[0],
        plant_xy[1],
        ICONS["building-fill"],
        fontproperties=icon_font(icon_sizes[0]),
        color=PRIMARY,
        ha="center",
        va="center",
    )
    ax.text(plant_xy[0], plant_xy[1] - 0.38, "power plant", fontsize=7.5, color=TEXT_MUTED, ha="center")

    ax.add_patch(Circle(sub_xy, sub_r, facecolor="white", edgecolor=PRIMARY, linewidth=1.4))
    ax.text(
        sub_xy[0],
        sub_xy[1],
        ICONS["hdd-network-fill"],
        fontproperties=icon_font(icon_sizes[1]),
        color=PRIMARY,
        ha="center",
        va="center",
    )
    ax.text(sub_xy[0], sub_xy[1] - 0.42, "substation", fontsize=7.5, color=TEXT_MUTED, ha="center")

    _curved_flow_arrow(
        ax, (plant_xy[0], plant_xy[1] - 0.3), (sub_xy[0], sub_xy[1] + 0.3), TEXT_MUTED, rad=plant_sub_rad
    )

    for i, house_xy in enumerate(house_xys):
        if decentralized:
            # Two distinct, offset arrows rather than one double-headed
            # arrow: at this scale a "<->" arrowhead reads as one
            # direction, not two. Grey supplies down, amber exports back up.
            _curved_flow_arrow(
                ax,
                (sub_xy[0] - 0.05, sub_xy[1] - 0.28),
                (house_xy[0] - 0.05, house_xy[1] + 0.24),
                TEXT_MUTED,
                rad=down_rads[i],
                linewidth=1.2,
            )
            _curved_flow_arrow(
                ax,
                (house_xy[0] + 0.05, house_xy[1] + 0.24),
                (sub_xy[0] + 0.05, sub_xy[1] - 0.28),
                WARNING,
                rad=up_rads[i],
                linewidth=1.4,
            )
        else:
            _curved_flow_arrow(
                ax,
                (sub_xy[0], sub_xy[1] - 0.28),
                (house_xy[0], house_xy[1] + 0.24),
                TEXT_MUTED,
                rad=feed_rads[i],
                linewidth=1.3,
            )

        # The third house on the decentralized side reads DANGER (a voltage
        # violation), not because that's inevitable, but because Chapter 1
        # of Part 4 spends real effort establishing that DER growth can push
        # a feeder past its limits; a diagram that shows only green would
        # undersell the point the surrounding chapter is actually making.
        node_color = DANGER if (decentralized and i == 2) else SUCCESS
        ax.add_patch(Circle(house_xy, node_r, facecolor=node_color, edgecolor="white", linewidth=1.2, zorder=3))
        ax.text(
            house_xy[0],
            house_xy[1],
            ICONS["house-fill"],
            fontproperties=icon_font(icon_sizes[2]),
            color="white",
            ha="center",
            va="center",
            zorder=4,
        )
        if decentralized and i != 1:
            badge_icon = "sun-fill" if i == 0 else "ev-front-fill"
            badge_color = WARNING if i == 0 else PRIMARY
            badge_xy = (house_xy[0] + 0.16, house_xy[1] + 0.15)
            ax.add_patch(Circle(badge_xy, 0.105, facecolor=badge_color, edgecolor="white", linewidth=0.8, zorder=5))
            ax.text(
                badge_xy[0],
                badge_xy[1],
                ICONS[badge_icon],
                fontproperties=icon_font(9),
                color="white",
                ha="center",
                va="center",
                zorder=6,
            )
        if i == 1:
            # One labeled example, not one badge per house: the point is
            # "this is where a smart meter sits," a concept, not a claim
            # that every house in this schematic has one. Same icon the
            # book's own cover page already uses for "Meter Signal," so
            # the two pages read as the same visual language.
            meter_xy = (house_xy[0] + 0.19, house_xy[1] - 0.16)
            ax.add_patch(Circle(meter_xy, 0.105, facecolor=INFO, edgecolor="white", linewidth=0.8, zorder=5))
            ax.text(
                meter_xy[0],
                meter_xy[1],
                ICONS["speedometer2"],
                fontproperties=icon_font(9),
                color="white",
                ha="center",
                va="center",
                zorder=6,
            )
            ax.text(
                meter_xy[0] + 0.14,
                meter_xy[1],
                "smart meter",
                fontsize=6.5,
                color=INFO,
                ha="left",
                va="center",
                style="italic",
            )

    caption = "power flows both ways" if decentralized else "power flows one way only"
    caption_color = WARNING if decentralized else TEXT_MUTED
    ax.text(
        0.5,
        -0.2,
        caption,
        fontsize=8.5,
        color=caption_color,
        ha="center",
        style="italic",
        fontweight="bold" if decentralized else "normal",
    )

    ax.set_title(title, fontsize=11.5, color=PRIMARY, fontweight="bold", loc="center", pad=8)
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.42, 6.25)
    ax.set_aspect("equal")
    ax.axis("off")


def centralized_vs_decentralized_grid_diagram() -> plt.Figure:
    """Draw the centralized-vs-decentralized grid comparison Part 4's first chapter opens with.

    Left panel: the traditional grid, one-way power flow from a plant
    through a substation down to passive customers. Right panel: the same
    chain, but with rooftop solar and EV charging at the low-voltage tier,
    where power now flows both ways, and one customer is deliberately
    marked as a voltage violation, the exact strain the surrounding chapter
    spends its first few sections establishing.

    Returns:
        The matplotlib Figure, ready to display in a notebook cell.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 5.4))
    _grid_panel(axes[0], "Centralized grid", decentralized=False)
    _grid_panel(axes[1], "Decentralized, DER-rich grid", decentralized=True)
    fig.tight_layout()
    plt.close(fig)
    return fig


def _failure_mode_panel(
    ax: plt.Axes,
    title: str,
    badge_icon: str,
    badge_color: str,
    arrow_up: bool,
    arrow_color: str,
    gauge_label: str,
    gauge_color: str,
) -> None:
    """One panel of `pv_vs_ev_failure_mode_diagram`: a substation, one house, and one flow arrow."""
    sub_xy = (0.5, 1.42)
    house_xy = (0.5, 0.32)

    ax.add_patch(Circle(sub_xy, 0.26, facecolor="white", edgecolor=PRIMARY, linewidth=1.6, zorder=2))
    ax.text(
        sub_xy[0],
        sub_xy[1],
        ICONS["hdd-network-fill"],
        fontproperties=icon_font(19),
        color=PRIMARY,
        ha="center",
        va="center",
        zorder=3,
    )

    ax.add_patch(Circle(house_xy, 0.22, facecolor=SUCCESS, edgecolor="white", linewidth=1.6, zorder=2))
    ax.text(
        house_xy[0],
        house_xy[1],
        ICONS["house-fill"],
        fontproperties=icon_font(15),
        color="white",
        ha="center",
        va="center",
        zorder=3,
    )
    badge_xy = (house_xy[0] + 0.22, house_xy[1] + 0.17)
    ax.add_patch(Circle(badge_xy, 0.135, facecolor=badge_color, edgecolor="white", linewidth=1.0, zorder=4))
    ax.text(
        badge_xy[0],
        badge_xy[1],
        ICONS[badge_icon],
        fontproperties=icon_font(11),
        color="white",
        ha="center",
        va="center",
        zorder=5,
    )

    start, end = (house_xy, sub_xy) if arrow_up else (sub_xy, house_xy)
    _curved_flow_arrow(
        ax,
        (start[0] + 0.07, start[1] + (0.28 if arrow_up else -0.28)),
        (end[0] + 0.07, end[1] + (-0.28 if arrow_up else 0.28)),
        arrow_color,
        rad=0.2 if arrow_up else -0.2,
        linewidth=2.4,
    )

    gauge_xy = (0.98, 1.42)
    ax.add_patch(Circle(gauge_xy, 0.155, facecolor="white", edgecolor=gauge_color, linewidth=2.6, zorder=2))
    ax.text(
        gauge_xy[0],
        gauge_xy[1],
        ICONS["speedometer2"],
        fontproperties=icon_font(11),
        color=gauge_color,
        ha="center",
        va="center",
        zorder=3,
    )
    ax.text(
        gauge_xy[0],
        gauge_xy[1] - 0.24,
        gauge_label,
        fontsize=7.5,
        color=gauge_color,
        ha="center",
        va="top",
        fontweight="bold",
    )

    ax.set_title(title, fontsize=13, color=PRIMARY, fontweight="bold", loc="center", pad=10)
    ax.set_xlim(0.02, 1.24)
    ax.set_ylim(-0.02, 1.85)
    ax.set_aspect("equal")
    ax.axis("off")


def pv_vs_ev_failure_mode_diagram() -> plt.Figure:
    """Draw the PV-vs-EV contrast Chapter 2 builds its second half around.

    Same feeder, same house, two different real problems: PV exports
    power at midday and pushes voltage toward its statutory ceiling
    first; EV charging imports power at the evening peak and pushes the
    transformer toward its thermal rating first, the opposite direction,
    the opposite time of day, and a different binding constraint, exactly
    the finding Chapter 2's own hosting-capacity sweeps land on.

    Returns:
        The matplotlib Figure, ready to display in a notebook cell.
    """
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.6))
    _failure_mode_panel(
        axes[0],
        "PV, midday",
        badge_icon="sun-fill",
        badge_color=WARNING,
        arrow_up=True,
        arrow_color=WARNING,
        gauge_label="voltage\nbinds first",
        gauge_color=DANGER,
    )
    _failure_mode_panel(
        axes[1],
        "EV, evening",
        badge_icon="ev-front-fill",
        badge_color=INFO,
        arrow_up=False,
        arrow_color=INFO,
        gauge_label="thermal\nbinds first",
        gauge_color=DANGER,
    )
    fig.tight_layout()
    plt.close(fig)
    return fig


def voltage_correlation_vs_magnitude_diagram() -> plt.Figure:
    """Draw the "correlation, not magnitude" contrast Chapter 3's phase-identification method rests on.

    Three synthetic customers on a feeder: two share phase A but sit at
    different distances from the transformer, so their absolute voltage
    levels differ; the third sits on phase B, but happens to land at an
    absolute level between the other two. Raw magnitude (left) groups by
    proximity to the transformer, pairing the wrong two customers.
    Correlation (right) reveals which voltage traces actually move
    together, correctly pairing the two customers who share phase A, the
    same distinction the surrounding Key Concept box states in words.

    Returns:
        The matplotlib Figure, ready to display in a notebook cell.
    """
    hours = np.linspace(0, 24, 96)
    phase_a_fluctuation = 0.02 * np.sin(2 * np.pi * (hours - 6) / 24) + 0.01 * np.sin(2 * np.pi * hours / 6)
    phase_b_fluctuation = 0.02 * np.sin(2 * np.pi * (hours - 14) / 24) + 0.008 * np.sin(2 * np.pi * (hours + 2) / 5)

    near_transformer = 1.02 + phase_a_fluctuation
    far_on_phase_a = 0.97 + 1.3 * phase_a_fluctuation
    on_phase_b = 1.00 + phase_b_fluctuation

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))

    ax = axes[0]
    ax.plot(hours, near_transformer, color=WARNING, linewidth=2.2, label="Customer A1 (phase A, near)")
    ax.plot(hours, far_on_phase_a, color=INFO, linewidth=2.2, label="Customer A2 (phase A, far)")
    ax.plot(hours, on_phase_b, color=DANGER, linewidth=2.2, linestyle="--", label="Customer B1 (phase B)")
    ax.set_ylabel("Voltage (p.u.)")
    ax.set_title("Cluster on raw magnitude", fontsize=12, color=PRIMARY, fontweight="bold")
    ax.text(
        0.5,
        -0.24,
        "Wrong pair: A1 and B1 sit closest in level",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        color=DANGER,
        fontweight="bold",
    )

    ax = axes[1]
    ax.plot(hours, near_transformer - near_transformer.mean(), color=WARNING, linewidth=2.2, label="Customer A1")
    ax.plot(hours, far_on_phase_a - far_on_phase_a.mean(), color=INFO, linewidth=2.2, label="Customer A2")
    ax.plot(
        hours,
        on_phase_b - on_phase_b.mean(),
        color=DANGER,
        linewidth=2.2,
        linestyle="--",
        label="Customer B1",
    )
    ax.set_ylabel("Voltage fluctuation (centered)")
    ax.set_title("Cluster on correlation", fontsize=12, color=PRIMARY, fontweight="bold")
    ax.text(
        0.5,
        -0.24,
        "Right pair: A1 and A2 move together",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        color=SUCCESS,
        fontweight="bold",
    )

    for ax in axes:
        ax.set_xlabel("Hour of day")
        ax.set_xlim(0, 24)
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.legend(loc="upper left", fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    plt.close(fig)
    return fig


def _household_curve(peak: float, *, hours: np.ndarray) -> np.ndarray:
    """A synthetic evening-peaking household load curve, scaled to a given peak (kW)."""
    shape = 0.15 + 0.85 * np.exp(-0.5 * ((hours - 19) / 2.3) ** 2)
    return shape * peak


def shape_vs_magnitude_diagram() -> plt.Figure:
    """Draw the "shape, not magnitude" contrast Chapter 4's clustering feature choice rests on.

    Two households share the same evening-peaking rhythm but draw very
    different absolute power: a small apartment (peak 1.0 kW) and a large
    house (peak 4.5 kW). Clustering on raw magnitude (left) separates them
    by size; clustering on peak-normalized shape (right), the choice this
    chapter actually makes, groups them together because their rhythm
    matches, the same distinction the surrounding Key Concept box states
    in words.

    Returns:
        The matplotlib Figure, ready to display in a notebook cell.
    """
    hours = np.linspace(0, 24, 96)
    small = _household_curve(1.0, hours=hours)
    large = _household_curve(4.5, hours=hours)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))

    ax = axes[0]
    ax.plot(hours, small, color=WARNING, linewidth=2.2, label="Household A (peak 1.0 kW)")
    ax.plot(hours, large, color=INFO, linewidth=2.2, label="Household B (peak 4.5 kW)")
    ax.set_ylabel("Demand (kW)")
    ax.set_title("Cluster on raw magnitude", fontsize=12, color=PRIMARY, fontweight="bold")
    ax.text(
        0.5,
        -0.24,
        "Different clusters: far apart in kW",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        color=DANGER,
        fontweight="bold",
    )

    ax = axes[1]
    ax.plot(hours, small / small.max(), color=WARNING, linewidth=2.2, label="Household A")
    ax.plot(hours, large / large.max(), color=INFO, linewidth=2.2, linestyle="--", label="Household B")
    ax.set_ylabel("Normalized demand")
    ax.set_title("Cluster on peak-normalized shape", fontsize=12, color=PRIMARY, fontweight="bold")
    ax.text(
        0.5,
        -0.24,
        "Same cluster: same evening rhythm",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        color=SUCCESS,
        fontweight="bold",
    )

    for ax in axes:
        ax.set_xlabel("Hour of day")
        ax.set_xlim(0, 24)
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.legend(loc="upper left", fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    plt.close(fig)
    return fig


def cluster_confidence_set_diagram() -> plt.Figure:
    """Draw the geometry behind a conformal cluster-membership confidence set.

    Three archetype centroids in embedding space, each surrounded by a
    circle of radius equal to the calibrated conformal threshold. A
    customer landing inside exactly one circle gets a confident,
    single-archetype set (Customer 1). One landing inside two overlapping
    circles gets an honest two-archetype set instead of a forced single
    guess (Customer 2). One landing outside every circle gets an empty
    set, itself a signal that this customer's own behavior does not
    confidently resemble any archetype (Customer 3). The same geometric
    idea Chapter 3 uses for phase centroids and Chapter 4 uses for
    archetype centroids.

    Returns:
        The matplotlib Figure, ready to display in a notebook cell.
    """
    fig, ax = plt.subplots(figsize=(6.4, 5.2))

    centroids = {
        "Archetype 1": ((-1.4, 0.6), INFO, "left"),
        "Archetype 2": ((1.4, 0.6), WARNING, "right"),
        "Archetype 3": ((0.0, -1.7), SUCCESS, "below"),
    }
    threshold_radius = 1.05

    for name, (xy, color, side) in centroids.items():
        ax.add_patch(Circle(xy, threshold_radius, facecolor=color, alpha=0.15, edgecolor=color, linewidth=1.8))
        ax.scatter(*xy, s=70, color=color, zorder=3)
        if side == "left":
            label_xy, ha = (xy[0] - threshold_radius - 0.1, xy[1]), "right"
        elif side == "right":
            label_xy, ha = (xy[0] + threshold_radius + 0.1, xy[1]), "left"
        else:
            label_xy, ha = (xy[0], xy[1] - threshold_radius - 0.25), "center"
        ax.text(*label_xy, name, ha=ha, va="center", fontsize=9.5, color=color, fontweight="bold")

    customers = [
        ("Customer 1", "confident: {1}", (-1.4, 1.35), (0.0, 0.32), "center", TEXT_MUTED),
        ("Customer 2", "ambiguous: {1, 2}", (0.0, 0.6), (0.0, 0.32), "center", TEXT_MUTED),
        ("Customer 3", "no match: {}", (2.5, -0.6), (0.0, 0.32), "center", DANGER),
    ]
    for name, status, xy, text_offset, ha, color in customers:
        ax.scatter(*xy, s=90, marker="x", color=color, linewidth=2.4, zorder=4)
        ax.text(
            xy[0] + text_offset[0],
            xy[1] + text_offset[1],
            f"{name}\n{status}",
            fontsize=8,
            color=color,
            ha=ha,
            va="bottom",
        )

    ax.set_xlim(-3.0, 3.2)
    ax.set_ylim(-3.2, 2.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "Each circle's radius is the calibrated threshold\ninside one, inside two, or inside none",
        fontsize=10,
        color=PRIMARY,
        fontweight="bold",
    )

    fig.tight_layout()
    plt.close(fig)
    return fig


def _lever_checklist_panel(
    ax: plt.Axes,
    title: str,
    badge_icon: str,
    badge_color: str,
    available: dict[str, bool],
) -> None:
    """One panel of `lever_applicability_diagram`: a house, a badge, and a checklist of levers."""
    house_xy = (0.28, 0.5)
    ax.add_patch(Circle(house_xy, 0.2, facecolor=PRIMARY, edgecolor="white", linewidth=1.6, zorder=2))
    ax.text(
        house_xy[0],
        house_xy[1],
        ICONS["house-fill"],
        fontproperties=icon_font(15),
        color="white",
        ha="center",
        va="center",
        zorder=3,
    )
    badge_xy = (house_xy[0] + 0.16, house_xy[1] + 0.15)
    ax.add_patch(Circle(badge_xy, 0.1, facecolor=badge_color, edgecolor="white", linewidth=1.0, zorder=4))
    ax.text(
        badge_xy[0],
        badge_xy[1],
        ICONS[badge_icon],
        fontproperties=icon_font(8),
        color="white",
        ha="center",
        va="center",
        zorder=5,
    )

    list_x = 0.58
    top_y = 0.86
    row_height = 0.19
    for row, (lever, is_available) in enumerate(available.items()):
        y = top_y - row * row_height
        mark_color = SUCCESS if is_available else DANGER
        mark = "✓" if is_available else "✕"
        ax.text(list_x, y, mark, color=mark_color, fontsize=13, fontweight="bold", ha="center", va="center")
        label_color = PRIMARY if is_available else TEXT_MUTED
        ax.text(list_x + 0.08, y, lever, color=label_color, fontsize=10, ha="left", va="center")

    ax.set_title(title, fontsize=12, color=PRIMARY, fontweight="bold", pad=10)
    ax.set_xlim(0, 1.5)
    ax.set_ylim(0.05, 1.0)
    ax.axis("off")


def lever_applicability_diagram() -> plt.Figure:
    """Draw which mitigation levers can even act on a voltage violation versus a thermal one.

    Volt-Watt and Volt-VAr are `InvControl` elements: they only govern a
    `PVSystem` element's own real and reactive power output as local
    voltage moves. A customer whose violation comes from EV charging
    pushing the transformer's own thermal loading has no PVSystem for
    either lever to control at all, not a weaker option, an inapplicable
    one. Storage is the only lever that reaches both violation types,
    the same physical constraint the recommendation section's own case-
    based retrieval is scoped around.

    Returns:
        The matplotlib Figure, ready to display in a notebook cell.
    """
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
    _lever_checklist_panel(
        axes[0],
        "Voltage violation (PV export)",
        badge_icon="sun-fill",
        badge_color=WARNING,
        available={"Volt-Watt": True, "Volt-VAr": True, "Storage": True, "No action": True},
    )
    _lever_checklist_panel(
        axes[1],
        "Thermal violation (EV charging)",
        badge_icon="ev-front-fill",
        badge_color=INFO,
        available={"Volt-Watt": False, "Volt-VAr": False, "Storage": True, "No action": True},
    )
    fig.tight_layout()
    plt.close(fig)
    return fig


def _feeder_line_panel(ax: plt.Axes, title: str, subtitle: str) -> tuple[tuple[float, float], tuple[float, float]]:
    """Draw a transformer and two customers at different distances along one feeder line.

    Returns the (near, far) customer node positions so the caller can add
    a panel-specific annotation on top of the shared schematic.
    """
    transformer_xy = (0.3, 0.5)
    near_xy = (1.6, 0.5)
    far_xy = (3.4, 0.5)

    ax.plot([transformer_xy[0], far_xy[0]], [0.5, 0.5], color=TEXT_MUTED, linewidth=2.0, zorder=1)

    ax.add_patch(Circle(transformer_xy, 0.22, facecolor=PRIMARY, edgecolor="white", linewidth=1.2, zorder=3))
    ax.text(
        *transformer_xy,
        ICONS["hdd-network-fill"],
        fontproperties=icon_font(15),
        color="white",
        ha="center",
        va="center",
        zorder=4,
    )

    for xy, label in [(near_xy, "Customer near"), (far_xy, "Customer far")]:
        ax.add_patch(Circle(xy, 0.19, facecolor=SUCCESS, edgecolor="white", linewidth=1.2, zorder=3))
        ax.text(
            *xy, ICONS["house-fill"], fontproperties=icon_font(12), color="white", ha="center", va="center", zorder=4
        )
        ax.text(xy[0], xy[1] - 0.42, label, fontsize=8, color=TEXT_MUTED, ha="center")
        ax.text(xy[0], xy[1] + 0.32, "same real shape", fontsize=7.5, color=TEXT_MUTED, ha="center", style="italic")

    ax.set_title(title, fontsize=11.5, color=PRIMARY, fontweight="bold", pad=14)
    ax.text(2.0, -0.35, subtitle, fontsize=8.5, color=TEXT_MUTED, ha="center", style="italic", wrap=True)
    ax.set_xlim(-0.3, 4.0)
    ax.set_ylim(-0.6, 1.9)
    ax.set_aspect("equal")
    ax.axis("off")
    return near_xy, far_xy


def voltage_vs_thermal_position_diagram() -> plt.Figure:
    """Draw why bus position matters for voltage risk but barely matters for thermal risk.

    Two customers share the exact same real smart-meter shape, only their
    distance from the transformer differs. Voltage rise depends on how
    much line impedance sits between a customer and the source, on top of
    how much they draw or export, something a smart-meter reading alone
    says nothing about, so the far customer sees a real, different
    voltage outcome (left). Thermal loading at the transformer depends
    only on how much power is flowing at its own peak moment, exactly
    what a smart-meter reading already captures, so position adds nothing
    once shape is known (right). The physical reason retrieval-from-shape
    works for one risk and struggles for the other, the same distinction
    the surrounding Key Concept box states in words.

    Returns:
        The matplotlib Figure, ready to display in a notebook cell.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))

    near_xy, far_xy = _feeder_line_panel(
        axes[0],
        "Voltage: depends on where you are",
        "Impedance between customer and source adds up with distance",
    )
    ax = axes[0]
    pu_y = near_xy[1] + 0.95
    ax.text(near_xy[0], pu_y, "1.01 pu", fontsize=9, color=SUCCESS, fontweight="bold", ha="center")
    ax.text(far_xy[0], pu_y, "0.96 pu", fontsize=9, color=DANGER, fontweight="bold", ha="center")
    ax.annotate(
        "",
        xy=(far_xy[0], pu_y - 0.14),
        xytext=(near_xy[0], pu_y - 0.14),
        arrowprops={"arrowstyle": "-|>", "color": WARNING, "linewidth": 1.6},
    )
    ax.text(2.5, pu_y + 0.28, "voltage drifts with distance", fontsize=7.5, color=WARNING, ha="center", style="italic")

    near_xy, far_xy = _feeder_line_panel(
        axes[1],
        "Thermal: depends on what you draw",
        "Transformer loading sums instantaneous power, position is invisible to it",
    )
    ax = axes[1]
    arrow_y = near_xy[1] + 0.85
    for xy in (near_xy, far_xy):
        _curved_flow_arrow(ax, (xy[0], arrow_y - 0.1), (0.3, arrow_y), INFO, rad=0.2)
    ax.text(2.0, arrow_y + 0.28, "same contribution either way", fontsize=7.5, color=INFO, ha="center", style="italic")

    fig.tight_layout()
    plt.close(fig)
    return fig


def _customer_row(ax: plt.Axes, y: float, n: int = 5) -> list[tuple[float, float]]:
    """Five evenly-spaced house icons along one row, returns their positions."""
    xs = np.linspace(0.6, 4.2, n)
    positions = [(x, y) for x in xs]
    for xy in positions:
        ax.add_patch(Circle(xy, 0.22, facecolor=SUCCESS, edgecolor="white", linewidth=1.2, zorder=3))
        ax.text(
            *xy, ICONS["house-fill"], fontproperties=icon_font(13), color="white", ha="center", va="center", zorder=4
        )
    return positions


def per_customer_vs_network_view_diagram() -> plt.Figure:
    """Draw why per-customer detection cannot see a coincident event that a network view catches immediately.

    Five real customers, the same real moment. Left: each customer's own
    self-baseline detector only ever sees its own history, and a
    regionally sunny day already sits inside that history, so nothing
    looks wrong. Right: the same five customers share one real
    transformer, and their real exports arrive at the same real moment;
    summed, they cross a real physical limit no single customer's own
    meter could ever reveal, the structural blind spot Section 2's own
    Key Concept box states in words.

    Returns:
        The matplotlib Figure, ready to display in a notebook cell.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8))

    ax = axes[0]
    positions = _customer_row(ax, 1.1)
    for xy in positions:
        ax.text(xy[0], xy[1] - 0.42, "own history", fontsize=7, color=TEXT_MUTED, ha="center", style="italic")
        ax.text(xy[0], xy[1] + 0.4, "✓", fontsize=13, color=SUCCESS, fontweight="bold", ha="center")
    ax.set_title(
        "Per-customer view: five isolated self-baselines",
        fontsize=10.8,
        color=PRIMARY,
        fontweight="bold",
        pad=14,
    )
    ax.text(
        2.4,
        -0.15,
        "A regionally sunny day already sits inside each customer's own real history",
        fontsize=8.2,
        color=TEXT_MUTED,
        ha="center",
        style="italic",
        wrap=True,
    )
    ax.set_xlim(-0.2, 5.0)
    ax.set_ylim(-0.5, 2.0)
    ax.set_aspect("equal")
    ax.axis("off")

    ax = axes[1]
    transformer_xy = (2.4, 0.2)
    positions = _customer_row(ax, 1.5)
    ax.add_patch(Circle(transformer_xy, 0.24, facecolor=DANGER, edgecolor="white", linewidth=1.4, zorder=3))
    ax.text(
        *transformer_xy,
        ICONS["hdd-network-fill"],
        fontproperties=icon_font(16),
        color="white",
        ha="center",
        va="center",
        zorder=4,
    )
    for xy in positions:
        ax.text(xy[0], xy[1] + 0.4, ICONS["sun-fill"], fontproperties=icon_font(11), color=WARNING, ha="center")
        _curved_flow_arrow(ax, (xy[0], xy[1] - 0.22), (transformer_xy[0], transformer_xy[1] + 0.24), DANGER, rad=0.12)
    ax.text(
        transformer_xy[0],
        transformer_xy[1] - 0.5,
        "real overload",
        fontsize=8.5,
        color=DANGER,
        fontweight="bold",
        ha="center",
    )
    ax.set_title(
        "Network view: the same real moment, summed",
        fontsize=10.8,
        color=PRIMARY,
        fontweight="bold",
        pad=14,
    )
    ax.text(
        2.4,
        -0.9,
        "Five individually unremarkable exports, one real, physical overload",
        fontsize=8.2,
        color=TEXT_MUTED,
        ha="center",
        style="italic",
        wrap=True,
    )
    ax.set_xlim(-0.2, 5.0)
    ax.set_ylim(-1.2, 2.2)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.tight_layout()
    plt.close(fig)
    return fig


def _persistent_signal(hours: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
    """A clean, strongly repeating daily cycle plus a little real noise."""
    daily = 0.5 + 0.45 * np.sin((hours - 6) / 24 * 2 * np.pi)
    return daily + rng.normal(0, 0.03, size=hours.shape)


def _volatile_signal(hours: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
    """The same average level and daily cycle, buried under much larger real noise."""
    daily = 0.5 + 0.15 * np.sin((hours - 6) / 24 * 2 * np.pi)
    return daily + rng.normal(0, 0.22, size=hours.shape)


def forecastability_diagram() -> plt.Figure:
    """Draw the "some signals are just harder to forecast" contrast this chapter opens with.

    Two signals share the same average level and the same underlying daily
    cycle. The left one repeats cleanly, day after day, real noise but not
    enough to hide the pattern: low entropy, a Hurst exponent above 0.5, the
    persistent case a model can learn from directly. The right one carries
    the same cycle under noise large enough to swamp it: high entropy, a
    Hurst exponent near the random-walk boundary, the volatile case no
    amount of modelling sophistication fixes on its own. This is the real
    distinction every metric in this chapter measures from a different
    angle, not a specific customer's own reading.

    Returns:
        The matplotlib Figure, ready to display in a notebook cell.
    """
    rng = np.random.default_rng(7)
    hours = np.linspace(0, 24 * 3, 300)  # three real days

    persistent = _persistent_signal(hours, rng=rng)
    volatile = _volatile_signal(hours, rng=rng)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), sharey=True)

    ax = axes[0]
    ax.plot(hours, persistent, color=SUCCESS, linewidth=1.6)
    ax.set_title("Persistent: forecastable", fontsize=12, color=PRIMARY, fontweight="bold")
    ax.text(
        0.5,
        -0.26,
        "Low entropy, Hurst > 0.5\nyesterday predicts today",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        color=SUCCESS,
        fontweight="bold",
    )

    ax = axes[1]
    ax.plot(hours, volatile, color=DANGER, linewidth=1.6)
    ax.set_title("Volatile: hard to forecast", fontsize=12, color=PRIMARY, fontweight="bold")
    ax.text(
        0.5,
        -0.26,
        "High entropy, Hurst near 0.5\nthe same cycle, buried in noise",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        color=DANGER,
        fontweight="bold",
    )

    for ax in axes:
        ax.set_xlabel("Hour (three real days)")
        ax.set_xlim(hours.min(), hours.max())
        ax.set_xticks([0, 24, 48, 72])
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Load (normalized)")

    fig.tight_layout()
    plt.close(fig)
    return fig


def _flow_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    label: str,
    *,
    color: str,
    filled: bool = True,
) -> None:
    """A single labeled rounded box, the shared unit every panel below is built from."""
    ax.add_patch(
        FancyBboxPatch(
            xy,
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.6,
            edgecolor=color,
            facecolor=color if filled else "none",
            alpha=1.0 if filled else 0.9,
        )
    )
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        label,
        fontsize=8.3,
        color="white" if filled else color,
        fontweight="bold",
        ha="center",
        va="center",
    )


def _flow_arrow(ax: plt.Axes, xy_start: tuple[float, float], xy_end: tuple[float, float], color: str) -> None:
    """A single straight downward flow arrow, the shared idiom both panels below use."""
    ax.add_patch(
        FancyArrowPatch(
            xy_start,
            xy_end,
            arrowstyle="-|>",
            mutation_scale=13,
            color=color,
            linewidth=1.5,
            shrinkA=2,
            shrinkB=2,
        )
    )


def known_vs_new_customer_diagram() -> plt.Figure:
    """Draw the two different mechanisms this chapter's real result calls for.

    Left: a customer the system already has real history for. Their own
    distance to each archetype centroid is just another feature, fed into
    the same single shared model as the lag features, a legitimate
    personalization signal because the model has already trained on this
    exact customer. Right: a brand-new customer with no history in any
    trained model yet. The same distance features, computed from whatever
    partial history exists, first pass through Chapter 5's own calibrated
    retrieval-confidence gate, reused directly, not reimplemented, before
    deciding whether to trust a distance-weighted blend of the already-
    trained archetype specialists or fall back to the safe global model.
    Two different answers to "how should clustering and forecasting
    combine," chosen by whether a customer is known or new, not by
    preference.

    Returns:
        The matplotlib Figure, ready to display in a notebook cell.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 5.0))

    # ---- left panel: known customer ----
    ax = axes[0]
    ax.set_title("Known customer", fontsize=12, color=PRIMARY, fontweight="bold")
    _flow_box(ax, (0.0, 3.6), 1.7, 0.6, "Lag features", color=INFO)
    _flow_box(ax, (2.1, 3.6), 1.9, 0.6, "Own archetype\ndistance", color=SUCCESS)
    _flow_box(ax, (0.9, 2.0), 2.2, 0.7, "One shared model", color=PRIMARY)
    _flow_arrow(ax, (0.85, 3.6), (1.6, 2.7), color=INFO)
    _flow_arrow(ax, (3.05, 3.6), (2.4, 2.7), color=SUCCESS)
    _flow_arrow(ax, (2.0, 2.0), (2.0, 1.0), color=PRIMARY)
    ax.text(
        2.0,
        0.55,
        "Personalized forecast",
        fontsize=9.5,
        color=SUCCESS,
        fontweight="bold",
        ha="center",
        va="center",
    )
    ax.text(
        2.0,
        4.5,
        "A static distance signature\ndoubles as a personalization signal",
        fontsize=8.2,
        color=TEXT_MUTED,
        style="italic",
        ha="center",
        va="bottom",
    )

    # ---- right panel: new customer ----
    ax = axes[1]
    ax.set_title("New customer", fontsize=12, color=PRIMARY, fontweight="bold")
    _flow_box(ax, (0.9, 4.3), 2.2, 0.6, "Partial-history distance", color=WARNING)
    _flow_arrow(ax, (2.0, 4.3), (2.0, 3.5), color=WARNING)
    ax.scatter([2.0], [3.15], s=1400, marker="D", facecolor="white", edgecolor=PRIMARY, linewidth=1.6, zorder=3)
    ax.text(2.0, 3.15, "Confident\nmatch?", fontsize=7.3, color=PRIMARY, fontweight="bold", ha="center", va="center")

    _flow_arrow(ax, (2.5, 2.9), (3.3, 2.15), color=SUCCESS)
    ax.text(3.45, 2.55, "yes", fontsize=8, color=SUCCESS, fontweight="bold", ha="left", va="center")
    _flow_box(ax, (2.75, 1.55), 2.1, 0.65, "Blend of archetype\nspecialists", color=SUCCESS)

    _flow_arrow(ax, (1.5, 2.9), (0.7, 2.15), color=DANGER)
    ax.text(0.55, 2.55, "no", fontsize=8, color=DANGER, fontweight="bold", ha="right", va="center")
    _flow_box(ax, (-0.35, 1.55), 2.1, 0.65, "Global model\n(safe fallback)", color=TEXT_MUTED)

    _flow_arrow(ax, (0.7, 1.55), (1.8, 0.75), color=TEXT_MUTED)
    _flow_arrow(ax, (3.8, 1.55), (2.2, 0.75), color=SUCCESS)
    ax.text(
        2.0,
        0.4,
        "Forecast, never worse than the floor",
        fontsize=8.6,
        color=PRIMARY,
        fontweight="bold",
        ha="center",
        va="center",
    )

    for ax in axes:
        ax.set_xlim(-0.8, 5.0)
        ax.set_ylim(0.0, 5.1)
        ax.set_aspect("equal")
        ax.axis("off")

    fig.tight_layout()
    plt.close(fig)
    return fig


def archetype_distance_and_conformal_diagram() -> plt.Figure:
    """Draw the two mechanics behind this chapter's real result.

    Distance-to-centroid weighting, and the split-conformal calibration
    that gates it. Left: a customer's own shape sits some distance from each of 5 archetype
    centroids. Those distances become a softmax over negative distance, the
    same mixture-of-experts idea Jacobs, Jordan, Nowlan and Hinton introduced
    in 1991, closer centroids get more weight, not a hard, winner-take-all
    assignment. Right: before trusting any of this for a new customer, a
    calibrated distance threshold, tau, decides whether a real match exists
    at all: tau is the (1-alpha) quantile of nearest-neighbor distances on a
    held-back calibration split, the split-conformal guarantee Lei, G'Sell,
    Rinaldo, Tibshirani and Wasserman formalized in 2018. A customer inside
    tau is trusted; one beyond it safely falls back to the blind model.

    Returns:
        The matplotlib Figure, ready to display in a notebook cell.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6))

    # ---- left panel: distance-to-centroid weighting ----
    ax = axes[0]
    ax.set_title("Distance becomes weight", fontsize=12, color=PRIMARY, fontweight="bold")
    customer_xy = (0.0, 0.0)
    rng = np.random.default_rng(3)
    angles = np.linspace(0, 2 * np.pi, 5, endpoint=False) + 0.3
    radii = np.array([1.0, 2.6, 1.8, 3.4, 2.2])
    weights = np.exp(-radii / 1.3)
    weights = weights / weights.sum()

    for angle, radius, weight in zip(angles, radii, weights, strict=True):
        cx, cy = radius * np.cos(angle), radius * np.sin(angle)
        ax.plot([customer_xy[0], cx], [customer_xy[1], cy], color=TEXT_MUTED, linewidth=1.0 + 5 * weight, zorder=1)
        ax.add_patch(
            Circle(
                (cx, cy),
                0.42,
                facecolor=SUCCESS,
                edgecolor="white",
                linewidth=1.2,
                zorder=2,
                alpha=0.55 + 0.45 * weight,
            )
        )
        ax.text(
            cx, cy, f"{weight:.0%}", fontsize=8, color="white", fontweight="bold", ha="center", va="center", zorder=3
        )

    ax.add_patch(Circle(customer_xy, 0.32, facecolor=PRIMARY, edgecolor="white", linewidth=1.4, zorder=4))
    ax.text(
        customer_xy[0],
        customer_xy[1] - 0.85,
        "New customer's\npartial-history shape",
        fontsize=8.2,
        color=PRIMARY,
        ha="center",
        va="top",
        fontweight="bold",
    )
    ax.text(
        0, 3.9, r"$w_i = e^{-d_i/T} \,/\, \sum_j e^{-d_j/T}$", fontsize=10.5, color=PRIMARY, ha="center", va="center"
    )
    ax.set_xlim(-4.0, 4.0)
    ax.set_ylim(-4.3, 4.3)
    ax.set_aspect("equal")
    ax.axis("off")

    # ---- right panel: split-conformal calibration ----
    ax = axes[1]
    ax.set_title('A calibrated "do we trust this" gate', fontsize=12, color=PRIMARY, fontweight="bold")
    calib_scores = rng.gamma(2.2, 0.35, size=400)
    tau = np.quantile(calib_scores, 0.9)
    counts, bin_edges = np.histogram(calib_scores, bins=28)
    bin_width = bin_edges[1] - bin_edges[0]
    for left, count in zip(bin_edges[:-1], counts, strict=True):
        trusted_bin = left + bin_width / 2 <= tau
        ax.bar(
            left,
            count,
            width=bin_width * 0.95,
            align="edge",
            color=SUCCESS if trusted_bin else DANGER,
            alpha=0.75,
            linewidth=0,
        )

    ax.axvline(tau, color=PRIMARY, linestyle="--", linewidth=1.6)
    ax.set_ylim(0, counts.max() * 1.35)
    ax.text(
        tau, counts.max() * 1.18, r"$\tau$ = 90th percentile", fontsize=9, color=PRIMARY, ha="center", fontweight="bold"
    )
    ax.annotate(
        "Trusted:\nblend the specialists",
        xy=(tau * 0.45, counts.max() * 0.55),
        fontsize=8.4,
        color=SUCCESS,
        ha="center",
        fontweight="bold",
    )
    ax.annotate(
        "Flagged:\nfall back to blind",
        xy=(tau * 1.5, counts.max() * 0.55),
        fontsize=8.4,
        color=DANGER,
        ha="center",
        fontweight="bold",
    )
    ax.set_xlabel("Distance to nearest labeled neighbor")
    ax.set_ylabel("Calibration customers")
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    plt.close(fig)
    return fig
