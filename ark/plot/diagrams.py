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
