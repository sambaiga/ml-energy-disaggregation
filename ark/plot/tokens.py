"""Brand tokens shared by every plot/table style in :mod:`ark.plot`.

These constants are extracted from the Anthony Faustine personal site's Quarto
theme (``assets/scss/_defaults.scss`` in the ``sambaiga.github.io`` repo) so
that figures and tables produced here read as part of the same brand when
embedded in a tutorial, a blog post, or a published book. They are plain
Python literals (not a live import) because the source of truth is an SCSS
file in a separate repo - update this module by hand if that file's palette
changes.
"""

from __future__ import annotations

# ── Grayscale ────────────────────────────────────────────────────────────────
WHITE = "#FFFFFF"
GRAY_100 = "#F8F9FA"
GRAY_300 = "#DEE2E6"
GRAY_400 = "#CED4DA"
GRAY_600 = "#6B7280"
GRAY_700 = "#495057"
GRAY_900 = "#212529"
BLACK = "#000000"

# ── Semantic brand colors (mirrors $primary/$success/... in _defaults.scss) ──
PRIMARY = "#1E293B"  # $teal -- slate-navy, the site's brand primary
SECONDARY = GRAY_700
SUCCESS = "#059669"  # $green
INFO = "#0369A1"  # $cyan -- also the site's link color
WARNING = "#EA580C"  # $orange
DANGER = "#DC2626"  # $red

# ── Accent colors (site-specific, beyond Bootstrap's semantic set) ───────────
ENERGY_ACCENT = "#10B981"
AI_ACCENT = "#8B5CF6"
PINK = "#FF777C"
ROSE = "#DB2777"
YELLOW = "#CA8A04"

# ── Text colors ────────────────────────────────────────────────────────────
TEXT_DARK = "#171717"  # $body-color
TEXT_MUTED = GRAY_600

# ── Surfaces ───────────────────────────────────────────────────────────────
SURFACE = WHITE  # $body-bg
SURFACE_MUTED = GRAY_100  # $code-block-bg
BORDER = GRAY_300

# Light tint of INFO/WARNING, for a fill that needs to read as "belongs to"
# that color's own category without competing with real data ink (a table's
# even-row stripe, a diagram's own highlighted-region fill).
INFO_TINT = "#EAF3FA"
WARNING_TINT = "#FFF1E6"

# ── Typography ─────────────────────────────────────────────────────────────
# Loaded as web fonts on the site; matplotlib needs them installed locally to
# render text in the same face, so callers should resolve against installed
# fonts (see plot_theme.resolve_font) rather than assuming these are present.
BODY_FONT_STACK = "'Libre Franklin', 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
HEADING_FONT_STACK = "'Jost', 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
BODY_FONT_CANDIDATES = ["Libre Franklin", "Segoe UI", "Roboto", "Helvetica Neue", "Arial"]
HEADING_FONT_CANDIDATES = ["Jost", "Segoe UI", "Roboto", "Helvetica Neue", "Arial"]

# ── Qualitative palette for multi-series plots ────────────────────────────────
# Starts with INFO (clear blue) as the standard first-series color, followed
# by WARNING (orange) for maximum hue contrast - the same blue/orange pairing
# used by Tableau, Observable, and Okabe-Ito. PRIMARY (#1E293B) is placed last
# because it is a near-black UI token that reads as text or axes on a white
# background rather than as a data series. The two greens (SUCCESS / ENERGY_ACCENT)
# are separated to positions 3 and 7 to reduce confusion under red-green
# colorblindness.
BRAND_PALETTE = [
    INFO,  # #0369A1 -- clear blue, universally reads as "series 1"
    WARNING,  # #EA580C -- orange, maximum hue contrast with blue
    SUCCESS,  # #059669 -- green, hue-distant from orange and blue
    DANGER,  # #DC2626 -- red (4th position reduces semantic alarm)
    AI_ACCENT,  # #8B5CF6 -- violet/purple
    PINK,  # #FF777C -- warm salmon-pink, hue-distant from violet
    ENERGY_ACCENT,  # #10B981 -- teal (separated from SUCCESS at position 3)
    YELLOW,  # #CA8A04 -- amber
    PRIMARY,  # #1E293B -- dark slate (rarely reached in charts)
    GRAY_600,  # #6B7280 -- muted last fallback
]

# ── Sequential palettes ───────────────────────────────────────────────────────
# Anchored on INFO at the dark end so heatmaps and density charts stay
# visually consistent with the categorical palette's primary blue.
BLUES_SEQUENTIAL = [
    "#F0F9FF",
    "#E0F2FE",
    "#BAE6FD",
    "#7DD3FC",
    "#38BDF8",
    "#0EA5E9",
    "#0284C7",
    INFO,
]

# ── Diverging palettes ────────────────────────────────────────────────────────
# Blue-red: anchored on INFO (cool) and DANGER (warm) so correlation matrices
# and gain/loss charts use the same semantic colors as categorical plots.
DIVERGING_BLUE_RED = [
    INFO,
    "#06B6D4",
    "#A7F3D0",
    "#F5F5F5",
    "#FECACA",
    "#EF4444",
    DANGER,
]

# Purple-orange: anchored on AI_ACCENT and WARNING to stay within the existing
# token set (avoids introducing a second purple at #7C3AED).
DIVERGING_PURPLE_ORANGE = [
    AI_ACCENT,
    "#A78BFA",
    "#E9D5FF",
    "#F5F5F5",
    "#FED7AA",
    "#FB923C",
    WARNING,
]

CATEGORICAL_PALETTE = [
    "#0369A1",  # INFO - Strong blue (best starter)
    "#EA580C",  # WARNING - Vibrant orange (excellent contrast)
    "#059669",  # SUCCESS - Rich teal-green
    "#8B5CF6",  # AI_ACCENT - Purple (great distinction)
    "#DC2626",  # DANGER - Red (careful, used in 4th position)
    "#10B981",  # ENERGY_ACCENT - Bright emerald
    "#DB2777",  # ROSE - Deep pink/magenta
    "#CA8A04",  # YELLOW - Warm amber (avoid using too early)
    "#1E40AF",  # Deeper blue variant
    "#7C3AED",  # Stronger violet
    "#F97316",  # Bright orange alternative
    "#14B8A6",  # Teal
    "#EC4899",  # Hot pink
    "#84CC16",  # Lime green
    "#6366F1",  # Indigo
]

SHAPE_PALETTE = [16, 17, 15, 18, 8, 5, 14, 9, 3, 4, 2, 6, 7, 11, 0, 13]
