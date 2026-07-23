"""Bootstrap Icons, rendered as matplotlib text glyphs.

Bootstrap Icons (https://icons.getbootstrap.com/, MIT licensed) is this
project's own site's icon set already (see the `bi bi-*` classes in
`_quarto.yml`). The vendored font here (`assets/bootstrap-icons.ttf`, see
`assets/BOOTSTRAP_ICONS_NOTICE.txt`) is the same one Quarto bundles into
`_book/site_libs/bootstrap/` at render time, converted once from `.woff` to
`.ttf`: matplotlib's bundled FreeType cannot open `.woff` directly
(`FT_Open_Face ... broken table`), only `.ttf`/`.otf`. Using it needs no new
runtime dependency - matplotlib is already required, and `fontTools` (used
only for that one conversion, not at import time here) was already a
transitive dependency.

Importing this module registers the font with matplotlib's font manager as
a side effect, the same pattern `ark.plot.matplot_theme.resolve_font` uses
for body fonts.
"""

from __future__ import annotations

from pathlib import Path

from matplotlib import font_manager
from matplotlib.font_manager import FontProperties

_FONT_PATH = Path(__file__).parent / "assets" / "bootstrap-icons.ttf"
_FONT_FAMILY = "bootstrap-icons"

font_manager.fontManager.addfont(str(_FONT_PATH))

# Bootstrap Icons v1.13.1 codepoints (icons.getbootstrap.com). Only the
# icons an actual diagram has needed so far; add more by looking up the
# codepoint on that site and copying it in here.
ICONS: dict[str, str] = {
    "house-fill": "",
    "sun-fill": "",
    "lightning-charge-fill": "",
    "hdd-network-fill": "",
    "record-circle-fill": "",
    "building-fill": "",
    "ev-front-fill": "",
    "arrow-down-up": "",
    "speedometer2": "",
    "scissors": "",
    "dice-5-fill": "",
    "check-circle-fill": "",
    "x-circle-fill": "",
}


def icon_font(size: float = 24) -> FontProperties:
    """A `FontProperties` for drawing a Bootstrap Icon glyph via `ax.text`.

    Args:
        size: Glyph size in points, same units as any other matplotlib
            font size.

    Returns:
        A `FontProperties` set to the vendored Bootstrap Icons font.

    Examples:
        >>> ax.text(0, 0, ICONS["house-fill"], fontproperties=icon_font(28), color=INFO, ha="center", va="center")
    """
    return FontProperties(family=_FONT_FAMILY, size=size)
