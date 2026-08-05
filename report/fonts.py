"""The three faces from docs/04-frontend-spec.md, inlined as base64.

The spec's constraint is "no external font requests at runtime", so the report
works offline from `file://` and renders identically on a machine that has never
heard of Google Fonts. That rules out a stylesheet link, which is the usual way
these three arrive, so the faces are committed as subsets and embedded.

Provenance, since embedded binaries deserve it: the sources are the upstream TTFs
in `google/fonts`, at the paths named in `SOURCES`. Archivo and Instrument Sans
ship as variable fonts; each was instanced at the axis values the spec asks for
(Archivo at the expanded end of `wdth` and heavy `wght`, per "wide engineering
signage") and then subset. All three are under the SIL Open Font License, whose
text is committed alongside the binaries in `report/fonts/`; the OFL permits
redistribution in this form and requires the license to travel with it.

The subset is Latin printable plus the handful of punctuation marks the report
actually emits. Every face is under 11 KB and the four together are about 32 KB,
against roughly 1.2 MB for the unsubset originals.

`fonttools` is deliberately **not** a dependency of this project. Nothing here
imports it: the binaries are committed and this module only base64-encodes them.
Regenerating a face is a rare, deliberate act, so it uses a throwaway environment
rather than widening the dependency set for everyone who runs the tests:

    uvx --from "fonttools[woff]==4.56.0" python -m fontTools.varLib.instancer \\
        Archivo[wdth,wght].ttf wght=700 wdth=125 -o archivo-700.ttf
    uvx --from "fonttools[woff]==4.56.0" python -m fontTools.subset \\
        archivo-700.ttf --unicodes=$CODEPOINTS --layout-features=kern,liga \\
        --no-hinting --desubroutinize --flavor=woff2 --output-file=archivo-700.woff2
"""

from __future__ import annotations

import base64
from pathlib import Path

FONT_DIR = Path(__file__).parent / "fonts"

# Where each binary came from, and what was done to it. Regenerating any of these
# means fetching the upstream path, instancing at the axis values below, and
# subsetting to CODEPOINTS with fonttools.
SOURCES = {
    "archivo-700": {
        "upstream": "ofl/archivo/Archivo[wdth,wght].ttf",
        "instance": "wght=700, wdth=125",
        "family": "Archivo",
        "weight": 700,
    },
    "instrument-400": {
        "upstream": "ofl/instrumentsans/InstrumentSans[wdth,wght].ttf",
        "instance": "wght=400, wdth=100",
        "family": "Instrument Sans",
        "weight": 400,
    },
    "instrument-600": {
        "upstream": "ofl/instrumentsans/InstrumentSans[wdth,wght].ttf",
        "instance": "wght=600, wdth=100",
        "family": "Instrument Sans",
        "weight": 600,
    },
    "plexmono-400": {
        "upstream": "ofl/ibmplexmono/IBMPlexMono-Regular.ttf",
        "instance": "static, no axes",
        "family": "IBM Plex Mono",
        "weight": 400,
    },
}

# Latin printable, plus the dashes, quotes, and ellipsis the report emits. A
# glyph outside this set falls back to the next family in the CSS stack, which is
# visible but not broken.
CODEPOINTS = "U+0020-007E,U+00A0,U+2013,U+2018,U+2019,U+201C,U+201D,U+2026"


def face(stem: str) -> str:
    """One `@font-face` rule with the binary inlined.

    `font-display: block` rather than `swap`: the report is read, not scrolled
    past, and a reflow partway through a numeric table is worse than a few
    milliseconds of nothing. The data is already in the document, so the block
    period is however long the browser takes to decode 10 KB.
    """
    spec = SOURCES[stem]
    payload = base64.b64encode((FONT_DIR / f"{stem}.woff2").read_bytes()).decode()
    return (
        "@font-face{"
        f"font-family:'{spec['family']}';"
        "font-style:normal;"
        f"font-weight:{spec['weight']};"
        "font-display:block;"
        f"src:url(data:font/woff2;base64,{payload}) format('woff2');"
        "}"
    )


def css() -> str:
    """All four faces, ready to drop at the top of the report's stylesheet."""
    return "".join(face(stem) for stem in SOURCES)


def embedded_bytes() -> int:
    """Total decoded size, reported in the artifact so the weight is visible."""
    return sum((FONT_DIR / f"{stem}.woff2").stat().st_size for stem in SOURCES)
