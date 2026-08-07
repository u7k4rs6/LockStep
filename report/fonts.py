"""The three faces from docs/kickoff/04-frontend-spec.md, inlined as base64."""

from __future__ import annotations

import base64
from pathlib import Path

FONT_DIR = Path(__file__).parent / "fonts"

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

CODEPOINTS = "U+0020-007E,U+00A0,U+2013,U+2018,U+2019,U+201C,U+201D,U+2026"


def face(stem: str) -> str:
    """One `@font-face` rule with the binary inlined."""
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
