"""The thesis table's row count and the counts stated in prose must agree.

`12b` is a row label and not a count, so the number of rows and the highest row
number are different quantities that can drift apart while both look right.

The table used to sit under a `## The thesis` heading. It now lives inside the
`<details>` beneath the thesis panel, so this locates it by its own header row
rather than by a heading that a redesign can remove.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parent.parent / "README.md"

WORDS = {
    12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen", 16: "sixteen",
    17: "seventeen", 18: "eighteen", 19: "nineteen", 20: "twenty",
}

ROW = re.compile(r"^\| (\d+b?) \|", re.M)
HEADER = "| # | Declared | Actually |"


def thesis_section() -> str:
    text = README.read_text()
    start = text.index(HEADER)
    end = text.index("</details>", start)
    return text[start:end]


def labels() -> list[str]:
    return ROW.findall(thesis_section())


def numbered() -> list[int]:
    return [int(label) for label in labels() if label.isdigit()]


def test_every_label_is_unique_and_numbering_has_no_gap():
    all_labels = labels()
    assert len(all_labels) == len(set(all_labels)), f"duplicate label: {all_labels}"
    nums = sorted(numbered())
    assert nums == list(range(1, len(nums) + 1)), (
        f"row numbers are not 1..N with no gap: {nums}. A gap would make every "
        "derived count below wrong in a way that still looks plausible."
    )


def test_the_stated_counts_match_the_table():
    text = README.read_text()
    total = len(numbered())
    word = WORDS[total]

    assert f"{word} times in this repository" in text, (
        f"the thesis paragraph should say '{word} times in this repository'; "
        f"the table has {total} numbered rows"
    )
    assert f"<summary>All {word}," in text, (
        f"the collapsed summary should open '<summary>All {word},'; the table "
        f"has {total} numbered rows"
    )


def test_no_other_count_word_is_claimed_anywhere():
    """A stale 'sixteen' left behind elsewhere is exactly how row 16 happened."""
    text = README.read_text()
    correct = WORDS[len(numbered())]
    for n, word in WORDS.items():
        if word == correct:
            continue
        assert f"{word} times in this repository" not in text, (
            f"the README also claims '{word} times in this repository' while the "
            f"table has {len(numbered())} rows"
        )


def test_the_highest_row_number_is_not_assumed_to_be_the_count():
    """12b is a row and not a count, so the two can drift apart silently."""
    assert len(labels()) > len(numbered()), (
        "no lettered row label is present any more. If 12b was renumbered, the "
        "convention in this module's docstring no longer describes the table "
        "and the counts should be re-derived rather than left to agree by luck."
    )


@pytest.mark.parametrize("n", [16, 17, 18])
def test_the_word_table_covers_the_range_the_readme_can_reach(n):
    assert n in WORDS
