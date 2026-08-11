"""The thesis table's counts, derived from the table rather than remembered.

Three places in README.md state how many entries the table has. All three were
written by hand. A table that catalogues off-by-one errors and undercounts
itself would be the most embarrassing possible instance of its own subject, so
the numbers are derived here and the prose is asserted against them.

The counting convention is the one already in the file, inferred from all three
statements agreeing under it and no other: **rows whose label is a plain
integer**. `12b` is a row and is deliberately not a count, because it refines
row 12 rather than adding a failure. Change the convention and all three
numbers move together, which is the point of deriving them in one place.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parent.parent / "README.md"

WORDS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
    8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve",
    13: "thirteen", 14: "fourteen", 15: "fifteen", 16: "sixteen",
    17: "seventeen", 18: "eighteen", 19: "nineteen", 20: "twenty",
}

ROW = re.compile(r"^\| (\d+b?) \|", re.M)


def thesis_section() -> str:
    text = README.read_text()
    start = text.index("## The thesis, demonstrated on its author")
    end = text.index("\n## ", start + 10)
    return text[start:end]


def labels() -> list[str]:
    return ROW.findall(thesis_section())


def numbered() -> list[int]:
    return [int(label) for label in labels() if label.isdigit()]


def inside_details() -> list[int]:
    section = thesis_section()
    start = section.index("<details>")
    end = section.index("</details>", start)
    return [int(x) for x in ROW.findall(section[start:end]) if x.isdigit()]


def test_every_label_is_unique_and_numbering_has_no_gap():
    all_labels = labels()
    assert len(all_labels) == len(set(all_labels)), f"duplicate label: {all_labels}"
    nums = sorted(numbered())
    assert nums == list(range(1, len(nums) + 1)), (
        f"row numbers are not 1..N with no gap: {nums}. A gap would make every "
        "derived count below wrong in a way that still looks plausible."
    )


def test_the_three_stated_counts_match_the_table():
    text = README.read_text()
    total, in_details = len(numbered()), len(inside_details())

    total_word = WORDS[total]
    assert text.count(f"{total_word} times this project failed its own test") == 1, (
        f"the navigation table should say '{total_word} times this project "
        f"failed its own test'; the table has {total} numbered rows"
    )
    assert f"It happened\n{total_word} times in this repository." in text or (
        f"{total_word} times in this repository." in text), (
        f"the thesis paragraph should say '{total_word} times in this repository'"
    )

    details_word = WORDS[in_details].capitalize()
    assert f"<summary>{details_word} more," in text, (
        f"the collapsed summary should open '{details_word} more,'; the table "
        f"has {in_details} numbered rows inside <details>"
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
