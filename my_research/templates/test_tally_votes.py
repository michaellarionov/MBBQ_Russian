"""Tests for tally_votes.py (run: python -m pytest)."""
from collections import Counter

import pytest

from tally_votes import NO, SKIPPED, UNSURE, YES, fleiss_kappa, keep


def test_keep_needs_more_than_half_of_all_voters():
    assert keep([YES, YES, NO])
    assert not keep([YES, UNSURE, NO])
    assert keep([YES, YES, YES, NO, NO])
    assert not keep([YES, YES, NO, NO])           # a tie rejects
    assert not keep([YES, SKIPPED, UNSURE])       # skipped and unsure are not-Да


def test_fleiss_kappa_wikipedia_example():
    # Fleiss (1971) as on Wikipedia: 10 items, 14 raters, 5 categories -> 0.210
    rows = [[0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0],
            [2, 2, 8, 1, 1], [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2],
            [6, 5, 2, 1, 0], [0, 2, 2, 3, 7]]
    table = [Counter(dict(enumerate(r))) for r in rows]
    assert fleiss_kappa(table) == pytest.approx(0.210, abs=0.0005)


def test_fleiss_kappa_perfect_agreement():
    assert fleiss_kappa([Counter({YES: 3}), Counter({NO: 3})]) == pytest.approx(1.0)
