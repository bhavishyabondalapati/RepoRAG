import pytest

from metrics import (
    Span,
    citation_precision,
    citation_recall,
    overlaps,
    parse_judge_score,
    recall_at_k,
    reciprocal_rank,
)

GOLD = [Span("a.py", 10, 20), Span("b.py", 1, 5)]


def test_overlap_requires_same_file_and_intersecting_lines():
    assert overlaps("a.py", 18, 30, GOLD[0])
    assert overlaps("a.py", 20, 20, GOLD[0])
    assert not overlaps("a.py", 21, 30, GOLD[0])
    assert not overlaps("c.py", 10, 20, GOLD[0])


def test_recall_counts_each_gold_span_once():
    retrieved = [("a.py", 1, 12), ("a.py", 15, 25), ("x.py", 1, 9)]
    assert recall_at_k(retrieved, GOLD, k=5) == 0.5
    assert recall_at_k(retrieved + [("b.py", 4, 4)], GOLD, k=4) == 1.0
    assert recall_at_k(retrieved + [("b.py", 4, 4)], GOLD, k=3) == 0.5


def test_recall_undefined_without_gold():
    with pytest.raises(ValueError):
        recall_at_k([], [], k=5)


def test_reciprocal_rank():
    assert reciprocal_rank([("x.py", 1, 2), ("b.py", 2, 3)], GOLD) == 0.5
    assert reciprocal_rank([("x.py", 1, 2)], GOLD) == 0.0
    assert reciprocal_rank([("x.py", 1, 2)] * 10 + [("a.py", 10, 10)], GOLD, k=10) == 0.0


def test_parse_judge_score():
    raw = (
        'Sure: {"claims": [{"claim": "x", "supported": true}, '
        '{"claim": "y", "supported": false}], "supported": 1, "total": 2}'
    )
    assert parse_judge_score(raw) == 0.5
    assert parse_judge_score('{"supported": 3, "total": 3}') == 1.0
    assert parse_judge_score("no json here") is None
    assert parse_judge_score("{broken") is None


def test_citation_precision_is_share_of_cited_chunks_on_gold():
    cited = [("a.py", 12, 15), ("x.py", 1, 9), ("b.py", 3, 4), ("a.py", 40, 50)]
    assert citation_precision(cited, GOLD) == 0.5
    assert citation_precision([("x.py", 1, 9)], GOLD) == 0.0
    assert citation_precision([], GOLD) is None


def test_citation_recall_is_share_of_gold_spans_cited():
    assert citation_recall([("a.py", 12, 15), ("x.py", 1, 9)], GOLD) == 0.5
    assert citation_recall([("a.py", 12, 15), ("b.py", 1, 1)], GOLD) == 1.0
    assert citation_recall([], GOLD) == 0.0
