import json
import math
from pathlib import Path

import pytest

from graphmind import Document, bm25, build_graph, expand, retrieve

CASE = json.loads((Path(__file__).parent / "fixtures/bridge.json").read_text())


def test_two_hop_bridge_and_disconnected_distractor():
    out = retrieve(CASE["documents"], CASE["question"], depth=2, mix=1, seeds=1)
    assert out == retrieve(CASE["documents"][::-1], CASE["question"], depth=2, mix=1, seeds=1)
    assert out["hops"] == [
        {"a": 1, "b": 0, "c": 0, "d": 0},
        {"a": 0, "b": 1, "c": 0, "d": 0},
        {"a": 0.5, "b": 0, "c": 0.5, "d": 0},
    ]
    scores = {r["id"]: r["score"] for r in out["ranking"]}
    assert scores == pytest.approx({"a": 9 / 14, "b": 2 / 7, "c": 1 / 14, "d": 0})


def test_distributions_per_seed_and_mix_on_cycle_with_dangling_node():
    # Undirected triangle A-B-C and isolated D. Seed mass A:D = 3:2.
    docs = [
        Document("a", "Alpha", "Beta Gamma"),
        Document("b", "Beta", "Gamma"),
        Document("c", "Gamma", "x"),
        Document("d", "Delta", "x"),
    ]
    out = expand(build_graph(docs), {"a": 3, "d": 2, "c": 1}, depth=3, mix=0.4, seeds=2)
    powers = [(1, 0, 0), (0, 0.5, 0.5), (0.5, 0.25, 0.25), (0.25, 0.375, 0.375)]
    expected_a = {}
    for i, name in enumerate("abc"):
        expected_a[name] = 0.6 * sum(row[i] * w for row, w in zip(powers, [8, 4, 2, 1])) / 15
    assert out["seed_weights"] == pytest.approx({"a": 0.6, "d": 0.4})
    for observed, p in zip(out["hops"], powers, strict=True):
        assert observed == pytest.approx(dict(zip("abcd", [*(0.6 * x for x in p), 0.4])))
    assert out["per_seed"]["a"] == pytest.approx({**expected_a, "d": 0})
    assert out["per_seed"]["d"] == pytest.approx({"a": 0, "b": 0, "c": 0, "d": 0.4})
    for r in out["ranking"]:
        assert r["graph_score"] == pytest.approx(sum(s[r["id"]] for s in out["per_seed"].values()))
        assert r["score"] == pytest.approx(0.6 * r["base_score"] + 0.4 * r["graph_score"])


def test_bm25_hand_calculation_and_query_deduplication():
    docs = [Document("b", "Tea", "tea and leaf"), Document("a", "Cup", "leaf")]
    # After stopword removal lengths = 3,2; tf(tea)=2, df=1.
    expected = math.log(2) * 2 * 2.2 / (2 + 1.2 * (0.25 + 0.75 * 3 / 2.5))
    assert bm25(docs, "Tea tea THE") == pytest.approx({"a": 0, "b": expected})
    assert bm25(docs, "xylophone") == {"a": 0, "b": 0}


def test_zero_match_bypass_edgeless_and_extreme_scores():
    graph = build_graph(CASE["documents"])
    zero = expand(graph, {})
    assert [r["id"] for r in zero["ranking"]] == list("abcd")
    assert all(r["score"] == 0 for r in zero["ranking"])
    assert zero["seed_weights"] == {}
    for args in [{"depth": 0}, {"mix": 0}]:
        out = expand(graph, {"a": 1e308, "b": 1e308}, seeds=1, **args)
        assert [r["score"] for r in out["ranking"]] == [0.5, 0.5, 0, 0]
    edgeless = build_graph([Document("x", "X", "o"), Document("y", "Y", "o")])
    assert [r["score"] for r in expand(edgeless, {"x": 2, "y": 1}, seeds=1)["ranking"]] == [
        2 / 3,
        1 / 3,
    ]


@pytest.mark.parametrize(
    "base,kwargs",
    [
        ({"z": 2}, {}),
        ({"a": -1}, {}),
        ({"a": float("nan")}, {}),
        ({"a": float("inf")}, {}),
        ({"a": "1"}, {}),
        ({"a": True}, {}),
        ([], {}),
        ({}, {"depth": 4}),
        ({}, {"depth": -1}),
        ({}, {"depth": True}),
        ({}, {"mix": float("nan")}),
        ({}, {"mix": 1.1}),
        ({}, {"seeds": 0}),
        ({}, {"seeds": 4}),
        ({}, {"seeds": 1.5}),
    ],
)
def test_expansion_validation(base, kwargs):
    with pytest.raises(ValueError):
        expand(build_graph(CASE["documents"]), base, **kwargs)


@pytest.mark.parametrize("question", [None, " \t", "a" * 2001])
def test_query_validation(question):
    with pytest.raises(ValueError):
        bm25(CASE["documents"], question)


def test_casefold_uses_whole_unicode_word_tokens():
    # U+0130 expands to i + combining dot. It remains part of the original word.
    docs = [Document("tr", "İstanbul", "x"), Document("other", "bul", "x")]
    assert bm25(docs, "stanbul") == {"other": 0, "tr": 0}
    assert bm25(docs, "İSTANBUL")["tr"] > 0


def test_equal_seed_scores_break_ties_by_id():
    out = expand(build_graph(CASE["documents"]), {"c": 1, "b": 1, "a": 1}, seeds=2)
    assert out["seed_weights"] == {"a": 0.5, "b": 0.5}


def test_empty_graph_returns_empty_distributions_and_zero_effective_mix():
    graph = {"paragraphs": [], "entities": [], "owns": [], "mentions": [], "projection": []}
    out = expand(graph, {}, depth=1)
    assert out["ranking"] == [] and out["hops"] == [{}, {}]
    assert out["effective_mix"] == 0


def test_components_stay_precise_when_final_walk_weight_is_zero():
    out = expand(build_graph(CASE["documents"]), {"a": 1, "b": 1}, seeds=1, depth=0)
    assert out["effective_mix"] == 0 and out["mix"] == 0.35
    assert out["per_seed"]["a"] == out["hops"][0] == {"a": 1, "b": 0, "c": 0, "d": 0}
    assert [r["score"] for r in out["ranking"]] == [0.5, 0.5, 0, 0]
