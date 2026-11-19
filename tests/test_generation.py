from copy import deepcopy

import pytest

from benchmarks.generation import build_prompt, check_output, check_predictions, score, tokenizer


def inputs():
    return [
        {
            "id": str(i),
            "title": "Café 😀 " + str(i),
            "text": "東京 e\u0301 🙂 real evidence and context. " * 90,
        }
        for i in range(4)
    ]


def test_prompt_gold_invariance_and_unicode_offsets():
    docs = inputs()
    altered = deepcopy(docs)
    for doc in altered:
        doc.update(answer="FORBIDDEN", is_supporting=True)
    tok = tokenizer()
    prompt = build_prompt(tok, "Which city?", docs)
    assert prompt == build_prompt(tok, "Which city?", altered)
    assert "FORBIDDEN" not in prompt["text"]
    assert prompt["input_ids"] == tok.encode(prompt["text"]).ids
    for doc, passage in zip(docs, prompt["passages"], strict=True):
        start, end = passage["prompt_span"]
        assert prompt["text"][start:end] == doc["text"][: passage["body_end"]]
        assert 0 < passage["body_end"] < len(doc["text"])
        assert 0 < passage["tokens"] <= 96
    assert sum(p["tokens"] for p in prompt["passages"]) <= 384
    assert len(prompt["input_ids"]) <= 512


def test_context_absence_and_unequal_passages():
    docs = inputs()
    docs[0]["text"] = "short"
    out = build_prompt(tokenizer(), "Who is absent?", docs)
    assert out["passages"][0]["tokens"] < 20
    assert all(p["tokens"] <= 96 for p in out["passages"][1:])
    assert "unknown gold string" not in out["text"]


def test_missing_or_repeated_predictions():
    keys = [("q", m) for m in ("dense", "hybrid", "graph-2")]
    with pytest.raises(ValueError):
        check_predictions(keys[:-1], keys)
    with pytest.raises(ValueError):
        check_predictions(keys + keys[:1], keys)
    check_predictions(keys[::-1], keys)


def test_stopping_is_strict():
    tok = tokenizer()
    assert check_output(tok, [0, 1]) == {"answer": "", "stop": "eos"}
    assert check_output(tok, [0] + [10] * 48)["stop"] == "length"
    for ids in ([0, 10], [1], [0, 1, 10], [0, 10**8, 1], [0] + [10] * 49):
        with pytest.raises(ValueError):
            check_output(tok, ids)


def test_reference_scoring_special_cases_and_aliases():
    assert score("hotpot", "yes", ["yes indeed"]) == {"em": 0, "f1": 0}
    assert score("musique", "the!", ["a"]) == {"em": 1, "f1": 1}
    assert score("hotpot", "the!", ["a"]) == {"em": 1, "f1": 0}
    assert score("musique", "UK", ["United Kingdom", "the UK"]) == {"em": 1, "f1": 1}


def test_float_paired_deltas_survive_cluster_bootstrap():
    from benchmarks.generation import paired

    single = paired([0.25, 0.5], [[0, 1]])
    assert single["delta"] == 0.375 and single["cluster_ci95"] is None
    grouped = paired([0.25, 0.5], [[0], [1]])
    assert grouped["query_ci95"] == [0.25, 0.5]
    assert grouped["cluster_ci95"] == [0.25, 0.5]
    assert 0 < grouped["p"] <= 1


def test_no_answer_string_or_support_needed_to_generate_context():
    docs = [{"id": "one", "title": "Berlin", "text": "The garden opened in 1902."}]
    prompt = build_prompt(tokenizer(), "Who designed the garden?", docs)
    assert prompt["passages"][0]["body_end"] == len(docs[0]["text"])
    for reference in ("Max Planck", "Ada Lovelace"):
        assert reference not in prompt["text"]
        assert score("hotpot", "1902", [reference]) == {"em": 0, "f1": 0}
