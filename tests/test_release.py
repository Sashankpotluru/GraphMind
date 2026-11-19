import copy

import pytest

from benchmarks.release import bound_rows, check_inventory, inventory, require_equal


def test_extra_artifact_rejected(tmp_path):
    (tmp_path / "observed").write_text("first")
    expected = inventory(tmp_path)
    (tmp_path / "extra").write_text("unexpected")
    with pytest.raises(ValueError):
        check_inventory(tmp_path, expected)
    assert (tmp_path / "observed").read_text() == "first"


def test_changed_labels_cannot_match_independent_source_binding():
    rows = [{"qid": "q", "support_ids": ["a"], "answers": ["yes"]}]
    observed = bound_rows(rows)
    forged = copy.deepcopy(rows)
    forged[0].update(answers=["no"], support_ids=["b"])
    with pytest.raises(ValueError):
        require_equal("source labels", observed, bound_rows(forged))


def test_forged_prompt_or_mention_span_fails():
    for source in ({"text": "Hello!", "input_ids": [0, 8, 1]}, {"span": [2, 5], "text": "abc"}):
        changed = {**source, "text": "forged"}
        with pytest.raises(ValueError):
            require_equal("reconstructed record", source, changed)


def test_forged_prompt_fails_retokenized_full_paragraph_replay(tmp_path):
    from benchmarks.generation import build_prompt, tokenizer
    from benchmarks.run_retrieval import write_shards
    from benchmarks.verify import fixed_shards

    tok = tokenizer()
    source = [{"id": "a", "title": "Portrait", "text": "Her name was René."}]
    actual = build_prompt(tok, "What was her name?", source)
    counterfeit = build_prompt(
        tok, "What was her name?", [{**source[0], "text": "His name was Robin."}]
    )
    write_shards(tmp_path, [counterfeit])
    with pytest.raises(ValueError, match="reconstructed records"):
        fixed_shards(tmp_path, [actual])


def test_graph_span_cannot_be_shifted_with_rehashed_shard(tmp_path):
    from benchmarks.run_retrieval import write_shards
    from benchmarks.verify import fixed_shards
    from graphmind import build_graph

    docs = [
        {"id": "a", "title": "Origine", "text": "À côté de Zürich."},
        {"id": "b", "title": "Zürich", "text": "City."},
    ]
    expected = {"qid": "q", "graph": build_graph(docs)}
    changed = copy.deepcopy(expected)
    edge = changed["graph"]["mentions"][0]
    edge["start"] -= 1
    edge["text"] = docs[0]["text"][edge["start"] : edge["end"]]
    write_shards(tmp_path, [changed])
    with pytest.raises(ValueError, match="reconstructed records"):
        fixed_shards(tmp_path, [expected], size=8)


def test_source_binding_rejects_coherently_changed_canonical_judgments(tmp_path, monkeypatch):
    from benchmarks.release import canonical_rows
    from benchmarks.run_retrieval import write_shards

    original = [{"qid": "q", "answers": ["yes"], "support_ids": ["a"]}]
    monkeypatch.setattr(
        "benchmarks.release.CANONICAL", {"hotpot": {"dev": {"judgments": bound_rows(original)}}}
    )
    altered = [{"qid": "q", "answers": ["no"], "support_ids": ["b"]}]
    write_shards(tmp_path / "hotpot/dev/judgments", altered)
    with pytest.raises(ValueError, match="source binding"):
        canonical_rows(tmp_path, "hotpot", "dev", "judgments")


def test_existing_evidence_survives_release_error(tmp_path):
    from benchmarks.release import build

    destination = tmp_path / "release"
    destination.mkdir()
    old = destination / "release.json"
    old.write_text("kept")
    with pytest.raises(ValueError, match="must be new"):
        build(tmp_path / "observations", destination)
    assert old.read_text() == "kept"
    fresh = tmp_path / "fresh"
    with pytest.raises(FileNotFoundError):
        build(tmp_path / "invalid", fresh)
    assert not fresh.exists()


def test_weight_free_decode_rejects_changed_answer():
    from benchmarks.generation import tokenizer
    from benchmarks.verify import check_prediction

    row = {
        "qid": "q",
        "method": "dense",
        "decoder_ids": [0, 1],
        "answer": "invented",
        "stop": "eos",
    }
    with pytest.raises(ValueError, match="decoded reader output"):
        check_prediction(tokenizer(), row)
