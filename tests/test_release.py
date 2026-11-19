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


@pytest.mark.parametrize("field", ["rankings", "input_ids"])
def test_json_booleans_cannot_replace_integer_indices(tmp_path, field):
    from benchmarks.run_retrieval import write_shards
    from benchmarks.verify import fixed_shards

    expected = {"qid": "q", field: [0, 1, 2]}
    changed = {"qid": "q", field: [False, True, 2]}
    write_shards(tmp_path, [changed])
    with pytest.raises(ValueError, match="reconstructed records"):
        fixed_shards(tmp_path, [expected])


@pytest.fixture
def reproduced_sources(tmp_path, monkeypatch):
    """Use real matching source metadata in a minimal, freshly hashed reproduction."""
    import json
    import shutil
    from pathlib import Path

    from benchmarks.release import read
    from benchmarks.sources import digest
    from benchmarks.verify import checked_sources

    original = Path(__file__).resolve().parents[1] / "results"
    # Preserve an actual matched prompt/prediction shard while changing only metadata.
    for table in ("inputs", "predictions"):
        shard = Path("answers/hotpot") / table / "0000.jsonl.gz"
        (tmp_path / shard).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original / shard, tmp_path / shard)
    artifacts = {
        name: read(original / name)
        for name in (
            "retrieval/provenance.json",
            "retrieval/elapsed.json",
            "answers/binding.json",
            "answers/elapsed.json",
            "answers/weight-audit.json",
        )
    }
    sidecar = read(original / "retrieval-provenance.json")
    artifacts["retrieval/provenance.json"]["source_binding"] = {
        "commit": sidecar["code_commit"],
        "source_sha256": sidecar["source_sha256"],
    }
    artifacts["answers/weight-audit.json"]["assembly"] = "before-generation"
    # Git traversal is orthogonal to these mutated fields. Leave inventory validation enabled.
    monkeypatch.setattr("benchmarks.verify.git_binding", lambda *_: None)

    def write(name, value):
        file = tmp_path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())

    def check():
        for name, value in artifacts.items():
            write(name, value)
        write(
            "retrieval/manifest.json",
            {"schema": 1, "files": inventory(tmp_path / "retrieval", ("manifest.json",))},
        )
        artifacts["answers/binding.json"]["retrieval_manifest_sha256"] = digest(
            tmp_path / "retrieval/manifest.json"
        )
        write("answers/binding.json", artifacts["answers/binding.json"])
        write("release.json", {"schema": 1, "files": inventory(tmp_path, ("release.json",))})
        check_inventory(tmp_path, read(tmp_path / "release.json")["files"], ("release.json",))
        checked_sources(tmp_path, recorded=False)

    check()
    return artifacts, check


@pytest.mark.parametrize(
    "key,value", [("min_new_tokens", 48), ("forced_bos_token_id", 42), ("use_cache", 1)]
)
def test_reproduction_rejects_altered_decoding_policy(reproduced_sources, key, value):
    artifacts, check = reproduced_sources
    artifacts["answers/binding.json"]["generation_config"][key] = value
    with pytest.raises(ValueError, match="generation config"):
        check()


@pytest.mark.parametrize("key", ["aliases", "tensor_sha256"])
@pytest.mark.parametrize("operation", ["omit", "change"])
def test_reproduction_requires_exact_verified_weight_identities(reproduced_sources, key, operation):
    artifacts, check = reproduced_sources
    audit = artifacts["answers/weight-audit.json"]
    if operation == "omit":
        del audit[key]
    else:
        audit[key][next(iter(audit[key]))] = "forged"
    with pytest.raises(ValueError, match="weight audit"):
        check()


@pytest.mark.parametrize(
    "name,key",
    [("answers", "model_load_inference_and_scoring_seconds"), ("retrieval", "measurement_seconds")],
)
@pytest.mark.parametrize(
    "corruption",
    ["invalid_json", "missing", "extra", "zero", "nan", "infinity", "bool", "negative"],
)
def test_reproduction_validates_elapsed_evidence(reproduced_sources, name, key, corruption):
    artifacts, check = reproduced_sources
    damaged = {
        "invalid_json": b"invalid",
        "missing": {},
        "extra": {key: 4, "unknown": 5},
        "zero": {key: 0},
        "nan": {key: float("nan")},
        "infinity": {key: float("inf")},
        "bool": {key: True},
        "negative": {key: -1},
    }
    artifacts[name + "/elapsed.json"] = damaged[corruption]
    with pytest.raises(ValueError):
        check()


def test_bm25_replay_accepts_linux_rounding_and_keeps_historical_ranks(monkeypatch, tmp_path):
    import math
    from pathlib import Path

    from benchmarks.release import read, table
    from benchmarks.run_retrieval import evaluate_case, write_shards
    from benchmarks.verify import fixed_shards, verified_bm25
    from graphmind import bm25, build_graph

    root = Path(__file__).resolve().parents[1] / "results/retrieval"
    sample = root / "musique/train"
    qid = "4hop1__771620_317689_1286_19331"
    case = next(r for r in table(sample / "cases") if r["qid"] == qid)
    run = next(r for r in table(sample / "runs") if r["qid"] == qid)
    docs = case["documents"]
    score = run["scores"]["bm25"]
    assert (score[14], score[16]) == (5.847271371822889, 8.203580623851579)
    linux = bm25(docs, case["question"])
    linux[docs[14]["id"]] = 5.84727137182289
    linux[docs[16]["id"]] = 8.20358062385158
    monkeypatch.setattr("benchmarks.verify.bm25", lambda *_: linux)
    checked = verified_bm25(case, score)
    assert list(checked.values()) == score
    dense = {d["id"]: v for d, v in zip(docs, run["scores"]["dense"], strict=True)}
    config = read(root / "frozen.json")["musique"]
    rebuilt, _ = evaluate_case(case, (build_graph(docs), checked, dense), config)
    write_shards(tmp_path, [run])
    fixed_shards(tmp_path, [rebuilt])

    for invalid in (math.nextafter(linux[docs[14]["id"]], math.inf, steps=5), 5.84, True, 5):
        with pytest.raises(ValueError, match="BM25"):
            verified_bm25(case, [*score[:14], invalid, *score[15:]])
    for invalid in (0, False, -0.0, math.ulp(0.0), math.nan, math.inf):
        forged = score.copy()
        forged[score.index(0.0)] = invalid
        with pytest.raises(ValueError, match="BM25"):
            verified_bm25(case, forged)

    run["rankings"]["bm25"][:2] = reversed(run["rankings"]["bm25"][:2])
    write_shards(tmp_path / "forged", [run])
    with pytest.raises(ValueError, match="reconstructed records"):
        fixed_shards(tmp_path / "forged", [rebuilt])


def test_bm25_replay_resolves_near_tie_from_validated_original_scores(monkeypatch):
    import math

    from benchmarks.run_retrieval import ranked
    from benchmarks.verify import verified_bm25

    case = {"qid": "q", "question": "alias", "documents": [{"id": "a"}, {"id": "b"}]}
    high = math.nextafter(1.0, math.inf)
    local = {"a": high, "b": 1.0}
    monkeypatch.setattr("benchmarks.verify.bm25", lambda *_: local)
    assert ranked(local) == ["a", "b"]
    assert ranked(verified_bm25(case, [1.0, high])) == ["b", "a"]


def test_retrieval_inference_checks_tail_with_relative_error_and_exact_counts_and_holm():
    import math
    from pathlib import Path

    from benchmarks.release import read
    from benchmarks.verify import check_retrieval_estimates

    path = Path(__file__).resolve().parents[1] / "results/retrieval/primary.json"
    observed = read(path)
    local = copy.deepcopy(observed)
    for row in local.values():
        row.pop("p_holm")
        row["p"] = math.nextafter(row["p"], math.inf)
    check_retrieval_estimates(local, observed)
    for field, value in (
        ("p", 0.0),
        ("p", observed["hotpot"]["p"] * 1.001),
        ("p", True),
        ("p", 0),
        ("p_holm", math.nextafter(observed["hotpot"]["p_holm"], math.inf)),
        ("wins", float(observed["hotpot"]["wins"])),
        ("ties", observed["hotpot"]["ties"] - 1),
    ):
        forged = copy.deepcopy(observed)
        forged["hotpot"][field] = value
        with pytest.raises(ValueError):
            check_retrieval_estimates(local, forged)
