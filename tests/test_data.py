import hashlib
import zipfile

import pytest


def hotpot():
    return {
        "id": "abc",
        "question": "Where?",
        "answer": "secret",
        "type": "bridge",
        "context": {
            "title": ["A", "B", "B", "B"],
            "sentences": [["a"], [" b", " x"], [" b", " x"], ["d"]],
        },
        "supporting_facts": {"title": ["A", "B"], "sent_id": [0, 0]},
    }


def musique():
    return {
        "id": "2hop__1_2",
        "question": "Where?",
        "answer": "secret",
        "answer_aliases": ["hidden"],
        "answerable": True,
        "question_decomposition": [{"paragraph_support_idx": 4}, {"paragraph_support_idx": 7}],
        "paragraphs": [
            {"idx": i, "title": "B", "paragraph_text": text, "is_supporting": i != 10}
            for i, text in [(4, "same"), (7, "same"), (10, "different")]
        ],
    }


def test_hotpot_title_support_covers_all_matching_bodies():
    from benchmarks.data import adapt

    case, labels = adapt("hotpot", hotpot())
    assert len(case["documents"]) == 3
    assert {d["text"] for d in case["documents"]} == {"a", " b x", "d"}
    assert all(set(d) == {"id", "title", "text"} for d in case["documents"])
    assert set(case) == {"qid", "question", "documents", "source_map"}
    assert labels["support_ids"] == sorted(d["id"] for d in case["documents"])
    assert labels["source_support_count"] == 4 and labels["answers"] == ["secret"]


def test_musique_index_provenance_survives_exact_duplicate():
    from benchmarks.data import adapt

    case, labels = adapt("musique", musique())
    assert len(case["documents"]) == 2
    assert case["source_map"][0][1] == case["source_map"][1][1]
    assert [x[0] for x in case["source_map"]] == [4, 7, 10]
    assert labels["source_support_count"] == 2 and len(labels["support_ids"]) == 1
    assert labels["answers"] == ["secret", "hidden"]
    assert "secret" not in str(case) and "hidden" not in str(case)


@pytest.mark.parametrize("mode", ["unknown", "empty", "mismatch", "duplicate_idx"])
def test_broken_support_cannot_pass(mode):
    from benchmarks.data import adapt

    rec = musique()
    if mode == "unknown":
        rec["question_decomposition"][0]["paragraph_support_idx"] = 100
    elif mode == "empty":
        rec["question_decomposition"] = []
        for p in rec["paragraphs"]:
            p["is_supporting"] = False
    elif mode == "mismatch":
        rec["paragraphs"][0]["is_supporting"] = False
    else:
        rec["paragraphs"][1]["idx"] = 4
    with pytest.raises(ValueError):
        adapt("musique", rec)


def test_missing_hotpot_support_rejected():
    from benchmarks.data import adapt

    rec = hotpot()
    rec["supporting_facts"]["title"][0] = "absent"
    with pytest.raises(ValueError):
        adapt("hotpot", rec)


def test_membership_is_order_independent_and_disjoint():
    from benchmarks.data import membership

    ids = [str(i) for i in range(900)]
    expected = sorted(
        ids, key=lambda i: (hashlib.sha256(("graphmind-v1-tune:" + i).encode()).digest(), i)
    )[:500]
    assert (
        membership(ids, ["x", "y"]) == membership(ids[::-1], ["y", "x"]) == (expected, ["x", "y"])
    )
    with pytest.raises(ValueError):
        membership(ids, ["899"])
    with pytest.raises(ValueError):
        membership(ids + ["3"], ["x"])


def test_verified_member_checks_digest_and_rejects_path_traversal(tmp_path):
    from benchmarks.sources import extract_member

    path = tmp_path / "a.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("data/good", b"ok")
        z.writestr("../escape", b"ok")
    spec = {"archive_member": "data/good", "bytes": 2, "sha256": hashlib.sha256(b"ok").hexdigest()}
    extract_member(path, spec, tmp_path / "good")
    assert (tmp_path / "good").read_bytes() == b"ok"
    for changed in [{"archive_member": "../escape"}, {"sha256": "0" * 64}]:
        with pytest.raises(ValueError):
            extract_member(path, spec | changed, tmp_path / "bad")
        assert not (tmp_path / "bad").exists()


def test_musique_composition_shapes_share_hop_count_stratum():
    from benchmarks.data import adapt

    rec = musique()
    for shape in ("3hop1", "3hop2"):
        rec["id"] = shape + "__1_2_3"
        assert adapt("musique", rec)[1]["stratum"] == "3hop"
