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
