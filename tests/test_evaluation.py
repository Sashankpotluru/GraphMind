import math

import pytest


def test_metrics_small_candidate_list_and_trec_ties():
    from benchmarks.evaluate import crosscheck, metrics

    ranking = ["a", "b", "c"]
    result = metrics(ranking, ["b", "c"])
    assert result["all@2"] == 0 and result["all@4"] == 1
    assert result["recall@2"] == 0.5 and result["recall@6"] == 1
    assert result["mrr"] == 0.5
    assert result["ndcg@4"] == pytest.approx((1 / math.log2(3) + 0.5) / (1 + 1 / math.log2(3)))
    diffs = crosscheck(ranking, ["b", "c"], result)
    assert set(diffs) == set(result) and max(diffs.values()) < 1e-10
    for labels in [[], ["unknown"]]:
        with pytest.raises(ValueError):
            metrics(ranking, labels)


def test_transitive_clusters_ratio_bootstrap_and_sign_flip():
    from benchmarks.statistics import components, holm, summarize

    groups = components([["x"], ["x", "y"], ["y"], ["z"]])
    assert groups == [[0, 1, 2], [3]]
    result = summarize([1, 1, 1, -1], groups)
    assert result["delta"] == 0.5
    assert result["p"] == pytest.approx(0.625)
    assert result["cluster_sizes"] == [3, 1] and result["largest_cluster_share"] == 0.75
    assert result["cluster_ci95"] == [-1, 1]
    assert summarize([0, 0], [[0, 1]])["cluster_ci95"] is None
    assert summarize([0, 0], [[0, 1]])["p"] == 1
    assert holm([0.02, 0.03]) == pytest.approx([0.04, 0.04])
