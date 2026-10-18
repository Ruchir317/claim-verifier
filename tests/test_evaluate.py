import evaluate_accuracy as ev


def test_label_mapping_collapses_the_six_point_scale():
    rows = [{"statement": s, "verdict": v} for s, v in
            [("a", "true"), ("b", "mostly-true"), ("c", "half-true"), ("d", "barely-true"), ("e", "false"), ("f", "pants-fire")]]
    truth = ev.build_truth_lookup(rows)
    assert [truth[s] for s in "abcdef"] == [True, True, True, False, False, False]
    strict = ev.build_truth_lookup(rows, ["true", "mostly-true"])
    assert strict["c"] is False


def test_compare_skips_bad_verdicts_and_reports_unmatched():
    truth = {"a": True, "b": False}
    outputs = [
        {"fact": "a", "verdict": "True"},
        {"fact": "b", "verdict": "True"},
        {"fact": "c", "verdict": "False"},
        {"fact": "a", "verdict": None},
        {"fact": "a", "verdict": "Maybe"},
    ]
    rows, unmatched = ev.compare(outputs, truth)
    assert len(rows) == 2 and unmatched == ["c"]
    assert ev.accuracy(rows) == 0.5


def test_accuracy_of_nothing_is_zero():
    assert ev.accuracy([]) == 0.0
