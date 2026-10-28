import pytest

from labeler import metrics

LABELS = ("a", "b", "c")


def test_accuracy():
    assert metrics.accuracy(["a", "b", "c", "a"], ["a", "b", "a", "a"]) == 0.75
    assert metrics.accuracy([], []) is None


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        metrics.accuracy(["a"], ["a", "b"])


def test_kappa_hand_computed():
    # p_o = 0.75; human counts a=2,b=1,c=1; ai counts a=3,b=1
    # p_e = (2*3 + 1*1 + 1*0) / 16 = 7/16; kappa = (12/16 - 7/16) / (9/16) = 5/9
    assert metrics.cohens_kappa(["a", "b", "c", "a"], ["a", "b", "a", "a"]) == pytest.approx(5 / 9)


def test_kappa_edge_cases():
    assert metrics.cohens_kappa([], []) is None
    assert metrics.cohens_kappa(["a", "a"], ["a", "a"]) == 1.0
    assert metrics.cohens_kappa(["a", "b"], ["a", "b"]) == 1.0
    # Systematic swap: worse than chance.
    assert metrics.cohens_kappa(["a", "b"], ["b", "a"]) == pytest.approx(-1.0)


def test_confusion_matrix_rows_are_human():
    m = metrics.confusion_matrix(["a", "a", "b"], ["a", "b", "b"], LABELS)
    assert m["a"] == {"a": 1, "b": 1, "c": 0}
    assert m["b"] == {"a": 0, "b": 1, "c": 0}
    assert sum(sum(row.values()) for row in m.values()) == 3


def test_per_label():
    stats = {s.label: s for s in metrics.per_label(["a", "a", "b"], ["a", "b", "b"], LABELS)}
    assert stats["a"].precision == 1.0 and stats["a"].recall == 0.5
    assert stats["a"].f1 == pytest.approx(2 / 3)
    assert stats["b"].precision == 0.5 and stats["b"].recall == 1.0
    # Never predicted and never present: undefined, not zero.
    assert stats["c"].precision is None and stats["c"].recall is None and stats["c"].f1 is None
    assert stats["c"].support == 0


def test_per_label_zero_f1_when_defined_but_wrong():
    s = {s.label: s for s in metrics.per_label(["a", "b"], ["b", "a"], LABELS)}
    assert s["a"].precision == 0.0 and s["a"].recall == 0.0 and s["a"].f1 == 0.0


def test_confidence_bands_boundaries():
    human = ["a", "a", "a", "a", "a"]
    ai = ["a", "b", "a", "a", "b"]
    conf = [0.1, 0.6, 0.8, 0.9, 1.0]
    bands = {b.band: b for b in metrics.by_confidence(human, ai, conf)}
    assert bands["<0.6"].n == 1 and bands["<0.6"].accuracy == 1.0
    assert bands["0.6-0.8"].n == 1 and bands["0.6-0.8"].accuracy == 0.0
    assert bands["0.8-0.9"].n == 1
    assert bands[">=0.9"].n == 2 and bands[">=0.9"].accuracy == 0.5
    assert sum(b.n for b in bands.values()) == len(human)


def test_report_bundles_everything():
    r = metrics.report(["a", "b"], ["a", "b"], [0.9, 0.95], LABELS)
    assert r.n == 2 and r.accuracy == 1.0 and r.kappa == 1.0
    assert len(r.per_label) == 3 and len(r.bands) == 4
