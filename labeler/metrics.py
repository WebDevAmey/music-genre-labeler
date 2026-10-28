"""Agreement metrics between AI suggestions and human (ground-truth) labels. Pure functions."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

# (name, low inclusive, high exclusive). The last band includes 1.0.
CONFIDENCE_BANDS: tuple[tuple[str, float, float], ...] = (
    ("<0.6", 0.0, 0.6),
    ("0.6-0.8", 0.6, 0.8),
    ("0.8-0.9", 0.8, 0.9),
    (">=0.9", 0.9, 1.0 + 1e-9),
)


@dataclass(frozen=True)
class LabelStats:
    label: str
    support: int  # human-labeled count
    precision: float | None  # None when the AI never predicted this label
    recall: float | None  # None when no human label has this value
    f1: float | None


@dataclass(frozen=True)
class BandStats:
    band: str
    n: int
    accuracy: float | None


@dataclass(frozen=True)
class Report:
    n: int
    accuracy: float | None
    kappa: float | None
    per_label: list[LabelStats]
    confusion: dict[str, dict[str, int]]  # confusion[human][ai] = count
    bands: list[BandStats]


def accuracy(human: Sequence[str], ai: Sequence[str]) -> float | None:
    _check_pairs(human, ai)
    if not human:
        return None
    return sum(h == a for h, a in zip(human, ai)) / len(human)


def cohens_kappa(human: Sequence[str], ai: Sequence[str]) -> float | None:
    """Agreement corrected for chance: (p_o - p_e) / (1 - p_e)."""
    _check_pairs(human, ai)
    n = len(human)
    if n == 0:
        return None
    p_o = sum(h == a for h, a in zip(human, ai)) / n
    h_counts, a_counts = Counter(human), Counter(ai)
    p_e = sum(h_counts[k] * a_counts[k] for k in h_counts) / (n * n)
    if p_e == 1.0:
        # Both raters used the same single label for every item (so p_o == 1).
        # The formula is 0/0; treat it as perfect agreement.
        return 1.0
    return (p_o - p_e) / (1 - p_e)


def confusion_matrix(
    human: Sequence[str], ai: Sequence[str], labels: Sequence[str]
) -> dict[str, dict[str, int]]:
    _check_pairs(human, ai)
    matrix = {h: {a: 0 for a in labels} for h in labels}
    for h, a in zip(human, ai):
        matrix[h][a] += 1
    return matrix


def per_label(human: Sequence[str], ai: Sequence[str], labels: Sequence[str]) -> list[LabelStats]:
    _check_pairs(human, ai)
    out = []
    for label in labels:
        tp = sum(h == label and a == label for h, a in zip(human, ai))
        predicted = sum(a == label for a in ai)
        actual = sum(h == label for h in human)
        p = tp / predicted if predicted else None
        r = tp / actual if actual else None
        if p is None or r is None:
            f1 = None
        else:
            f1 = 2 * p * r / (p + r) if p + r else 0.0
        out.append(LabelStats(label, actual, p, r, f1))
    return out


def by_confidence(
    human: Sequence[str], ai: Sequence[str], confidence: Sequence[float]
) -> list[BandStats]:
    _check_pairs(human, ai)
    if len(confidence) != len(human):
        raise ValueError("confidence must align with labels")
    out = []
    for name, lo, hi in CONFIDENCE_BANDS:
        idx = [i for i, c in enumerate(confidence) if lo <= c < hi]
        acc = sum(human[i] == ai[i] for i in idx) / len(idx) if idx else None
        out.append(BandStats(name, len(idx), acc))
    return out


def report(
    human: Sequence[str],
    ai: Sequence[str],
    confidence: Sequence[float],
    labels: Sequence[str],
) -> Report:
    return Report(
        n=len(human),
        accuracy=accuracy(human, ai),
        kappa=cohens_kappa(human, ai),
        per_label=per_label(human, ai, labels),
        confusion=confusion_matrix(human, ai, labels),
        bands=by_confidence(human, ai, confidence),
    )


def _check_pairs(human: Sequence[str], ai: Sequence[str]) -> None:
    if len(human) != len(ai):
        raise ValueError("human and ai label lists must be the same length")
