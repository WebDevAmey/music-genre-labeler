"""Write labels to CSV."""

from __future__ import annotations

import csv
import io
from typing import Iterable

from .store import Item

COLUMNS = ("id", "text", "label", "ai_label", "ai_confidence", "ai_correct", "ai_model", "reviewed_at")


def to_csv(items: Iterable[Item]) -> str:
    """`label` is the human label (the ground truth); empty for unreviewed items."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(COLUMNS)
    for it in items:
        correct = "" if it.human_label is None or it.ai_label is None else it.human_label == it.ai_label
        writer.writerow(
            (it.id, it.text, it.human_label or "", it.ai_label or "",
             "" if it.ai_confidence is None else it.ai_confidence,
             correct, it.ai_model or "", it.reviewed_at or "")
        )
    return buf.getvalue()
