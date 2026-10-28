"""SQLite storage. One row per item; every write is committed immediately."""

from __future__ import annotations

import csv
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .task import LABELS

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id            TEXT PRIMARY KEY,
    text          TEXT NOT NULL,
    ai_label      TEXT,
    ai_confidence REAL,
    ai_rationale  TEXT,
    ai_model      TEXT,
    human_label   TEXT,
    reviewed_at   TEXT
)
"""


@dataclass(frozen=True)
class Item:
    id: str
    text: str
    ai_label: str | None
    ai_confidence: float | None
    ai_rationale: str | None
    ai_model: str | None
    human_label: str | None
    reviewed_at: str | None

    @property
    def status(self) -> str:
        if self.human_label is not None:
            return "reviewed"
        return "suggested" if self.ai_label is not None else "pending"


class Store:
    def __init__(self, path: str | Path):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # --- writes -------------------------------------------------------------

    def import_csv(self, path: str | Path) -> int:
        """Insert rows from a CSV with `id,text` columns. Returns how many were new."""
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or not {"id", "text"} <= set(reader.fieldnames):
                raise ValueError("CSV must have 'id' and 'text' columns")
            rows = [(r["id"].strip(), r["text"].strip()) for r in reader]

        seen: set[str] = set()
        for line, (item_id, text) in enumerate(rows, start=2):
            if not item_id or not text:
                raise ValueError(f"line {line}: empty id or text")
            if item_id in seen:
                raise ValueError(f"line {line}: duplicate id {item_id!r}")
            seen.add(item_id)

        with self.conn:
            before = self.count()
            self.conn.executemany(
                "INSERT OR IGNORE INTO items (id, text) VALUES (?, ?)", rows
            )
            return self.count() - before

    def save_suggestion(
        self, item_id: str, label: str, confidence: float, rationale: str, model: str
    ) -> None:
        _check_label(label)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence out of range: {confidence}")
        with self.conn:
            cur = self.conn.execute(
                "UPDATE items SET ai_label=?, ai_confidence=?, ai_rationale=?, ai_model=?"
                " WHERE id=?",
                (label, confidence, rationale, model, item_id),
            )
        _check_found(cur, item_id)

    def save_review(self, item_id: str, label: str) -> None:
        _check_label(label)
        with self.conn:
            cur = self.conn.execute(
                "UPDATE items SET human_label=?, reviewed_at=? WHERE id=?",
                (label, datetime.now(timezone.utc).isoformat(timespec="seconds"), item_id),
            )
        _check_found(cur, item_id)

    # --- reads --------------------------------------------------------------

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]

    def counts_by_status(self) -> dict[str, int]:
        row = self.conn.execute(
            "SELECT"
            " COUNT(*) FILTER (WHERE ai_label IS NULL AND human_label IS NULL),"
            " COUNT(*) FILTER (WHERE ai_label IS NOT NULL AND human_label IS NULL),"
            " COUNT(*) FILTER (WHERE human_label IS NOT NULL)"
            " FROM items"
        ).fetchone()
        return dict(zip(("pending", "suggested", "reviewed"), row))

    def get(self, item_id: str) -> Item | None:
        row = self.conn.execute("SELECT * FROM items WHERE id=?", (item_id,)).fetchone()
        return Item(**row) if row else None

    def all_items(self) -> list[Item]:
        return self._items("SELECT * FROM items ORDER BY id")

    def pending(self, limit: int | None = None) -> list[Item]:
        sql = "SELECT * FROM items WHERE ai_label IS NULL ORDER BY id"
        return self._items(sql + (f" LIMIT {int(limit)}" if limit else ""))

    def review_queue(self) -> list[Item]:
        """Suggested but unreviewed items, least confident first."""
        return self._items(
            "SELECT * FROM items WHERE ai_label IS NOT NULL AND human_label IS NULL"
            " ORDER BY ai_confidence ASC, id ASC"
        )

    def reviewed(self) -> list[Item]:
        return self._items(
            "SELECT * FROM items WHERE human_label IS NOT NULL ORDER BY reviewed_at, id"
        )

    def _items(self, sql: str) -> list[Item]:
        return [Item(**row) for row in self.conn.execute(sql)]


def _check_label(label: str) -> None:
    if label not in LABELS:
        raise ValueError(f"unknown label {label!r}; allowed: {', '.join(LABELS)}")


def _check_found(cur: sqlite3.Cursor, item_id: str) -> None:
    if cur.rowcount == 0:
        raise KeyError(item_id)
