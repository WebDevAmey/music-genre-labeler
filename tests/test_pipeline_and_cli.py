import csv
import threading

import pytest

from labeler.cli import main
from labeler.export import COLUMNS, to_csv
from labeler.labelers import LabelingError, Suggestion
from labeler.pipeline import suggest_all
from labeler.store import Store


class ScriptedLabeler:
    name = "scripted"

    def __init__(self, fail_on=()):
        self.fail_on, self.seen = set(fail_on), []
        self.lock = threading.Lock()

    def suggest(self, text):
        with self.lock:
            self.seen.append(text)
        if text in self.fail_on:
            raise LabelingError("nope")
        return Suggestion("rock", 0.5, "because")


class Boom(Exception):
    pass


def test_suggest_all_skips_failures_and_is_resumable(store, csv_file):
    store.import_csv(csv_file("id,text\n1,a\n2,b\n3,c\n"))
    result = suggest_all(store, ScriptedLabeler(fail_on={"b"}))
    assert result.labeled == 2 and result.failed == [("2", "nope")]
    assert [it.id for it in store.pending()] == ["2"]

    # Second run only touches what is still pending.
    labeler = ScriptedLabeler()
    assert suggest_all(store, labeler).labeled == 1
    assert labeler.seen == ["b"]


def test_suggest_all_stops_on_fatal_error(store, csv_file):
    store.import_csv(csv_file("id,text\n1,a\n"))

    class Fatal:
        name = "fatal"

        def suggest(self, text):
            raise Boom()

    with pytest.raises(Boom):
        suggest_all(store, Fatal())
    assert store.counts_by_status()["pending"] == 1


def test_export_columns_and_correctness_flag(store, csv_file):
    store.import_csv(csv_file("id,text\n1,\"has, comma\"\n2,b\n3,c\n"))
    store.save_suggestion("1", "rock", 0.9, "", "m")
    store.save_suggestion("2", "rock", 0.9, "", "m")
    store.save_review("1", "rock")
    store.save_review("2", "pop")
    rows = list(csv.DictReader(to_csv(store.all_items()).splitlines()))
    assert tuple(rows[0].keys()) == COLUMNS
    by_id = {r["id"]: r for r in rows}
    assert by_id["1"]["text"] == "has, comma"
    assert by_id["1"]["ai_correct"] == "True" and by_id["2"]["ai_correct"] == "False"
    assert by_id["3"]["label"] == "" and by_id["3"]["ai_correct"] == ""


def test_cli_end_to_end(tmp_path, capsys):
    db, data, out = tmp_path / "e2e.db", tmp_path / "in.csv", tmp_path / "out.csv"
    data.write_text(
        "id,text\n"
        "a,A heavy metal song with distorted guitar.\n"
        "b,A jazz trio with saxophone and upright bass.\n"
        "c,A dog barks.\n",
        encoding="utf-8",
    )
    assert main(["--db", str(db), "import", str(data)]) == 0
    assert main(["--db", str(db), "suggest", "--labeler", "keyword"]) == 0

    s = Store(db)
    for it in s.review_queue():  # a human accepts everything except one fix
        s.save_review(it.id, "world" if it.id == "c" else it.ai_label)
    s.close()

    assert main(["--db", str(db), "stats"]) == 0
    printed = capsys.readouterr().out
    assert "Reviewed items: 3" in printed and "AI accuracy:    66.7%" in printed

    assert main(["--db", str(db), "export", "-o", str(out)]) == 0
    rows = list(csv.DictReader(out.open()))
    assert {r["id"]: r["label"] for r in rows} == {"a": "rock", "b": "jazz_blues", "c": "world"}
