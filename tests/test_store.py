import pytest


def test_import_and_reimport_is_idempotent(store, csv_file):
    p = csv_file("id,text\n1,first\n2,second\n")
    assert store.import_csv(p) == 2
    assert store.import_csv(p) == 0
    assert store.count() == 2


def test_import_ignores_extra_columns_and_strips(store, csv_file):
    assert store.import_csv(csv_file("id,text,extra\n 1 , hello ,x\n")) == 1
    assert store.get("1").text == "hello"


@pytest.mark.parametrize("content", [
    "id,body\n1,x\n",          # missing text column
    "id,text\n1,\n",           # empty text
    "id,text\n1,a\n1,b\n",     # duplicate id
])
def test_import_rejects_bad_files_without_partial_writes(store, csv_file, content):
    with pytest.raises(ValueError):
        store.import_csv(csv_file(content))
    assert store.count() == 0


def test_status_lifecycle(store, csv_file):
    store.import_csv(csv_file("id,text\n1,a\n2,b\n3,c\n"))
    assert store.counts_by_status() == {"pending": 3, "suggested": 0, "reviewed": 0}

    store.save_suggestion("1", "rock", 0.9, "guitars", "m")
    store.save_suggestion("2", "pop", 0.4, "vocals", "m")
    assert store.get("1").status == "suggested"
    assert [it.id for it in store.pending()] == ["3"]

    store.save_review("2", "rock")
    assert store.get("2").status == "reviewed"
    assert store.get("2").reviewed_at is not None
    assert store.counts_by_status() == {"pending": 1, "suggested": 1, "reviewed": 1}


def test_review_queue_is_least_confident_first(store, csv_file):
    store.import_csv(csv_file("id,text\n1,a\n2,b\n3,c\n"))
    store.save_suggestion("1", "rock", 0.9, "", "m")
    store.save_suggestion("2", "pop", 0.2, "", "m")
    store.save_suggestion("3", "pop", 0.5, "", "m")
    assert [it.id for it in store.review_queue()] == ["2", "3", "1"]
    store.save_review("2", "pop")
    assert [it.id for it in store.review_queue()] == ["3", "1"]


def test_pending_limit(store, csv_file):
    store.import_csv(csv_file("id,text\n1,a\n2,b\n3,c\n"))
    assert len(store.pending(limit=2)) == 2


def test_rejects_unknown_label_bad_confidence_and_missing_item(store, csv_file):
    store.import_csv(csv_file("id,text\n1,a\n"))
    with pytest.raises(ValueError):
        store.save_suggestion("1", "polka", 0.5, "", "m")
    with pytest.raises(ValueError):
        store.save_suggestion("1", "rock", 1.5, "", "m")
    with pytest.raises(ValueError):
        store.save_review("1", "polka")
    with pytest.raises(KeyError):
        store.save_review("nope", "rock")


def test_data_survives_reopen(tmp_path, csv_file):
    from labeler.store import Store
    s = Store(tmp_path / "x.db")
    s.import_csv(csv_file("id,text\n1,a\n"))
    s.save_suggestion("1", "rock", 0.7, "r", "m")
    s.save_review("1", "pop")
    s.close()
    s2 = Store(tmp_path / "x.db")
    it = s2.get("1")
    assert (it.ai_label, it.human_label) == ("rock", "pop")
    s2.close()
