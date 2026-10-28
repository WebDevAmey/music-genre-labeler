from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_review_flow(tmp_path, monkeypatch):
    monkeypatch.setenv("LABELER_DB", str(tmp_path / "ui.db"))
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception

    at.sidebar.button[0].click().run()                  # load sample
    at.sidebar.radio[0].set_value("Keyword baseline").run()
    at.sidebar.button[0].click().run()                  # AI pre-label
    assert not at.exception
    assert "200 awaiting review" in at.sidebar.caption[0].value

    first = at.radio[0].value                           # preselected AI label
    at.button[0].click().run()                          # accept
    at.radio[0].set_value("other").run()
    at.button[0].click().run()                          # fix
    assert not at.exception
    assert at.sidebar.metric[0].value == "2 / 200"
    assert first in at.radio[0].options
    assert at.metric[0].value == "2"                     # metrics tab rendered
