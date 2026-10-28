import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from labeler.store import Store  # noqa: E402


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "test.db")
    yield s
    s.close()


@pytest.fixture
def csv_file(tmp_path):
    def make(text: str) -> Path:
        p = tmp_path / "items.csv"
        p.write_text(text, encoding="utf-8")
        return p
    return make
