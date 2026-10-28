"""Pre-label every pending item. API calls run in parallel; writes stay on one thread."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable

from .labelers import Labeler, LabelingError
from .store import Store


@dataclass
class RunResult:
    labeled: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)  # (item id, reason)


def suggest_all(
    store: Store,
    labeler: Labeler,
    limit: int | None = None,
    workers: int = 4,
    on_progress: Callable[[int, int], None] | None = None,
) -> RunResult:
    items = store.pending(limit)
    result = RunResult()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(labeler.suggest, item.text): item.id for item in items}
        for done, future in enumerate(as_completed(futures), start=1):
            item_id = futures[future]
            try:
                s = future.result()
            except LabelingError as e:
                result.failed.append((item_id, str(e)))
            except BaseException:
                for f in futures:  # fatal (e.g. bad API key): stop scheduling new calls
                    f.cancel()
                raise
            else:
                store.save_suggestion(item_id, s.label, s.confidence, s.rationale, labeler.name)
                result.labeled += 1
            if on_progress:
                on_progress(done, len(items))
    return result
