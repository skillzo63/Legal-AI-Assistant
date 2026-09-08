"""Golden eval dataset: auto-sampled + hand-written queries.

Auto set
--------
Randomly samples ``EVAL_GOLDEN_SET_SIZE`` records from the indexed metadata.
Each sample's question is the eval query; the record's ``id`` is the
ground-truth document the retriever must surface. Seeded for reproducibility.

Hand-written set
----------------
Loaded from ``hand_written.json`` in this directory. These have no
``expected_id``, so they contribute only to the faithfulness eval (retrieval
recall can't be checked without a known ground-truth doc).
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evals.config import eval_settings


@dataclass
class EvalSample:
    """One evaluation query with optional ground-truth document indicators."""

    query: str
    expected_id: Any | None = None  # None for hand-written samples
    expected_doc_id: int | None = None
    expected_citation: str | None = None
    expected_answer: str = ""       # reference answer (informational, not scored)
    category: str = "legal"         # "legal", "clear_legal", "ambiguous_legal", "casual"


def build_auto_set(
    metadata: list[dict[str, Any]],
    n: int,
    seed: int,
) -> list[EvalSample]:
    """Randomly sample ``n`` records from the index as eval queries.

    Args:
        metadata: Full list of indexed records (from ``metadata.json``).
        n: Number of samples to draw.
        seed: Random seed for reproducibility.

    Returns:
        List of EvalSample with expected_id, expected_doc_id, and expected_citation.
    """
    rng = random.Random(seed)
    sampled = rng.sample(metadata, min(n, len(metadata)))
    return [
        EvalSample(
            query=entry.get("question", ""),
            expected_id=entry.get("chunk_id", entry.get("id")),
            expected_doc_id=entry.get("doc_id"),
            expected_citation=entry.get("citation", entry.get("source_name")),
            expected_answer=entry.get("answer", ""),
            category="legal",
        )
        for entry in sampled
    ]


def load_hand_written() -> list[EvalSample]:
    """Load hand-written queries from ``hand_written.json``.

    Each entry must have ``"query"`` and may have ``"expected_answer"``.
    Missing file → empty list (eval still runs on auto set only).

    Returns:
        List of EvalSample with ``expected_id=None``.
    """
    path = Path(__file__).parent / "hand_written.json"
    if not path.exists():
        return []
    items: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    return [
        EvalSample(
            query=item["query"],
            expected_id=None,
            expected_answer=item.get("expected_answer", ""),
            category=item.get("category", "clear_legal"),
        )
        for item in items
    ]


def load_golden_set(metadata: list[dict[str, Any]]) -> list[EvalSample]:
    """Build the full golden set: auto samples + hand-written queries.

    Args:
        metadata: Full list of indexed records.

    Returns:
        Combined list, auto set first, then hand-written, deduplicated by
        query text so a hand-written query that matches a sampled one counts
        once.
    """
    auto = build_auto_set(metadata, eval_settings.golden_set_size, eval_settings.random_seed)
    hand = load_hand_written()
    seen = {s.query for s in auto}
    deduped: list[EvalSample] = []
    for sample in hand:
        if sample.query in seen:
            continue
        seen.add(sample.query)
        deduped.append(sample)
    return auto + deduped
