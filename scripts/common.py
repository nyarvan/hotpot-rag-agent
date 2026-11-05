"""Shared helpers for the analysis scripts (not run on its own)."""
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

DEV_FILE = Path('runs/agent_dev25.jsonl')
DEV_SIZE = 25
HARNESS_RUNS = Path('../rag-eval-harness/runs')
HARNESS_DATA = Path('../rag-eval-harness/data')
QUESTION_TYPES = ('bridge', 'comparison')


def read_lines(path: str | Path) -> list[str]:
    """Non-blank lines of a JSONL file, verbatim."""
    text = Path(path).read_text(encoding='utf-8')
    return [line for line in text.splitlines() if line.strip()]


def line_id(line: str, path: str | Path) -> str:
    """`question_id` of one JSONL line; ValueError names the file."""
    try:
        return json.loads(line)['question_id']
    except (ValueError, KeyError, TypeError) as error:
        raise ValueError(f'{path}: line without question_id') from error


def read_rows(path: str | Path) -> dict[str, dict[str, Any]]:
    """Parsed JSONL rows keyed by `question_id`, in file order.

    Raises `ValueError` on a duplicate question_id.
    """
    rows: dict[str, dict[str, Any]] = {}
    for line in read_lines(path):
        row = json.loads(line)
        qid = line_id(line, path)
        if qid in rows:
            raise ValueError(f'{path}: duplicate question_id {qid}')
        rows[qid] = row
    return rows


def dev_ids(path: str | Path = DEV_FILE) -> set[str]:
    """The 25 development question ids.

    Raises `ValueError` unless the file has exactly 25 lines with
    25 distinct ids.
    """
    lines = read_lines(path)
    ids = {line_id(line, path) for line in lines}
    if len(lines) != DEV_SIZE or len(ids) != DEV_SIZE:
        raise ValueError(
            f'{path}: expected {DEV_SIZE} lines with distinct ids, got '
            f'{len(lines)} lines and {len(ids)} ids'
        )
    return ids


def question_types() -> dict[str, str]:
    """`question_id` -> 'bridge' | 'comparison' from the cached subset.

    Needs `HF_HUB_OFFLINE=1` so that nothing is downloaded.
    """
    from rag_eval.data import load_hotpotqa

    subset = load_hotpotqa(cache_dir=HARNESS_DATA / 'hf_cache')
    return {q.question_id: q.type for q in subset.questions}


def type_of(types: Mapping[str, str], question_id: str) -> str:
    """Question type of an id; ValueError if unknown or not a known type."""
    if question_id not in types:
        raise ValueError(f'question_id {question_id} is not in the subset')
    question_type = types[question_id]
    if question_type not in QUESTION_TYPES:
        raise ValueError(
            f'question {question_id} has unexpected type {question_type!r}'
        )
    return question_type


def check_unique_names(paths: Iterable[str | Path]) -> None:
    """Raise `ValueError` when two inputs share a file name."""
    names = [Path(p).name for p in paths]
    clash = sorted({n for n in names if names.count(n) > 1})
    if clash:
        raise ValueError(f'inputs share a file name: {", ".join(clash)}')


def write_lines(path: Path, lines: Sequence[str]) -> None:
    """Write JSONL lines; no lines gives an empty file, not a lone newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = '\n'.join(lines) + '\n' if lines else ''
    path.write_text(text, encoding='utf-8')


def drop_dev(
        lines: Sequence[str], dev: set[str], path: str | Path,
) -> list[str]:
    """Lines without the dev questions; exactly `DEV_SIZE` must be dropped."""
    kept = [line for line in lines if line_id(line, path) not in dev]
    dropped = len(lines) - len(kept)
    if dropped != DEV_SIZE:
        raise ValueError(
            f'{path}: expected to drop {DEV_SIZE} dev rows, dropped {dropped}'
        )
    return kept
