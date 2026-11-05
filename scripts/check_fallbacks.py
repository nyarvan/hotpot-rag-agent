"""Find questions whose hop-1 decide call fell back instead of answering.

`LLMDecider.decide` caches a reply only when it parsed, so a fallback
(invalid JSON or a refusal) leaves no cache file and the graph answers after
hop 1. For every record with `hops == 1` this rebuilds the hop-1 decide
prompt (top-5 titles = `retrieved[:5]`) and checks that its cache file
exists; a missing file means a fallback. Sanity check: for `hops == 2`
records the hop-1 decide file must exist. Their hop-1 top-5 is not stored,
so it is the first 3 titles plus any ordered pair of the remaining
`retrieved` titles.

Each hops==1 record is classified by reading its cache file and parsing it
with `DecideReply`: (a) enough=true, (b) enough=false but the follow-up was
dropped (empty or a repeat of an earlier query), (c) missing or invalid, i.e.
a fallback. This checks the cache as it is now, not what happened at run
time: a file could have been added or removed since.

Run from the repo root (offline, no API calls):
    HF_HUB_OFFLINE=1 venv/bin/python -m scripts.check_fallbacks \
        [--run runs/agent_200.jsonl] [--cache-dir data/decide_cache]
"""
import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from itertools import permutations
from pathlib import Path

from pydantic import ValidationError
from rag_eval.data import Document, load_hotpotqa
from rag_eval.generate import cache_key

from agent.components import DEFAULT_CHAT_MODEL
from agent.decide import (
    DECIDE_SCHEMA,
    SCHEMA_NAME,
    SYSTEM_PROMPT,
    DecideReply,
    build_decide_prompt,
)
from agent.llm import OpenAIJsonChat
from agent.nodes import CONTEXT_K, FIRST_HOP_K, clean_follow_up, merge_titles

from .common import HARNESS_DATA, read_rows

DEFAULT_RUN = 'runs/agent_200.jsonl'
DEFAULT_CACHE = 'data/decide_cache'


def cache_path(
        chat_name: str, cache_dir: Path, question: str,
        titles: Sequence[str], documents: Mapping[str, Document],
) -> Path:
    """Cache file of the decide reply for this context."""
    user = build_decide_prompt(question, [documents[t] for t in titles])
    key = cache_key(chat_name, SYSTEM_PROMPT, user)
    return cache_dir / f'{key}.json'


def cache_exists(
        chat_name: str, cache_dir: Path, question: str,
        titles: Sequence[str], documents: Mapping[str, Document],
) -> bool:
    """Whether the decide reply for this context is cached."""
    return cache_path(
        chat_name, cache_dir, question, titles, documents).exists()


def classify(path: Path, queries: Sequence[str]) -> str:
    """'enough', 'dropped' (follow-up empty or repeated), 'follow_up' or
    'fallback' (missing or invalid cache file)."""
    try:
        text = json.loads(path.read_text(encoding='utf-8'))['reply']
        reply = DecideReply.model_validate_json(text)
    except (OSError, ValueError, KeyError, TypeError, ValidationError):
        return 'fallback'
    if reply.enough:
        return 'enough'
    follow_up = clean_follow_up(reply.follow_up).strip()
    if not follow_up or follow_up in queries:
        return 'dropped'
    return 'follow_up'


def hop1_cached_for_two_hops(
        chat_name: str, cache_dir: Path, row: Mapping,
        documents: Mapping[str, Document],
) -> bool:
    """Whether some hop-1 top-5 consistent with the record is cached."""
    retrieved = row['retrieved']
    head = list(retrieved[:FIRST_HOP_K])
    pool = retrieved[FIRST_HOP_K:]
    for extra in permutations(pool, CONTEXT_K - FIRST_HOP_K):
        titles = merge_titles([head + list(extra)])
        if cache_exists(
                chat_name, cache_dir, row['question'], titles, documents):
            return True
    return False


def main(argv: Sequence[str] | None = None) -> int:
    """Print counts and fallback ids; exit 1 if the sanity check fails."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--run', default=DEFAULT_RUN)
    parser.add_argument('--cache-dir', default=DEFAULT_CACHE)
    args = parser.parse_args(argv)
    cache_dir = Path(args.cache_dir)
    chat_name = OpenAIJsonChat(
        DEFAULT_CHAT_MODEL, schema=DECIDE_SCHEMA, schema_name=SCHEMA_NAME,
    ).name
    subset = load_hotpotqa(cache_dir=HARNESS_DATA / 'hf_cache')
    documents = {d.doc_id: d for d in subset.documents}
    rows = read_rows(args.run)

    one_hop = [r for r in rows.values() if r['hops'] == 1]
    two_hop = [r for r in rows.values() if r['hops'] == 2]
    classes = {
        r['question_id']: classify(cache_path(
            chat_name, cache_dir, r['question'],
            merge_titles([r['retrieved']]), documents), r['queries'])
        for r in one_hop
    }
    fallbacks = [q for q, c in classes.items() if c == 'fallback']
    missing = [
        r['question_id'] for r in two_hop
        if not hop1_cached_for_two_hops(chat_name, cache_dir, r, documents)
    ]
    print(f'records {len(rows)}: hops==1 {len(one_hop)}, '
          f'hops==2 {len(two_hop)}')
    kinds = list(classes.values())
    counts = {c: kinds.count(c) for c in set(kinds)}
    print(f'hops==1 (a) enough=true: {counts.get("enough", 0)}')
    print(f'hops==1 (b) enough=false, follow-up dropped: '
          f'{counts.get("dropped", 0)}')
    print(f'hops==1 (c) missing or invalid (fallback): {len(fallbacks)}')
    for qid in fallbacks:
        print(f'  {qid}')
    if counts.get('follow_up'):
        print(f'unexpected: {counts["follow_up"]} hops==1 records have a '
              'usable follow-up')
    print(f'sanity: hops==2 with hop-1 decide cache: '
          f'{len(two_hop) - len(missing)}/{len(two_hop)}')
    for qid in missing:
        print(f'  missing: {qid}')
    return 1 if missing else 0


if __name__ == '__main__':
    sys.exit(main())
