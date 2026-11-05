"""Per-question wall time of the full agent graph with empty caches (PAID).

Compares the baseline (`max_hops=1`, the single-step pipeline) with the
agent (`max_hops=2`). Both graphs are built up front, each with its own
temporary rerank, decide and answer caches, so the agent never reuses the
baseline's rerank scores. Questions alternate order (baseline first, then
agent first, ...) so that drift in API latency does not favour one config.

Nothing is prefetched: the timed path of every question includes its own
single-question hop-1 dense search and cross-encoder rerank of 30
candidates, as in `rag_eval.latency` (a batched prefetch would hide that
cost and, with an empty rerank cache, would score 200 x 30 pairs). The
dense encoder and the cross-encoder are loaded once and shared, and two
untimed warm-up questions (at least one with hops == 2) run on each config.

Sample: 5 held-out questions with hops == 2 and 5 with hops == 1 in
runs/agent_200.jsonl, first in subset order. The sample is 50% hop 2, the
held-out set is not, so the summary also gives a mean weighted by the
real share w of hops == 2 among held-out questions. Hops seen in this run
can differ from the run file (single-question encoding may reorder
near-ties); both are shown, strata use the run file's hops.

Cost: about 12 x 1 + 12 x 2 gpt-4o-mini calls (10 questions + 2 warm-ups
per config), well under $0.01. Needs --yes. Rows go to
runs/latency_<YYYYMMDD>.jsonl, metadata to runs/latency_<YYYYMMDD>.json.

Run from the repo root:
    venv/bin/python -m scripts.latency --yes --env-file .env
"""
import argparse
import json
import os
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

# Offline before any model library is imported
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import numpy as np  # noqa: E402
from langgraph.graph.state import CompiledStateGraph  # noqa: E402
from rag_eval.data import (  # noqa: E402
    DEFAULT_N,
    Document,
    Question,
    load_hotpotqa,
)
from rag_eval.generate import Generator, OpenAIChat  # noqa: E402
from rag_eval.rerank import (  # noqa: E402
    DEFAULT_RERANKER,
    MAX_LENGTH,
    PairScorer,
    RerankedRetriever,
)
from rag_eval.retrieve import DenseRetriever  # noqa: E402

from agent.components import (  # noqa: E402
    CANDIDATES_K,
    DEFAULT_CHAT_MODEL,
    DEFAULT_DATA_DIR,
    DENSE_MODEL,
    RerankSearcher,
    check_api_key,
    check_caches,
)
from agent.decide import DECIDE_SCHEMA, SCHEMA_NAME, LLMDecider  # noqa: E402
from agent.graph import SEARCH_K, build_graph  # noqa: E402
from agent.llm import OpenAIJsonChat  # noqa: E402
from agent.nodes import (  # noqa: E402
    CONTEXT_K,
    Answerer,
    Decider,
    Decision,
    Searcher,
)

from .common import DEV_FILE, dev_ids, read_rows  # noqa: E402

PER_HOPS = 5
POOL_SIZE = 200
WARMUPS = 2
RUN_FILE = 'runs/agent_200.jsonl'
OUT_DIR = Path('runs')
P90 = 90
MIN_N_FOR_P90 = 10
CONFIGS = {'baseline': 1, 'agent': 2}  # name -> max_hops
CALLS_PER_QUESTION = {'baseline': 1, 'agent': 2}  # upper bound
PARTS = ('hop1', 'decide', 'hop2', 'generate', 'other')


@dataclass
class Timings:
    """Seconds spent in each timed part during one `graph.invoke`."""

    search: list[float] = field(default_factory=list)
    decide: float = 0.0
    generate: float = 0.0

    def reset(self) -> None:
        """Forget the previous question."""
        self.search.clear()
        self.decide = 0.0
        self.generate = 0.0


class TimedSearcher:
    """Searcher proxy that records the duration of every search."""

    def __init__(self, inner: Searcher, timings: Timings) -> None:
        self._inner = inner
        self._timings = timings

    def search(self, query: str, k: int) -> Sequence[str]:
        """Delegate and time."""
        start = time.perf_counter()
        result = self._inner.search(query, k)
        self._timings.search.append(time.perf_counter() - start)
        return result


class TimedDecider:
    """Decider proxy that records the duration of the decide call."""

    def __init__(self, inner: Decider, timings: Timings) -> None:
        self._inner = inner
        self._timings = timings

    def decide(self, question: str, contexts: Sequence[Document]) -> Decision:
        """Delegate and time."""
        start = time.perf_counter()
        result = self._inner.decide(question, contexts)
        self._timings.decide += time.perf_counter() - start
        return result


class TimedAnswerer:
    """Answerer proxy that records the duration of the answer call."""

    def __init__(self, inner: Answerer, timings: Timings) -> None:
        self._inner = inner
        self._timings = timings

    def answer(self, question: str, contexts: Sequence[Document]) -> str:
        """Delegate and time."""
        start = time.perf_counter()
        result = self._inner.answer(question, contexts)
        self._timings.generate += time.perf_counter() - start
        return result


@dataclass
class Rig:
    """One config: a compiled graph, its timings and its decider."""

    name: str
    graph: CompiledStateGraph
    timings: Timings
    decider: LLMDecider


def pick_sample(
        questions: Sequence[Question], hops_by_id: Mapping[str, int],
        skip: set[str],
) -> list[Question]:
    """First `PER_HOPS` held-out questions with 2 hops, then with 1 hop."""
    picked: dict[int, list[Question]] = {1: [], 2: []}
    for question in questions[:POOL_SIZE]:
        hops = hops_by_id.get(question.question_id)
        if question.question_id in skip or hops not in picked:
            continue
        if len(picked[hops]) < PER_HOPS:
            picked[hops].append(question)
    return picked[2] + picked[1]


def pick_warmups(
        questions: Sequence[Question], hops_by_id: Mapping[str, int],
        sample: Sequence[Question], skip: set[str],
) -> list[Question]:
    """`WARMUPS` questions outside the sample, the first with hops == 2."""
    taken = {q.question_id for q in sample}
    free = [
        q for q in questions[:POOL_SIZE]
        if q.question_id not in taken and q.question_id not in skip
        and q.question_id in hops_by_id
    ]
    first = next(q for q in free if hops_by_id[q.question_id] == 2)
    rest = [q for q in free if q is not first]
    return [first] + rest[:WARMUPS - 1]


def heldout_share(hops_by_id: Mapping[str, int], skip: set[str]) -> float:
    """Share of hops == 2 among the held-out questions of the run file."""
    held = [h for q, h in hops_by_id.items() if q not in skip]
    return sum(1 for h in held if h == 2) / len(held)


def build_rig(
        name: str, root: Path, dense: DenseRetriever, scorer: PairScorer,
        subset_documents: Sequence[Document],
        documents: Mapping[str, Document],
) -> Rig:
    """Graph with its own empty rerank, decide and answer caches."""
    timings = Timings()
    reranked = RerankedRetriever(
        dense, subset_documents, candidates_k=CANDIDATES_K,
        cache_dir=root / 'rerank', scorer=scorer,
    )
    decider = LLMDecider(
        OpenAIJsonChat(
            DEFAULT_CHAT_MODEL, schema=DECIDE_SCHEMA, schema_name=SCHEMA_NAME,
        ),
        root / 'decide',
    )
    answerer = Generator(OpenAIChat(DEFAULT_CHAT_MODEL), root / 'answer')
    graph = build_graph(
        TimedSearcher(RerankSearcher(reranked), timings),
        TimedDecider(decider, timings),
        TimedAnswerer(answerer, timings),
        documents, max_hops=CONFIGS[name], search_k=SEARCH_K,
        context_k=CONTEXT_K,
    )
    return Rig(name, graph, timings, decider)


def time_question(
        rig: Rig, question: Question, run_hops: int,
) -> dict[str, Any]:
    """Run the graph once and return the timing row."""
    rig.timings.reset()
    start = time.perf_counter()
    state = rig.graph.invoke({'question': question.question})
    total = time.perf_counter() - start
    t = rig.timings
    row = {
        'question_id': question.question_id,
        'run_hops': run_hops,
        'config': rig.name,
        'max_hops': CONFIGS[rig.name],
        'hops': state['hops'],
        'total': total,
        'hop1': t.search[0],
        'decide': t.decide,
        'hop2': t.search[1] if len(t.search) > 1 else 0.0,
        'generate': t.generate,
    }
    row['other'] = total - sum(row[p] for p in PARTS[:-1])
    return row


def stats(values: Sequence[float]) -> str:
    """'n  mean  median[  p90]'; p90 only from `MIN_N_FOR_P90` values."""
    text = f'n={len(values):<3}mean {mean(values):>7.2f}  ' \
           f'median {median(values):>7.2f}'
    if len(values) >= MIN_N_FOR_P90:
        text += f'  p90 {float(np.percentile(values, P90)):>7.2f}'
    return text


def print_table(config: str, rows: Sequence[Mapping[str, Any]]) -> None:
    """Per-question table of one config (seconds)."""
    print(f'\n{config} (max_hops={CONFIGS[config]}), seconds')
    print(f'{"question_id":<26}{"run":>4}{"now":>4}{"total":>8}'
          + ''.join(f'{p:>9}' for p in PARTS))
    for r in rows:
        print(f'{r["question_id"]:<26}{r["run_hops"]:>4}{r["hops"]:>4}'
              f'{r["total"]:>8.2f}' + ''.join(f'{r[p]:>9.2f}' for p in PARTS))


def print_block(
        title: str, by_stratum: Mapping[int, Sequence[float]], w: float,
) -> None:
    """Per-stratum, pooled and weighted statistics of one quantity."""
    print(f'  {title}')
    for stratum in (1, 2):
        print(f'    run hops={stratum}  {stats(by_stratum[stratum])}')
    pooled = [v for s in (1, 2) for v in by_stratum[s]]
    print(f'    sample (50% hop 2)  {stats(pooled)}')
    weighted = (
        w * mean(by_stratum[2]) + (1 - w) * mean(by_stratum[1]))
    print(f'    weighted mean (w={w:.3f} hop 2): {weighted:.2f}')


def print_summary(
        rows: Sequence[Mapping[str, Any]], w: float,
) -> None:
    """Totals per config and the paired agent - baseline difference."""
    print('\ntotal seconds by stratum = hops in runs/agent_200.jsonl')
    totals: dict[str, dict[str, float]] = {c: {} for c in CONFIGS}
    for config in CONFIGS:
        by_stratum: dict[int, list[float]] = {1: [], 2: []}
        for r in rows:
            if r['config'] == config:
                by_stratum[r['run_hops']].append(r['total'])
                totals[config][r['question_id']] = r['total']
        print_block(config, by_stratum, w)
    diffs: dict[int, list[float]] = {1: [], 2: []}
    for r in rows:
        if r['config'] == 'agent':
            diffs[r['run_hops']].append(
                r['total'] - totals['baseline'][r['question_id']])
    print('  paired difference agent - baseline')
    print_block('(seconds per question)', diffs, w)


def device_of(scorer: Any) -> str:
    """Torch device of the cross-encoder, 'unknown' if not found."""
    try:
        return str(scorer.model.device)
    except AttributeError:
        return 'unknown'


def main(argv: Sequence[str] | None = None) -> int:
    """Run the timing after an explicit --yes."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--yes', action='store_true',
                        help='confirm the paid run')
    parser.add_argument('--env-file',
                        help='dotenv file with OPENAI_API_KEY (not printed)')
    parser.add_argument('--data-dir', default=str(DEFAULT_DATA_DIR))
    args = parser.parse_args(argv)
    total_calls = sum(
        (2 * PER_HOPS + WARMUPS) * CALLS_PER_QUESTION[c] for c in CONFIGS)
    print(f'Estimate: up to {total_calls} gpt-4o-mini requests '
          f'(10 questions + {WARMUPS} warm-ups per config), under $0.01.')
    if not args.yes:
        print('Add --yes to run.')
        return 1
    if args.env_file:
        if not Path(args.env_file).is_file():
            print(f'error: env file not found: {args.env_file}',
                  file=sys.stderr)
            return 1
        from dotenv import load_dotenv
        load_dotenv(args.env_file, override=True)
    check_api_key(os.environ)
    data_dir = Path(args.data_dir)
    check_caches(data_dir)
    stamp = datetime.now()
    rows_path = OUT_DIR / f'latency_{stamp:%Y%m%d}.jsonl'
    meta_path = rows_path.with_suffix('.json')
    if rows_path.exists() or meta_path.exists():
        print(f'error: {rows_path} or {meta_path} exists', file=sys.stderr)
        return 1

    subset = load_hotpotqa(cache_dir=data_dir / 'hf_cache')
    hops_by_id = {
        qid: row['hops'] for qid, row in read_rows(RUN_FILE).items()}
    dev = dev_ids(DEV_FILE)
    sample = pick_sample(subset.questions, hops_by_id, dev)
    if len(sample) != 2 * PER_HOPS:
        print(f'only {len(sample)} sample questions found', file=sys.stderr)
        return 1
    warmups = pick_warmups(subset.questions, hops_by_id, sample, dev)
    share = heldout_share(hops_by_id, dev)
    documents = {d.doc_id: d for d in subset.documents}
    dense = DenseRetriever(
        DENSE_MODEL, subset.documents, subset_size=DEFAULT_N,
        data_dir=data_dir,
    )
    from sentence_transformers import CrossEncoder
    scorer = CrossEncoder(DEFAULT_RERANKER, max_length=MAX_LENGTH)

    with ExitStack() as stack:
        rigs = {
            name: build_rig(
                name, Path(stack.enter_context(tempfile.TemporaryDirectory())),
                dense, scorer, subset.documents, documents)
            for name in CONFIGS
        }
        for rig in rigs.values():
            for question in warmups:
                rig.graph.invoke({'question': question.question})
        rows = []
        for i, question in enumerate(sample):
            order = list(CONFIGS) if i % 2 == 0 else list(CONFIGS)[::-1]
            for name in order:
                rows.append(time_question(
                    rigs[name], question, hops_by_id[question.question_id]))
        fallbacks = {name: rig.decider.fallbacks for name, rig in rigs.items()}

    for config in CONFIGS:
        print_table(config, [r for r in rows if r['config'] == config])
    print('\ndecider fallbacks (incl. warm-ups): ' + ', '.join(
        f'{c}={n}' for c, n in fallbacks.items()))
    print_summary(rows, share)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows_path.write_text(
        ''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    meta_path.write_text(json.dumps({
        'timestamp': stamp.isoformat(timespec='seconds'),
        'sample_ids': [q.question_id for q in sample],
        'warmup_ids': [q.question_id for q in warmups],
        'torch_device': device_of(scorer),
        'dense_model': DENSE_MODEL,
        'reranker': DEFAULT_RERANKER,
        'chat_model': DEFAULT_CHAT_MODEL,
        'w_hop2_share_heldout': share,
        'fallbacks': fallbacks,
    }, indent=2) + '\n', encoding='utf-8')
    print(f'\nwrote {rows_path} and {meta_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
