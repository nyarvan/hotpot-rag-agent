"""The `run` command: the agent over the first n questions, as a run file.

Every extra hop costs one decide request, so the run is announced with an
upper-bound estimate and needs a confirmation. Answers and decisions that are
already cached are free, which is why the estimate is an upper bound. The
estimate is the harness formula, so an agent run and a baseline run can be
compared on the same footing.
"""
import argparse
import os
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from statistics import fmean
from typing import TextIO

from rag_eval.cost import (
    CostEstimate,
    estimate_generation,
    format_estimate,
)
from rag_eval.data import DEFAULT_N, Document
from rag_eval.generate import DEFAULT_MAX_TOKENS as GENERATE_MAX_TOKENS
from rag_eval.generate import SYSTEM_PROMPT as GENERATE_SYSTEM_PROMPT
from rag_eval.generate import format_context
from rag_eval.records import write_records

from . import decide, llm
from .components import DEFAULT_CHAT_MODEL, Components
from .graph import build_graph
from .nodes import CONTEXT_K
from .run import AgentRecord, run_agent

EXIT_CANCELLED = 3
BASELINE_MODEL = 'gpt-4o'
CANCELLED_MESSAGE = (
    'Cancelled: nothing was run. Pass --yes to run without asking.'
)


def average_paragraph_chars(documents: Mapping[str, Document]) -> float:
    """Mean length of a paragraph as it appears in a prompt."""
    return fmean(len(format_context(d)) for d in documents.values())


def estimate_run(
        model: str, *, n_questions: int, decide_calls: int,
        avg_paragraph_chars: float,
) -> CostEstimate:
    """Harness formula for `decide_calls` decisions plus the answers."""
    deciding = estimate_generation(
        model, n_questions=decide_calls, context_k=CONTEXT_K,
        avg_paragraph_chars=avg_paragraph_chars,
        system_chars=len(decide.SYSTEM_PROMPT),
        max_output_tokens=llm.DEFAULT_MAX_TOKENS,
    )
    answering = estimate_generation(
        model, n_questions=n_questions, context_k=CONTEXT_K,
        avg_paragraph_chars=avg_paragraph_chars,
        system_chars=len(GENERATE_SYSTEM_PROMPT),
        max_output_tokens=GENERATE_MAX_TOKENS,
    )
    usd = None if deciding.usd is None or answering.usd is None else (
        deciding.usd + answering.usd
    )
    return CostEstimate(
        calls=deciding.calls + answering.calls,
        input_tokens=deciding.input_tokens + answering.input_tokens,
        output_tokens=deciding.output_tokens + answering.output_tokens,
        usd=usd,
    )


def summarize(
        records: Sequence[AgentRecord], model: str,
        avg_paragraph_chars: float,
) -> str:
    """Run statistics and the formula cost of a fresh run of these counts."""
    total = len(records)
    multi_hop = sum(1 for r in records if r.hops > 1)
    llm_calls = sum(r.llm_calls for r in records)
    decide_calls = llm_calls - total
    lines = [
        f'Questions: {total}',
        f'Multi-hop (hops > 1): {multi_hop} ({multi_hop / total:.1%})',
        f'LLM calls: {llm_calls}',
    ]
    agent = estimate_run(
        model, n_questions=total, decide_calls=decide_calls,
        avg_paragraph_chars=avg_paragraph_chars,
    )
    baseline = estimate_run(
        BASELINE_MODEL, n_questions=total, decide_calls=0,
        avg_paragraph_chars=avg_paragraph_chars,
    )
    lines.append(format_estimate(f'Agent with {model}', agent))
    lines.append(format_estimate(
        f'Single-step baseline with {BASELINE_MODEL}', baseline,
    ))
    if agent.usd and baseline.usd is not None:
        lines.append(f'Baseline / agent cost: {baseline.usd / agent.usd:.1f}x')
    return '\n'.join(lines)


def check_out_path(path: Path) -> None:
    """Fail before any paid call if `path` cannot be written."""
    if path.is_dir():
        raise ValueError(f'--out is a directory: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)


def write_atomically(path: Path, records: Sequence[AgentRecord]) -> None:
    """Write via a temp file in the same directory, then replace `path`."""
    temp = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        write_records(temp, records)
        os.replace(temp, path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def run_batch(
        args: argparse.Namespace, components: Components, *,
        confirm_fn: Callable[..., bool], out: TextIO,
) -> int:
    """Run the first `args.n` questions; return an exit code.

    Raises `ValueError` if `args.n` exceeds the subset. Records are written
    only after every question has finished, so a failure leaves no file.
    """
    questions = components.subset.questions
    if args.n > len(questions):
        raise ValueError(
            f'--n {args.n} exceeds the {len(questions)} subset questions'
        )
    check_out_path(args.out)
    avg_chars = average_paragraph_chars(components.documents)
    if args.max_hops > 1:
        estimate = estimate_run(
            DEFAULT_CHAT_MODEL, n_questions=args.n,
            decide_calls=args.n * (args.max_hops - 1),
            avg_paragraph_chars=avg_chars,
        )
        print(format_estimate(
            f'Upper bound with {DEFAULT_CHAT_MODEL}', estimate,
        ), file=out)
        if not confirm_fn('Run it?', assume_yes=args.yes):
            print(CANCELLED_MESSAGE, file=out)
            return EXIT_CANCELLED
    graph = build_graph(
        components.searcher, components.decider, components.answerer,
        components.documents, max_hops=args.max_hops,
    )
    records = run_agent(
        questions[:args.n], graph, components.searcher,
        prefetch_questions=questions[:DEFAULT_N],
    )
    write_atomically(args.out, records)
    print(f'Wrote {len(records)} records to {args.out}', file=out)
    print(summarize(records, DEFAULT_CHAT_MODEL, avg_chars), file=out)
    return 0
