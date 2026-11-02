"""Command line entry point: `python -m agent.cli ask|run`.

`ask "question"` runs the graph with human review: before the second search
it shows the proposed follow-up and lets the user approve, edit or reject it.
`run --out FILE` runs the graph without review over the first n questions
and writes a harness-compatible run file, after a cost confirmation.
"""
import argparse
import logging
import os
import sqlite3
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO

import openai
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from rag_eval.cost import confirm
from rag_eval.data import DEFAULT_N

from .batch import run_batch
from .components import DEFAULT_DATA_DIR, Components, build_components
from .graph import MAX_HOPS, SEARCH_K, build_graph

DEFAULT_CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[1] / 'data' / 'checkpoints.sqlite'
)
CHOICE_PROMPT = '[a]pprove / [e]dit / [r]eject: '
EDIT_PROMPT = 'New query: '
EMPTY_EDIT_HINT = 'Empty query, type a new one or choose another action.'
INTERRUPT_KEY = '__interrupt__'
MAX_HOPS_LIMIT = 5
MAX_QUESTION_CHARS = 1000
EXIT_ERROR = 1
EXIT_ABORTED = 130
REPLACEMENT_CHAR = '\ufffd'
LOG_FORMAT = '%(asctime)s %(message)s'
PROGRESS_HANDLER = logging.StreamHandler()
PROGRESS_HANDLER.setFormatter(logging.Formatter(LOG_FORMAT))


def _positive_int(text: str) -> int:
    """Argparse type: an integer in 1..MAX_HOPS_LIMIT."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f'not an integer: {text!r}')
    if not 1 <= value <= MAX_HOPS_LIMIT:
        raise argparse.ArgumentTypeError(
            f'must be between 1 and {MAX_HOPS_LIMIT}, got {value}',
        )
    return value


def _question(text: str) -> str:
    """Argparse type: a non-blank question, normalised, not truncated."""
    printable = ''.join(c for c in text if c.isprintable() or c.isspace())
    cleaned = ' '.join(printable.split())
    if not cleaned:
        raise argparse.ArgumentTypeError('question must not be blank')
    if len(cleaned) > MAX_QUESTION_CHARS:
        raise argparse.ArgumentTypeError(
            f'question is longer than {MAX_QUESTION_CHARS} characters',
        )
    return cleaned


def _safe(text: str) -> str:
    """Replace non-printable characters (except newline, tab) for display."""
    return ''.join(
        c if c.isprintable() or c in '\n\t' else REPLACEMENT_CHAR
        for c in text
    )


def _parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(prog='agent')
    commands = parser.add_subparsers(dest='command', required=True)
    ask = commands.add_parser('ask', help='answer one question interactively')
    ask.add_argument('question', type=_question)
    ask.add_argument('--max-hops', type=_positive_int, default=MAX_HOPS)
    ask.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    batch = commands.add_parser(
        'run', help='run the agent over the first n questions',
    )
    batch.add_argument('--out', type=Path, required=True)
    batch.add_argument('--max-hops', type=_positive_int, default=MAX_HOPS)
    batch.add_argument('--n', type=_positive_int, default=DEFAULT_N)
    batch.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    batch.add_argument(
        '--yes', action='store_true', help='do not ask before paid calls',
    )
    return parser


def _review(
        payload: Mapping[str, Any], input_fn: Callable[[str], str],
        out: TextIO,
) -> str:
    """Show the proposed follow-up; return the resume value for the graph."""
    titles = ', '.join(_safe(t) for t in payload['titles'])
    print(f'Context: {titles}', file=out)
    print(f'Proposed follow-up: {_safe(payload["follow_up"])}', file=out)
    while True:
        choice = input_fn(CHOICE_PROMPT).strip().lower()
        if choice in ('a', 'approve'):
            return payload['follow_up']
        if choice in ('r', 'reject'):
            return ''
        if choice not in ('e', 'edit'):
            print('Unknown choice, enter a, e or r.', file=out)
            continue
        typed = input_fn(EDIT_PROMPT)
        if typed.strip():
            return typed
        print(EMPTY_EDIT_HINT, file=out)


def _run_with_review(
        graph: Any, question: str, config: Mapping[str, Any],
        input_fn: Callable[[str], str], out: TextIO,
) -> tuple[dict[str, Any], bool]:
    """Invoke the graph, resolving reviews; also report if the user edited."""
    result = graph.invoke({'question': question}, config)
    edited = False
    while INTERRUPT_KEY in result:
        payload = result[INTERRUPT_KEY][0].value
        reply = _review(payload, input_fn, out)
        edited = reply not in ('', payload['follow_up'])
        result = graph.invoke(Command(resume=reply), config)
    return result, edited


def _print_result(
        result: Mapping[str, Any], edited: bool, thread_id: str, out: TextIO,
) -> None:
    """Print the answer and run statistics."""
    if edited and result['hops'] == 1:
        print('note: follow-up rejected, answered from hop 1', file=out)
    print(f'answer: {_safe(result["answer"])}', file=out)
    print(f'queries: {result["queries"]}', file=out)
    print(f'titles: {result["titles"]}', file=out)
    print(f'llm_calls: {result["llm_calls"]}', file=out)
    print(f'thread_id: {thread_id}', file=out)


def _ask(
        question: str, graph: Any, thread_id: str,
        input_fn: Callable[[str], str], out: TextIO,
) -> None:
    """Run one question through the graph and print the outcome."""
    config = {'configurable': {'thread_id': thread_id}}
    result, edited = _run_with_review(graph, question, config, input_fn, out)
    _print_result(result, edited, thread_id, out)


def _prefetch_known(question: str, components: Components) -> None:
    """Batch-rank the subset when the question is in it (demo fidelity)."""
    questions = components.subset.questions
    if question in {q.question for q in questions}:
        components.searcher.prefetch(questions[:DEFAULT_N], SEARCH_K)


def _connect(path: Path) -> sqlite3.Connection:
    """Open the checkpoint database, readable by the owner only."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    connection = sqlite3.connect(path, check_same_thread=False)
    os.chmod(path, 0o600)
    return connection


def _run(
        args: argparse.Namespace, components: Components,
        saver: SqliteSaver, input_fn: Callable[[str], str], out: TextIO,
) -> None:
    """Prefetch, build the graph, ask, and always drop the thread's state."""
    _prefetch_known(args.question, components)
    graph = build_graph(
        components.searcher, components.decider, components.answerer,
        components.documents, max_hops=args.max_hops,
        human_review=True, checkpointer=saver,
    )
    thread_id = uuid.uuid4().hex
    try:
        _ask(args.question, graph, thread_id, input_fn, out)
    finally:
        saver.delete_thread(thread_id)


def _enable_progress_log() -> None:
    """Show agent progress on stderr; third-party loggers stay untouched."""
    logger = logging.getLogger(__package__)
    logger.setLevel(logging.INFO)
    if PROGRESS_HANDLER not in logger.handlers:
        logger.addHandler(PROGRESS_HANDLER)


def _ask_command(
        args: argparse.Namespace, components: Components,
        input_fn: Callable[[str], str], out: TextIO, checkpoint_path: Path,
) -> None:
    """Run `ask` with a checkpoint database for the review pause."""
    connection = _connect(checkpoint_path)
    try:
        saver = SqliteSaver(
            connection,
            serde=JsonPlusSerializer(allowed_msgpack_modules=None),
        )
        _run(args, components, saver, input_fn, out)
    finally:
        connection.close()


def main(
        argv: Sequence[str] | None = None, *,
        components_factory: Callable[[Path], Components] = build_components,
        input_fn: Callable[[str], str] = input,
        confirm_fn: Callable[..., bool] = confirm,
        out: TextIO = sys.stdout,
        err: TextIO = sys.stderr,
        checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
) -> int:
    """Run the CLI; return the process exit code."""
    args = _parser().parse_args(argv)
    try:
        components = components_factory(args.data_dir)
    except (FileNotFoundError, RuntimeError) as error:
        print(f'error: {error}', file=err)
        return EXIT_ERROR
    try:
        if args.command == 'run':
            _enable_progress_log()
            return run_batch(args, components, confirm_fn=confirm_fn, out=out)
        _ask_command(args, components, input_fn, out, checkpoint_path)
    except (EOFError, KeyboardInterrupt):
        print('aborted', file=err)
        return EXIT_ABORTED
    except (
            ValueError, OSError, RuntimeError, openai.OpenAIError,
    ) as error:
        print(f'error: {type(error).__name__}: {error}', file=err)
        return EXIT_ERROR
    return 0


if __name__ == '__main__':
    sys.exit(main())
