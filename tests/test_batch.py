"""Tests for the `run` command and its cost estimate, on fakes (no network)."""
import io
import json
import logging
import os
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from rag_eval import cli as harness_cli
from rag_eval.cost import estimate_generation, format_estimate
from rag_eval.data import DEFAULT_N, Question
from rag_eval.generate import DEFAULT_MAX_TOKENS as GENERATE_MAX_TOKENS
from rag_eval.generate import SYSTEM_PROMPT as GENERATE_PROMPT
from rag_eval.records import read_records

from agent import decide, llm
from agent import batch
from agent.batch import (
    BASELINE_MODEL,
    EXIT_CANCELLED,
    estimate_run,
    summarize,
)
from agent.cli import EXIT_ERROR, main
from agent.components import RerankSearcher
from agent.graph import build_graph
from agent.nodes import CONTEXT_K, Decision
from agent.run import run_agent
from tests.fakes import DOCUMENTS, FakeAnswerer, FakeDecider
from tests.test_run import OrderedRetriever, make_question

MODEL = 'gpt-4o-mini'
NEED_BRIDGE = Decision(enough=False, follow_up='Bridge Entity')
DOCUMENTS_BY_ID = {d.doc_id: d for d in DOCUMENTS.values()}
AVG_CHARS = 100.0
BRIDGE_ONE = Decision(enough=False, follow_up='Bridge Entity')
BRIDGE_TWO = Decision(enough=False, follow_up='Second Bridge')
ENOUGH = Decision(enough=True)
AGENT_LOGGER = 'agent'


@pytest.fixture(autouse=True)
def _restore_agent_logger() -> Any:
    logger = logging.getLogger(AGENT_LOGGER)
    handlers, level = list(logger.handlers), logger.level
    yield
    logger.handlers[:] = handlers
    logger.setLevel(level)


class RecordingRetriever(OrderedRetriever):
    """Ranks like `OrderedRetriever` and records the batched calls."""

    def __init__(self) -> None:
        self.many_calls: list[list[Question]] = []

    def retrieve_many(
            self, questions: Sequence[Question], k: int,
    ) -> list[list[str]]:
        """Record the batch, then rank."""
        self.many_calls.append(list(questions))
        return super().retrieve_many(questions, k)


class FakeConfirm:
    """A confirm function with a canned reply that records its calls."""

    def __init__(self, reply: bool) -> None:
        self._reply = reply
        self.calls: list[tuple[str, bool]] = []

    def __call__(self, question: str, *, assume_yes: bool = False) -> bool:
        """Record the call and return the canned reply."""
        self.calls.append((question, assume_yes))
        return self._reply


def _components(
        n_questions: int = 4,
) -> tuple[SimpleNamespace, RecordingRetriever, FakeAnswerer]:
    retriever = RecordingRetriever()
    answerer = FakeAnswerer('fake')
    components = SimpleNamespace(
        subset=SimpleNamespace(
            questions=[make_question(i) for i in range(n_questions)],
        ),
        documents=DOCUMENTS_BY_ID, searcher=RerankSearcher(retriever),
        decider=FakeDecider([NEED_BRIDGE]), answerer=answerer,
    )
    return components, retriever, answerer


def _run(
        components: Any, argv: Sequence[str], confirm_fn: FakeConfirm,
) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(
        argv, components_factory=lambda data_dir: components,
        confirm_fn=confirm_fn, out=out, err=err,
    )
    return code, out.getvalue(), err.getvalue()


def test_run_writes_one_agent_record_per_question(tmp_path: Path) -> None:
    # Arrange
    components, _, _ = _components(4)
    path = tmp_path / 'runs' / 'agent.jsonl'

    # Act
    code, _, _ = _run(
        components, ['run', '--out', str(path), '--n', '3'],
        FakeConfirm(True),
    )

    # Assert
    assert code == 0
    records = read_records(path)
    assert [r.question_id for r in records] == ['q0', 'q1', 'q2']
    raw = [json.loads(line) for line in path.read_text().splitlines()]
    assert all({'hops', 'queries', 'llm_calls'} <= set(r) for r in raw)
    assert raw[0]['hops'] == 2
    assert raw[0]['queries'][1] == 'Bridge Entity'


def test_run_prints_estimate_and_asks_once(tmp_path: Path) -> None:
    components, _, _ = _components()
    confirm = FakeConfirm(True)

    _, out, _ = _run(
        components, ['run', '--out', str(tmp_path / 'a.jsonl'), '--n', '4'], confirm,
    )

    assert confirm.calls == [('Run it?', False)]
    assert 'about 8 requests' in out  # 4 decide + 4 generate


def test_run_not_confirmed_writes_nothing_and_runs_nothing(
        tmp_path: Path,
) -> None:
    components, retriever, answerer = _components()
    path = tmp_path / 'a.jsonl'

    code, out, _ = _run(
        components, ['run', '--out', str(path), '--n', '4'], FakeConfirm(False),
    )

    assert code == EXIT_CANCELLED == 3
    assert 'Cancelled: nothing was run. Pass --yes' in out
    assert not path.exists()
    assert retriever.many_calls == []
    assert answerer.received == []


def test_run_yes_flag_is_passed_to_confirm(tmp_path: Path) -> None:
    components, _, _ = _components()
    confirm = FakeConfirm(True)

    _run(
        components, ['run', '--out', str(tmp_path / 'a.jsonl'), '--n', '4', '--yes'],
        confirm,
    )

    assert confirm.calls == [('Run it?', True)]


def test_run_single_hop_does_not_ask(tmp_path: Path) -> None:
    components, _, _ = _components()
    confirm = FakeConfirm(False)
    path = tmp_path / 'a.jsonl'

    code, _, _ = _run(
        components, ['run', '--out', str(path), '--n', '4', '--max-hops', '1'],
        confirm,
    )

    assert code == 0
    assert confirm.calls == []
    assert {r.question_id for r in read_records(path)} == {
        'q0', 'q1', 'q2', 'q3',
    }


def test_run_prefetches_default_n_questions_regardless_of_n(
        tmp_path: Path,
) -> None:
    components, retriever, _ = _components(DEFAULT_N + 5)

    code, _, _ = _run(
        components,
        ['run', '--out', str(tmp_path / 'a.jsonl'), '--n', '3', '--yes',
         '--max-hops', '1'],
        FakeConfirm(True),
    )

    assert code == 0
    assert len(retriever.many_calls) == 1
    assert retriever.many_calls[0] == components.subset.questions[:DEFAULT_N]


def test_run_missing_cache_is_one_line_error(tmp_path: Path) -> None:
    def factory(data_dir: Path) -> Any:
        raise FileNotFoundError('Missing cache: x')

    out, err = io.StringIO(), io.StringIO()
    code = main(
        ['run', '--out', str(tmp_path / 'a.jsonl')],
        components_factory=factory, confirm_fn=FakeConfirm(True),
        out=out, err=err,
    )

    assert code == EXIT_ERROR
    assert err.getvalue() == 'error: Missing cache: x\n'


def test_run_n_zero_is_rejected_by_parser(tmp_path: Path) -> None:
    components, _, _ = _components()

    with pytest.raises(SystemExit) as exc:
        _run(
            components, ['run', '--out', str(tmp_path / 'a.jsonl'), '--n', '0'],
            FakeConfirm(True),
        )

    assert exc.value.code == 2


def test_run_n_above_subset_size_is_an_error(tmp_path: Path) -> None:
    components, _, _ = _components(4)
    confirm = FakeConfirm(True)
    path = tmp_path / 'a.jsonl'

    code, _, err = _run(
        components, ['run', '--out', str(path), '--n', '5'], confirm,
    )

    assert code == EXIT_ERROR
    assert err.count('\n') == 1 and 'Traceback' not in err
    assert confirm.calls == []
    assert not path.exists()


def test_run_unwritable_out_is_one_line_error(tmp_path: Path) -> None:
    components, _, _ = _components()
    blocker = tmp_path / 'file'
    blocker.write_text('x')

    confirm = FakeConfirm(True)

    code, _, err = _run(
        components,
        ['run', '--out', str(blocker / 'a.jsonl'), '--n', '4', '--yes'],
        confirm,
    )

    assert code == EXIT_ERROR
    assert err.startswith('error: ') and err.count('\n') == 1
    assert confirm.calls == []


def test_estimate_run_scales_with_hops_and_questions() -> None:
    def generation(n: int) -> Any:
        return estimate_generation(
            MODEL, n_questions=n, context_k=CONTEXT_K,
            avg_paragraph_chars=AVG_CHARS, system_chars=len(GENERATE_PROMPT),
            max_output_tokens=GENERATE_MAX_TOKENS,
        )

    def deciding(n: int) -> Any:
        return estimate_generation(
            MODEL, n_questions=n, context_k=CONTEXT_K,
            avg_paragraph_chars=AVG_CHARS,
            system_chars=len(decide.SYSTEM_PROMPT),
            max_output_tokens=llm.DEFAULT_MAX_TOKENS,
        )

    single = estimate_run(
        MODEL, n_questions=10, decide_calls=0, avg_paragraph_chars=AVG_CHARS,
    )
    triple = estimate_run(
        MODEL, n_questions=10, decide_calls=20, avg_paragraph_chars=AVG_CHARS,
    )

    assert single.calls == 10
    assert triple.calls == 30
    assert single.input_tokens == generation(10).input_tokens
    assert triple.input_tokens == (
        generation(10).input_tokens + deciding(20).input_tokens
    )
    assert triple.output_tokens == (
        generation(10).output_tokens + deciding(20).output_tokens
    )
    assert triple.usd == pytest.approx(
        generation(10).usd + deciding(20).usd,
    )


def test_estimate_run_unknown_model_has_no_price() -> None:
    estimate = estimate_run(
        'mystery', n_questions=1, decide_calls=1, avg_paragraph_chars=1.0,
    )

    assert estimate.usd is None


def _mixed_records() -> list[Any]:
    """Four questions, max_hops=3: decisions 2, 1, 2 and 1 per question."""
    components, _, _ = _components(4)
    components.decider = FakeDecider(
        [BRIDGE_ONE, BRIDGE_TWO, ENOUGH, BRIDGE_ONE, ENOUGH, ENOUGH],
    )
    graph = build_graph(
        components.searcher, components.decider, components.answerer,
        DOCUMENTS_BY_ID, max_hops=3,
    )
    return run_agent(components.subset.questions, graph, components.searcher)


def test_summary_counts_actual_decide_calls() -> None:
    records = _mixed_records()
    assert [r.hops for r in records] == [3, 1, 2, 1]
    assert [r.llm_calls for r in records] == [3, 2, 3, 2]

    text = summarize(records, MODEL, AVG_CHARS)

    # 10 LLM calls = 6 decide + 4 generate; baseline = 4 generate
    assert 'Questions: 4' in text
    assert 'Multi-hop (hops > 1): 2 (50.0%)' in text
    assert 'LLM calls: 10' in text
    assert f'Agent with {MODEL}: about 10 requests' in text
    assert (
        f'Single-step baseline with {BASELINE_MODEL}: about 4 requests'
        in text
    )
    agent = estimate_run(
        MODEL, n_questions=4, decide_calls=6, avg_paragraph_chars=AVG_CHARS,
    )
    base = estimate_run(
        BASELINE_MODEL, n_questions=4, decide_calls=0,
        avg_paragraph_chars=AVG_CHARS,
    )
    assert f'Baseline / agent cost: {base.usd / agent.usd:.1f}x' in text


def test_summary_omits_ratio_for_unpriced_model() -> None:
    text = summarize(_mixed_records(), 'mystery', AVG_CHARS)

    assert 'Agent with mystery' in text
    assert 'Baseline / agent cost' not in text


def test_baseline_estimate_equals_harness_run_estimate(
        capsys: pytest.CaptureFixture[str],
) -> None:
    code = harness_cli.main(
        ['run', '--chat-model', BASELINE_MODEL, '--n', '200'],
        corpus_chars_fn=lambda: AVG_CHARS,
        confirm_fn=lambda question, *, assume_yes=False: False,
    )

    expected = format_estimate(
        f'Generation with {BASELINE_MODEL}',
        estimate_run(
            BASELINE_MODEL, n_questions=200, decide_calls=0,
            avg_paragraph_chars=AVG_CHARS,
        ),
    )
    assert code != 0
    assert expected in capsys.readouterr().out


def test_run_out_directory_fails_fast(tmp_path: Path) -> None:
    components, _, _ = _components()
    confirm = FakeConfirm(True)

    code, _, err = _run(
        components, ['run', '--out', str(tmp_path), '--n', '4'], confirm,
    )

    assert code == EXIT_ERROR
    assert err.startswith('error: ') and err.count('\n') == 1
    assert confirm.calls == []


def test_run_overwrites_existing_file(tmp_path: Path) -> None:
    components, _, _ = _components()
    path = tmp_path / 'a.jsonl'
    path.write_text('old\n')

    code, _, _ = _run(
        components, ['run', '--out', str(path), '--n', '4', '--yes'],
        FakeConfirm(True),
    )

    assert code == 0
    assert len(read_records(path)) == 4
    assert os.listdir(tmp_path) == ['a.jsonl']


def test_failed_write_keeps_old_file_and_leaves_no_temp(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    components, _, _ = _components()
    path = tmp_path / 'a.jsonl'
    path.write_text('old\n')

    def failing_write(target: Path, records: Any) -> None:
        Path(target).write_text('partial')
        raise OSError('disk full')

    monkeypatch.setattr(batch, 'write_records', failing_write)

    code, _, err = _run(
        components, ['run', '--out', str(path), '--n', '4', '--yes'],
        FakeConfirm(True),
    )

    assert code == EXIT_ERROR
    assert 'disk full' in err
    assert path.read_text() == 'old\n'
    assert os.listdir(tmp_path) == ['a.jsonl']


def test_run_runtime_error_is_one_line_error(tmp_path: Path) -> None:
    components, _, _ = _components()

    class BrokenDecider:
        def decide(self, question: str, contexts: Any) -> Decision:
            raise RuntimeError('boom')

    components.decider = BrokenDecider()

    code, _, err = _run(
        components, ['run', '--out', str(tmp_path / 'a.jsonl'), '--n', '4',
                     '--yes'],
        FakeConfirm(True),
    )

    assert code == EXIT_ERROR
    assert 'boom' in err and err.count('\n') == 1


def test_run_logging_uses_one_agent_handler_not_root(
        tmp_path: Path,
) -> None:
    root_handlers = list(logging.getLogger().handlers)
    argv = ['run', '--out', str(tmp_path / 'a.jsonl'), '--n', '4', '--yes']

    _run(_components()[0], argv, FakeConfirm(True))
    _run(_components()[0], argv, FakeConfirm(True))

    agent_logger = logging.getLogger(AGENT_LOGGER)
    streams = [
        h for h in agent_logger.handlers
        if type(h) is logging.StreamHandler
    ]
    assert len(streams) == 1
    assert agent_logger.level == logging.INFO
    assert logging.getLogger().handlers == root_handlers
    assert logging.getLogger('httpx').level == logging.NOTSET


def test_run_agent_logs_progress_every_twenty_questions(
        caplog: pytest.LogCaptureFixture,
) -> None:
    questions = [make_question(i) for i in range(45)]
    searcher = RerankSearcher(OrderedRetriever())
    graph = build_graph(
        searcher, FakeDecider([NEED_BRIDGE]), FakeAnswerer(),
        DOCUMENTS_BY_ID, max_hops=1,
    )

    with caplog.at_level(logging.INFO, logger='agent.run'):
        run_agent(questions, graph, searcher)

    assert [r.getMessage() for r in caplog.records] == [
        '20/45 questions', '40/45 questions',
    ]

