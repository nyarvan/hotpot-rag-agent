"""Behaviour tests for the `ask` command, run on fakes (no network)."""
import io
import re
import sqlite3
import stat
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import openai
import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from agent.cli import main
from agent.graph import SEARCH_K
from agent.nodes import Decision
from rag_eval.data import Document
from tests.fakes import DOCUMENTS, FakeAnswerer, FakeDecider, FakeSearcher

QUESTION = 'Who directed the film?'
RANKING = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']
BRIDGE_RANKING = ['B1', 'B2', 'B3', 'B4']
NEED_BRIDGE = Decision(enough=False, follow_up='Bridge Entity')
THREAD_ID = re.compile(r'thread_id: (\w+)')


def _components(
        *decisions: Decision,
) -> tuple[SimpleNamespace, FakeSearcher, FakeAnswerer]:
    decisions = decisions or (NEED_BRIDGE,)
    searcher = FakeSearcher({
        QUESTION: RANKING,
        'Bridge Entity': BRIDGE_RANKING,
        'Typed Query': BRIDGE_RANKING,
        'Second Bridge': BRIDGE_RANKING,
    })
    answerer = FakeAnswerer('Nolan')
    components = SimpleNamespace(
        documents=DOCUMENTS, searcher=searcher,
        decider=FakeDecider(decisions), answerer=answerer,
        subset=SimpleNamespace(questions=[]),
    )
    return components, searcher, answerer


def _scripted(replies: Sequence[str]) -> Callable[[str], str]:
    iterator: Iterator[str] = iter(replies)
    return lambda prompt: next(iterator)


def _run_full(
        tmp_path: Path, input_fn: Callable[[str], str], components: Any,
        argv: Sequence[str] = ('ask', QUESTION),
) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(
        argv, components_factory=lambda data_dir: components,
        input_fn=input_fn, out=out, err=err,
        checkpoint_path=tmp_path / 'sub' / 'checkpoints.sqlite',
    )
    return code, out.getvalue(), err.getvalue()


def _run(
        tmp_path: Path, replies: Sequence[str], components: Any,
        argv: Sequence[str] = ('ask', QUESTION),
) -> tuple[int, str]:
    code, out, _ = _run_full(tmp_path, _scripted(replies), components, argv)
    return code, out


def test_ask_approve_prints_answer_and_queries(tmp_path: Path) -> None:
    # Arrange
    components, searcher, _ = _components()

    # Act
    code, output = _run(tmp_path, ['a'], components)

    # Assert
    assert code == 0
    assert 'Proposed follow-up: Bridge Entity' in output
    assert 'Context: A1, A2, A3, A4, A5' in output
    assert 'Nolan' in output
    assert f'queries: [{QUESTION!r}, {"Bridge Entity"!r}]' in output
    assert 'llm_calls: 2' in output
    assert [q for q, _ in searcher.calls] == [QUESTION, 'Bridge Entity']


def test_ask_edit_sends_typed_query_to_second_search(tmp_path: Path) -> None:
    # Arrange
    components, searcher, _ = _components()

    # Act
    code, _ = _run(tmp_path, ['e', 'Typed Query'], components)

    # Assert
    assert code == 0
    assert [q for q, _ in searcher.calls] == [QUESTION, 'Typed Query']


def test_ask_reject_answers_without_second_search(tmp_path: Path) -> None:
    # Arrange
    components, searcher, answerer = _components()

    # Act
    code, output = _run(tmp_path, ['r'], components)

    # Assert
    assert code == 0
    assert [q for q, _ in searcher.calls] == [QUESTION]
    assert len(answerer.received) == 1
    assert 'Nolan' in output


def test_ask_reprompts_on_unknown_choice(tmp_path: Path) -> None:
    # Arrange
    components, searcher, _ = _components()

    # Act
    code, output = _run(tmp_path, ['x', '', 'a'], components)

    # Assert
    assert code == 0
    assert output.count('Unknown choice') == 2
    assert [q for q, _ in searcher.calls] == [QUESTION, 'Bridge Entity']


def test_ask_without_interrupt_does_not_prompt(tmp_path: Path) -> None:
    # Arrange
    components, searcher, _ = _components(Decision(enough=True))

    # Act
    code, output = _run(tmp_path, [], components)

    # Assert
    assert code == 0
    assert len(searcher.calls) == 1
    assert 'Nolan' in output


def test_ask_uses_fresh_thread_id_per_run(tmp_path: Path) -> None:
    # Arrange / Act
    _, first = _run(tmp_path, ['a'], _components()[0])
    _, second = _run(tmp_path, ['a'], _components()[0])

    # Assert
    first_match, second_match = THREAD_ID.search(first), THREAD_ID.search(second)
    assert first_match is not None
    assert second_match is not None
    assert first_match.group(1) != second_match.group(1)
    assert f'queries: [{QUESTION!r}, {"Bridge Entity"!r}]' in second


def test_ask_passes_max_hops_and_data_dir(tmp_path: Path) -> None:
    # Arrange
    components, searcher, _ = _components()
    seen: list[Path] = []

    def factory(data_dir: Path) -> Any:
        seen.append(data_dir)
        return components

    # Act
    code = main(
        ['ask', QUESTION, '--max-hops', '1', '--data-dir', str(tmp_path)],
        components_factory=factory, input_fn=_scripted([]),
        out=io.StringIO(), checkpoint_path=tmp_path / 'c.sqlite',
    )

    # Assert
    assert code == 0
    assert seen == [tmp_path]
    assert len(searcher.calls) == 1


@pytest.mark.parametrize('error', [
    RuntimeError('OPENAI_API_KEY is not set; load .env first'),
    FileNotFoundError('Missing cache: /x'),
])
def test_ask_reports_factory_error_in_one_line(
        tmp_path: Path, error: Exception,
) -> None:
    # Arrange
    def factory(data_dir: Path) -> Any:
        raise error

    out, err = io.StringIO(), io.StringIO()

    # Act
    code = main(
        ['ask', QUESTION], components_factory=factory,
        input_fn=_scripted([]), out=out, err=err,
        checkpoint_path=tmp_path / 'c.sqlite',
    )

    # Assert
    assert code == 1
    assert err.getvalue() == f'error: {error}\n'
    assert out.getvalue() == ''


@pytest.mark.parametrize('argv', [
    ['ask', QUESTION, '--max-hops', '0'],
    ['ask', QUESTION, '--max-hops', '6'],
    ['ask', QUESTION, '--max-hops', 'x'],
    ['ask', '   '],
    ['ask', 'q' * 1001],
])
def test_invalid_arguments_exit_2_without_building_components(
        argv: list[str],
) -> None:
    # Arrange
    def factory(data_dir: Path) -> Any:
        raise AssertionError('factory must not be called')

    # Act
    with pytest.raises(SystemExit) as exit_info:
        main(argv, components_factory=factory, err=io.StringIO())

    # Assert
    assert exit_info.value.code == 2


def test_question_is_normalised_before_search(tmp_path: Path) -> None:
    # Arrange
    components, searcher, _ = _components(Decision(enough=True))

    # Act
    code, _ = _run(tmp_path, [], components, ['ask', f'  Who  directed\n the film?\x1b '])

    # Assert
    assert code == 0
    assert [q for q, _ in searcher.calls] == [QUESTION]


def test_question_of_max_length_is_accepted(tmp_path: Path) -> None:
    # Arrange
    long_question = 'q' * 1000
    components, searcher, _ = _components(Decision(enough=True))
    searcher._rankings_by_query[long_question] = RANKING

    # Act
    code, _ = _run(tmp_path, [], components, ['ask', long_question])

    # Assert
    assert code == 0
    assert [q for q, _ in searcher.calls] == [long_question]


class RecordingSearcher(FakeSearcher):
    """FakeSearcher that also records prefetch calls."""

    def __init__(self, rankings: dict[str, list[str]]) -> None:
        super().__init__(rankings)
        self.prefetched: list[tuple[list[Any], int]] = []

    def prefetch(self, questions: Sequence[Any], k: int) -> None:
        """Record the prefetch call."""
        self.prefetched.append((list(questions), k))


def test_known_question_triggers_batch_prefetch(tmp_path: Path) -> None:
    # Arrange
    components, _, _ = _components(Decision(enough=True))
    known = [SimpleNamespace(question=QUESTION), SimpleNamespace(question='Other?')]
    components.subset = SimpleNamespace(questions=known)
    components.searcher = RecordingSearcher({QUESTION: RANKING})

    # Act
    code, _ = _run(tmp_path, [], components)

    # Assert
    assert code == 0
    assert components.searcher.prefetched == [(known, SEARCH_K)]


def test_unknown_question_skips_prefetch(tmp_path: Path) -> None:
    # Arrange
    components, _, _ = _components(Decision(enough=True))
    components.subset = SimpleNamespace(
        questions=[SimpleNamespace(question='Other?')],
    )
    components.searcher = RecordingSearcher({QUESTION: RANKING})

    # Act
    _run(tmp_path, [], components)

    # Assert
    assert components.searcher.prefetched == []


def test_two_interrupts_in_a_row(tmp_path: Path) -> None:
    # Arrange
    components, searcher, _ = _components(
        NEED_BRIDGE, Decision(enough=False, follow_up='Second Bridge'),
    )

    # Act
    code, output = _run(
        tmp_path, ['a', 'a'], components,
        ['ask', QUESTION, '--max-hops', '3'],
    )

    # Assert
    assert code == 0
    assert output.count('Proposed follow-up:') == 2
    assert [q for q, _ in searcher.calls] == [
        QUESTION, 'Bridge Entity', 'Second Bridge',
    ]
    assert 'note:' not in output


def test_blank_edit_reprompts_with_hint(tmp_path: Path) -> None:
    # Arrange
    components, searcher, _ = _components()

    # Act
    code, output = _run(tmp_path, ['e', '   ', 'e', 'Typed Query'], components)

    # Assert
    assert code == 0
    assert 'Empty query' in output
    assert [q for q, _ in searcher.calls] == [QUESTION, 'Typed Query']


def test_rejected_repeat_edit_prints_note(tmp_path: Path) -> None:
    # Arrange
    components, searcher, _ = _components()

    # Act
    code, output = _run(tmp_path, ['e', QUESTION], components)

    # Assert
    assert code == 0
    assert 'note: follow-up rejected, answered from hop 1' in output
    assert len(searcher.calls) == 1


def test_reject_prints_no_note(tmp_path: Path) -> None:
    # Arrange
    components, _, _ = _components()

    # Act
    _, output = _run(tmp_path, ['r'], components)

    # Assert
    assert 'note:' not in output


def test_eof_aborts_with_code_130(tmp_path: Path) -> None:
    # Arrange
    def eof(prompt: str) -> str:
        raise EOFError

    # Act
    code, out, err = _run_full(tmp_path, eof, _components()[0])

    # Assert
    assert code == 130
    assert err == 'aborted\n'
    assert 'answer:' not in out


def test_keyboard_interrupt_aborts_with_code_130(tmp_path: Path) -> None:
    # Arrange
    def interrupted(prompt: str) -> str:
        raise KeyboardInterrupt

    # Act
    code, _, err = _run_full(tmp_path, interrupted, _components()[0])

    # Assert
    assert code == 130
    assert err == 'aborted\n'


@pytest.mark.parametrize('error', [
    ValueError('bad value'), OSError('disk gone'),
    openai.OpenAIError('api down'),
])
def test_runtime_errors_become_one_line(
        tmp_path: Path, error: Exception,
) -> None:
    # Arrange
    def failing(prompt: str) -> str:
        raise error

    # Act
    code, out, err = _run_full(tmp_path, failing, _components()[0])

    # Assert
    assert code == 1
    assert err == f'error: {type(error).__name__}: {error}\n'
    assert 'answer:' not in out


def test_terminal_escapes_are_replaced_in_output(tmp_path: Path) -> None:
    # Arrange
    components, _, _ = _components()
    evil = 'A1\x1b[31m'
    components.documents = {
        **DOCUMENTS, evil: Document(doc_id='id-evil', title=evil, text='t'),
    }
    components.searcher = FakeSearcher({
        QUESTION: [evil, *RANKING[1:]], 'Bridge Entity': BRIDGE_RANKING,
    })
    components.answerer = FakeAnswerer('Nol\x1b[2Jan')

    # Act
    _, output = _run(tmp_path, ['a'], components)

    # Assert
    assert '\x1b' not in output
    assert 'A1\ufffd[31m' in output
    assert 'Nol\ufffd[2Jan' in output


def test_checkpoint_file_is_private_and_thread_deleted(
        tmp_path: Path,
) -> None:
    # Arrange
    components, _, _ = _components()
    path = tmp_path / 'sub' / 'checkpoints.sqlite'

    # Act
    code, output = _run(tmp_path, ['a'], components)

    # Assert
    assert code == 0
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    match = THREAD_ID.search(output)
    assert match is not None
    connection = sqlite3.connect(path)
    try:
        checkpoints = list(SqliteSaver(connection).list(
            {'configurable': {'thread_id': match.group(1)}},
        ))
    finally:
        connection.close()
    assert checkpoints == []
