"""Behaviour tests for the hop-limited agent graph, run on fakes."""
import copy
import logging
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from rag_eval.data import Document
from rag_eval.generate import format_context

from agent.graph import build_graph
from agent.decide import MAX_FOLLOW_UP_CHARS
from agent.nodes import Decision, make_retrieve, review_query
from tests.fakes import (
    DOCUMENTS,
    FakeAnswerer,
    FakeDecider,
    FakeSearcher,
    RaisingDecider,
)

QUESTION = 'Who directed the film?'
RANKING = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']
ENOUGH = Decision(enough=True)
NEED_BRIDGE = Decision(enough=False, follow_up='Bridge Entity')


def _titles(documents: Sequence[Document]) -> list[str]:
    return [document.title for document in documents]


def test_answers_after_one_hop_when_decider_says_enough() -> None:
    # Arrange
    searcher = FakeSearcher({QUESTION: RANKING})
    decider = FakeDecider([ENOUGH])
    graph = build_graph(searcher, decider, FakeAnswerer('Nolan'), DOCUMENTS)

    # Act
    result = graph.invoke({'question': QUESTION})

    # Assert
    assert [q for q, _ in searcher.calls] == [QUESTION]
    assert result['queries'] == [QUESTION]
    assert result['hops'] == 1
    assert len(decider.seen_contexts) == 1
    assert result['answer'] == 'Nolan'
    assert result['llm_calls'] == 2


def test_second_hop_searches_the_follow_up_query() -> None:
    # Arrange
    searcher = FakeSearcher({QUESTION: RANKING, 'Bridge Entity': ['B1', 'B2']})
    decider = FakeDecider([NEED_BRIDGE, ENOUGH])
    graph = build_graph(searcher, decider, FakeAnswerer(), DOCUMENTS)

    # Act
    result = graph.invoke({'question': QUESTION})

    # Assert
    assert searcher.calls == [(QUESTION, 10), ('Bridge Entity', 10)]
    assert result['queries'] == [QUESTION, 'Bridge Entity']
    assert result['hops'] == 2
    assert len(decider.seen_contexts) == 1
    assert result['llm_calls'] == 2


@pytest.mark.parametrize('max_hops', [2, 3])
def test_hop_limit_forces_generate_when_decider_never_satisfied(
        max_hops: int,
) -> None:
    # Arrange
    follow_ups = [f'Bridge {hop}' for hop in range(1, max_hops)]
    searcher = FakeSearcher(
        {QUESTION: RANKING} | {query: ['B1', 'B2'] for query in follow_ups},
    )
    decider = FakeDecider(
        [Decision(enough=False, follow_up=query) for query in follow_ups],
    )
    graph = build_graph(
        searcher, decider, FakeAnswerer('done'), DOCUMENTS, max_hops=max_hops,
    )

    # Act
    result = graph.invoke({'question': QUESTION})

    # Assert
    assert len(searcher.calls) == max_hops
    assert len(decider.seen_contexts) == max_hops - 1
    assert result['answer'] == 'done'
    assert result['hops'] == len(result['queries'])


def test_max_hops_one_never_calls_decide() -> None:
    # Arrange
    searcher = FakeSearcher({QUESTION: RANKING})
    graph = build_graph(
        searcher, RaisingDecider(), FakeAnswerer(), DOCUMENTS, max_hops=1,
    )

    # Act
    result = graph.invoke({'question': QUESTION})

    # Assert
    assert len(searcher.calls) == 1
    assert result['llm_calls'] == 1


def test_one_hop_context_matches_baseline_top_k() -> None:
    # Arrange
    searcher = FakeSearcher({QUESTION: RANKING})
    answerer = FakeAnswerer()
    graph = build_graph(searcher, FakeDecider([ENOUGH]), answerer, DOCUMENTS)

    # Act
    result = graph.invoke({'question': QUESTION})

    # Assert
    expected = RANKING[:5]
    assert _titles(answerer.received[0]) == expected
    assert result['titles'] == expected
    assert result['contexts'] == [
        format_context(DOCUMENTS[title]) for title in expected
    ]
    assert searcher.calls == [(QUESTION, 10)]


def test_blank_follow_up_is_treated_as_enough() -> None:
    # Arrange
    searcher = FakeSearcher({QUESTION: RANKING})
    decider = FakeDecider([Decision(enough=False, follow_up='  ')])
    graph = build_graph(searcher, decider, FakeAnswerer(), DOCUMENTS)

    # Act
    result = graph.invoke({'question': QUESTION})

    # Assert
    assert len(searcher.calls) == 1
    assert result['hops'] == 1
    assert result['answer'] == 'fake answer'


def test_enough_decision_ignores_its_follow_up() -> None:
    # Arrange
    searcher = FakeSearcher({QUESTION: RANKING})
    decider = FakeDecider([Decision(enough=True, follow_up='Bridge Entity')])
    graph = build_graph(searcher, decider, FakeAnswerer(), DOCUMENTS)

    # Act
    result = graph.invoke({'question': QUESTION})

    # Assert
    assert len(searcher.calls) == 1
    assert result['follow_up'] == ''


def test_repeated_query_is_treated_as_enough_and_logged(
        caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    searcher = FakeSearcher({QUESTION: RANKING})
    decider = FakeDecider([Decision(enough=False, follow_up=QUESTION)])
    graph = build_graph(searcher, decider, FakeAnswerer(), DOCUMENTS)

    # Act
    with caplog.at_level(logging.INFO, logger='agent.nodes'):
        result = graph.invoke({'question': QUESTION})

    # Assert
    assert len(searcher.calls) == 1
    assert result['hops'] == 1
    assert result['llm_calls'] == 2  # a rejected repeat still cost a decide
    assert 'repeats an earlier query' in caplog.text


def test_follow_up_is_stripped_before_searching() -> None:
    # Arrange
    searcher = FakeSearcher({QUESTION: RANKING, 'Bridge Entity': ['B1']})
    decider = FakeDecider(
        [Decision(enough=False, follow_up='  Bridge Entity\n')],
    )
    graph = build_graph(searcher, decider, FakeAnswerer(), DOCUMENTS)

    # Act
    graph.invoke({'question': QUESTION})

    # Assert
    assert searcher.calls[1][0] == 'Bridge Entity'


@pytest.mark.parametrize('max_hops', [0, -1])
def test_build_graph_rejects_non_positive_max_hops(max_hops: int) -> None:
    with pytest.raises(ValueError, match='max_hops'):
        build_graph(
            FakeSearcher({}), FakeDecider([ENOUGH]), FakeAnswerer(), DOCUMENTS,
            max_hops=max_hops,
        )


def test_build_graph_rejects_context_k_above_search_k() -> None:
    with pytest.raises(
            ValueError,
            match=r'context_k \(6\) must not exceed search_k \(5\)',
    ):
        build_graph(
            FakeSearcher({}), FakeDecider([ENOUGH]), FakeAnswerer(), DOCUMENTS,
            search_k=5, context_k=6,
        )


@pytest.mark.parametrize('context_k', [2, 0, -1])
def test_build_graph_rejects_context_k_below_first_hop_share(
        context_k: int,
) -> None:
    with pytest.raises(ValueError, match=rf'context_k \({context_k}\)'):
        build_graph(
            FakeSearcher({}), FakeDecider([ENOUGH]), FakeAnswerer(), DOCUMENTS,
            context_k=context_k,
        )


def test_retrieve_node_returns_partial_update_without_mutating_state() -> None:
    # Arrange
    node = make_retrieve(FakeSearcher({QUESTION: RANKING}), search_k=10)
    state = {
        'question': QUESTION, 'queries': [], 'rankings': [], 'hops': 0,
        'follow_up': '',
    }
    before = copy.deepcopy(state)

    # Act
    update = node(state)

    # Assert
    assert set(update) == {'queries', 'rankings', 'hops'}
    assert update == {'queries': [QUESTION], 'rankings': [RANKING], 'hops': 1}
    assert state == before


def test_unknown_title_raises_key_error_in_generate() -> None:
    # Arrange
    searcher = FakeSearcher({QUESTION: ['A1', 'missing']})
    graph = build_graph(
        searcher, FakeDecider([ENOUGH]), FakeAnswerer(), DOCUMENTS,
    )

    # Act / Assert
    with pytest.raises(KeyError):
        graph.invoke({'question': QUESTION})


def test_two_hop_answer_uses_merged_context() -> None:
    # Arrange
    searcher = FakeSearcher({
        QUESTION: RANKING,
        'Bridge Entity': ['A1', 'B1', 'B2', 'B3'],
    })
    answerer = FakeAnswerer()
    graph = build_graph(
        searcher, FakeDecider([NEED_BRIDGE]), answerer, DOCUMENTS,
    )

    # Act
    result = graph.invoke({'question': QUESTION})

    # Assert
    assert _titles(answerer.received[0]) == ['A1', 'A2', 'A3', 'B1', 'B2']
    assert result['titles'] == ['A1', 'A2', 'A3', 'B1', 'B2']


def test_decider_sees_the_current_merged_context() -> None:
    # Arrange
    searcher = FakeSearcher({QUESTION: RANKING, 'Bridge Entity': ['B1', 'B2']})
    decider = FakeDecider([NEED_BRIDGE, ENOUGH])
    graph = build_graph(
        searcher, decider, FakeAnswerer(), DOCUMENTS, max_hops=3,
    )

    # Act
    graph.invoke({'question': QUESTION})

    # Assert
    assert decider.seen_contexts == [
        ['A1', 'A2', 'A3', 'A4', 'A5'],
        ['A1', 'A2', 'A3', 'B1', 'B2'],
    ]


# Human review (day 3)

BRIDGE_RANKING = ['B1', 'B2', 'B3']


def _review_graph(
        decider: FakeDecider | None = None,
        *,
        extra: dict[str, Sequence[str]] | None = None,
        checkpointer: Any = None,
) -> tuple[Any, FakeSearcher, FakeDecider, FakeAnswerer]:
    searcher = FakeSearcher(
        {QUESTION: RANKING, 'Bridge Entity': BRIDGE_RANKING} | (extra or {}),
    )
    decider = decider or FakeDecider([NEED_BRIDGE])
    answerer = FakeAnswerer()
    graph = build_graph(
        searcher, decider, answerer, DOCUMENTS, human_review=True,
        checkpointer=checkpointer or InMemorySaver(),
    )
    return graph, searcher, decider, answerer


def _config(thread_id: str = 't1') -> dict[str, Any]:
    return {'configurable': {'thread_id': thread_id}}


def test_human_review_without_checkpointer_raises() -> None:
    with pytest.raises(ValueError, match='checkpointer'):
        build_graph(
            FakeSearcher({}), FakeDecider([ENOUGH]), FakeAnswerer(), DOCUMENTS,
            human_review=True,
        )


def test_interrupts_before_second_search_with_follow_up_and_titles() -> None:
    # Arrange
    graph, searcher, _, _ = _review_graph()

    # Act
    result = graph.invoke({'question': QUESTION}, _config())

    # Assert
    assert searcher.calls == [(QUESTION, 10)]
    payload = result['__interrupt__'][0].value
    assert payload == {
        'question': QUESTION,
        'follow_up': 'Bridge Entity',
        'titles': ['A1', 'A2', 'A3', 'A4', 'A5'],
    }
    assert 'answer' not in result


def test_resume_with_edited_query_searches_exactly_it() -> None:
    # Arrange
    graph, searcher, _, _ = _review_graph(
        extra={'Edited Entity': ['B4', 'B1']},
    )
    graph.invoke({'question': QUESTION}, _config())

    # Act
    result = graph.invoke(Command(resume='Edited Entity'), _config())

    # Assert
    assert searcher.calls[1] == ('Edited Entity', 10)
    assert len(searcher.calls) == 2
    assert result['queries'][-1] == 'Edited Entity'
    assert result['answer'] == 'fake answer'


def test_resume_with_proposed_follow_up_searches_it() -> None:
    # Arrange
    graph, searcher, _, _ = _review_graph()
    graph.invoke({'question': QUESTION}, _config())

    # Act
    result = graph.invoke(Command(resume='Bridge Entity'), _config())

    # Assert
    assert searcher.calls[1] == ('Bridge Entity', 10)
    assert result['hops'] == 2


def test_resume_with_empty_string_answers_from_first_hop() -> None:
    # Arrange
    graph, searcher, _, answerer = _review_graph()
    graph.invoke({'question': QUESTION}, _config())

    # Act
    result = graph.invoke(Command(resume=''), _config())

    # Assert
    assert searcher.calls == [(QUESTION, 10)]
    assert result['hops'] == 1
    assert _titles(answerer.received[0]) == RANKING[:5]


def test_resume_does_not_call_decider_again() -> None:
    # Arrange
    graph, _, decider, _ = _review_graph()
    graph.invoke({'question': QUESTION}, _config())

    # Act
    graph.invoke(Command(resume='Bridge Entity'), _config())

    # Assert
    assert len(decider.seen_contexts) == 1


def test_no_interrupt_when_decider_says_enough() -> None:
    # Arrange
    graph, searcher, _, _ = _review_graph(FakeDecider([ENOUGH]))

    # Act
    result = graph.invoke({'question': QUESTION}, _config())

    # Assert
    assert '__interrupt__' not in result
    assert result['answer'] == 'fake answer'
    assert len(searcher.calls) == 1


def test_edited_query_repeating_earlier_query_is_rejected() -> None:
    # Arrange
    graph, searcher, _, _ = _review_graph()
    graph.invoke({'question': QUESTION}, _config())

    # Act
    result = graph.invoke(Command(resume=QUESTION), _config())

    # Assert
    assert searcher.calls == [(QUESTION, 10)]
    assert result['hops'] == 1
    assert result['answer'] == 'fake answer'


def test_non_string_resume_raises_type_error() -> None:
    # Arrange
    graph, _, _, _ = _review_graph()
    graph.invoke({'question': QUESTION}, _config())

    # Act / Assert
    with pytest.raises(TypeError):
        graph.invoke(Command(resume=5), _config())


@pytest.mark.parametrize(('reply', 'expected'), [
    ('  Edited \n Entity\t', 'Edited Entity'),
    ('Edited\x00Entity', 'EditedEntity'),
    ('x' * (MAX_FOLLOW_UP_CHARS + 50), 'x' * MAX_FOLLOW_UP_CHARS),
    ('   ', ''),
    ('', ''),
])
def test_review_query_cleans_reply(reply: str, expected: str) -> None:
    assert review_query(reply, [QUESTION]) == expected


@pytest.mark.parametrize('reply', [None, 5, ['a'], b'a'])
def test_review_query_rejects_non_string(reply: object) -> None:
    with pytest.raises(TypeError):
        review_query(reply, [])  # type: ignore[arg-type]


def test_review_query_logs_repeated_query(
        caplog: pytest.LogCaptureFixture,
) -> None:
    # Act
    with caplog.at_level(logging.INFO, logger='agent.nodes'):
        result = review_query(f' {QUESTION} ', [QUESTION])

    # Assert
    assert result == ''
    assert 'repeats an earlier query' in caplog.text


def test_resume_after_reopening_sqlite_checkpoint(tmp_path: Path) -> None:
    # Arrange
    db_path = tmp_path / 'checkpoints.sqlite'
    first_conn = sqlite3.connect(db_path, check_same_thread=False)
    graph, _, _, _ = _review_graph(checkpointer=SqliteSaver(first_conn))
    graph.invoke({'question': QUESTION}, _config('durable'))
    first_conn.close()

    # Act
    second_conn = sqlite3.connect(db_path, check_same_thread=False)
    try:
        graph2, searcher2, _, _ = _review_graph(
            checkpointer=SqliteSaver(second_conn),
            extra={'Edited Entity': ['B4']},
        )
        result = graph2.invoke(
            Command(resume='Edited Entity'), _config('durable'),
        )
    finally:
        second_conn.close()

    # Assert
    assert searcher2.calls == [('Edited Entity', 10)]
    assert result['queries'] == [QUESTION, 'Edited Entity']
    assert result['answer'] == 'fake answer'
