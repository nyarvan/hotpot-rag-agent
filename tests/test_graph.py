"""Behaviour tests for the hop-limited agent graph, run on fakes."""
import copy
import logging
from collections.abc import Sequence

import pytest
from rag_eval.data import Document
from rag_eval.generate import format_context

from agent.graph import build_graph
from agent.nodes import Decision, make_retrieve
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
