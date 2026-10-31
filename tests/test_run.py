"""Tests for `run_agent` and `AgentRecord`."""
from collections.abc import Sequence
from pathlib import Path

import pytest
from rag_eval.data import Document, Question
from rag_eval.generate import format_context
from rag_eval.records import RunRecord, read_records, write_records
from rag_eval.run import run_dense

from agent.components import RerankSearcher
from agent.graph import SEARCH_K, build_graph
from agent.nodes import Decision
from agent.run import AgentRecord, run_agent, to_record
from tests.fakes import (
    DOCUMENTS,
    TITLES,
    FakeDecider,
    FakeSearcher,
    RaisingDecider,
)

DOC_IDS = [DOCUMENTS[title].doc_id for title in TITLES]
DOCUMENTS_BY_ID = {d.doc_id: d for d in DOCUMENTS.values()}


def make_question(index: int) -> Question:
    return Question(
        question_id=f'q{index}', question=f'question {index}?',
        answer=f'ans {index}', type='bridge',
        gold_titles=('A1', 'B2'), evidence=(),
    )


class OrderedRetriever:
    """A deterministic ranking per question: rotated document ids."""

    def retrieve(self, question: Question, k: int) -> list[str]:
        """Rank the ids by a question-dependent rotation."""
        shift = sum(map(ord, question.question)) % len(DOC_IDS)
        return (DOC_IDS[shift:] + DOC_IDS[:shift])[:k]

    def retrieve_many(
            self, questions: Sequence[Question], k: int,
    ) -> list[list[str]]:
        """Rank each question."""
        return [self.retrieve(q, k) for q in questions]


class RecordingOrderedRetriever(OrderedRetriever):
    """`OrderedRetriever` that records the calls it receives."""

    def __init__(self) -> None:
        self.single_calls: list[tuple[Question, int]] = []
        self.many_calls: list[tuple[list[Question], int]] = []

    def retrieve(self, question: Question, k: int) -> list[str]:
        """Record, then rank."""
        self.single_calls.append((question, k))
        return super().retrieve(question, k)

    def retrieve_many(
            self, questions: Sequence[Question], k: int,
    ) -> list[list[str]]:
        """Record, then rank without counting per-question calls."""
        self.many_calls.append((list(questions), k))
        return [OrderedRetriever.retrieve(self, q, k) for q in questions]


class DerivedAnswerer:
    """Answer derived from the question and the context titles."""

    def answer(self, question: str, contexts: Sequence[Document]) -> str:
        """Join the question length and the titles."""
        titles = ','.join(d.title for d in contexts)
        return f'{len(question)}:{titles}'


class SpySearcher:
    """Records prefetch calls."""

    def __init__(self) -> None:
        self.prefetch_calls: list[tuple[list[Question], int]] = []

    def prefetch(self, questions: Sequence[Question], k: int) -> None:
        """Record the call."""
        self.prefetch_calls.append((list(questions), k))


def test_max_hops_one_records_equal_harness_run_dense() -> None:
    questions = [make_question(i) for i in range(3)]
    retriever = OrderedRetriever()
    answerer = DerivedAnswerer()
    searcher = RerankSearcher(retriever)
    graph = build_graph(
        searcher, RaisingDecider(), answerer, DOCUMENTS_BY_ID, max_hops=1,
    )

    baseline = run_dense(
        questions, list(DOCUMENTS.values()), retriever, answerer,
        retrieve_k=10, context_k=5,
    )
    records = run_agent(questions, graph, searcher)

    assert len(records) == len(baseline) == 3
    for record, expected in zip(records, baseline, strict=True):
        assert isinstance(record, AgentRecord)
        assert RunRecord.model_validate(record.model_dump()) == expected
        assert record.hops == 1
        assert record.llm_calls == 1


def test_agent_record_round_trips_through_harness_run_file(
        tmp_path: Path,
) -> None:
    record = AgentRecord(
        question_id='q', question='why?', contexts=('c',), answer='a',
        reference='r', retrieved=('A1',), gold_titles=('A1',),
        hops=2, queries=('why?', 'more'), llm_calls=3,
    )
    path = tmp_path / 'runs' / 'agent.jsonl'

    write_records(path, [record])
    loaded = read_records(path)

    assert loaded == [RunRecord.model_validate(record.model_dump())]


def test_two_hop_retrieved_is_context_then_rest_of_hop_one() -> None:
    question = make_question(0)
    hop_one = [f'A{i}' for i in range(1, 9)]
    state = {
        'queries': [question.question, 'next'],
        'rankings': [hop_one, ['B1', 'A2', 'B2']],
        'hops': 2,
        'titles': ['A1', 'A2', 'A3', 'B1', 'B2'],
        'contexts': ['c1', 'c2'],
        'answer': 'x',
        'llm_calls': 3,
    }

    record = to_record(question, state)

    assert record.retrieved == (
        'A1', 'A2', 'A3', 'B1', 'B2', 'A4', 'A5', 'A6', 'A7', 'A8',
    )
    assert record.queries == (question.question, 'next')
    assert record.contexts == ('c1', 'c2')
    assert (record.hops, record.llm_calls) == (2, 3)
    assert record.gold_titles == question.gold_titles
    assert record.reference == question.answer


def test_run_agent_two_hops_end_to_end() -> None:
    question = make_question(0)
    searcher = FakeSearcher({
        question.question: TITLES[:10], 'follow': ['B1', 'B2', 'B3'],
    })
    decider = FakeDecider([Decision(False, 'follow'), Decision(True)])
    graph = build_graph(searcher, decider, DerivedAnswerer(), DOCUMENTS)

    [record] = run_agent([question], graph, SpySearcher())

    assert record.hops == 2
    assert record.queries == (question.question, 'follow')
    assert record.llm_calls == 2
    assert record.contexts[0] == format_context(DOCUMENTS['A1'])
    assert len(set(record.retrieved)) == len(record.retrieved)


def test_run_agent_prefetches_once_with_all_questions_and_search_k() -> None:
    questions = [make_question(i) for i in range(2)]
    spy = SpySearcher()
    graph = build_graph(
        RerankSearcher(OrderedRetriever()), RaisingDecider(),
        DerivedAnswerer(), DOCUMENTS_BY_ID, max_hops=1,
    )

    run_agent(questions, graph, spy)

    assert spy.prefetch_calls == [(questions, SEARCH_K)]


def make_one_hop_graph():
    return build_graph(
        RerankSearcher(OrderedRetriever()), RaisingDecider(),
        DerivedAnswerer(), DOCUMENTS_BY_ID, max_hops=1,
    )


def test_run_agent_prefetches_superset_and_runs_subset() -> None:
    everything = [make_question(i) for i in range(4)]
    retriever = RecordingOrderedRetriever()
    searcher = RerankSearcher(retriever)
    graph = build_graph(
        searcher, RaisingDecider(), DerivedAnswerer(), DOCUMENTS_BY_ID,
        max_hops=1,
    )

    records = run_agent(
        everything[1:3], graph, searcher, prefetch_questions=everything,
    )

    assert retriever.many_calls == [(everything, SEARCH_K)]
    assert retriever.single_calls == []
    assert [r.question_id for r in records] == ['q1', 'q2']


def test_run_agent_forwards_search_k_to_prefetch() -> None:
    questions = [make_question(i) for i in range(2)]
    spy = SpySearcher()

    run_agent(questions[:0], make_one_hop_graph(), spy, search_k=7)

    assert spy.prefetch_calls == [([], 7)]


def test_run_agent_raises_when_a_question_was_not_prefetched() -> None:
    questions = [make_question(i) for i in range(2)]
    spy = SpySearcher()

    with pytest.raises(ValueError, match='question 1'):
        run_agent(
            questions, make_one_hop_graph(), spy,
            prefetch_questions=questions[:1],
        )

    assert spy.prefetch_calls == []
