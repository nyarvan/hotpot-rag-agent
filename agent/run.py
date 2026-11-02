"""Run the agent graph over questions and produce harness-compatible records.

`AgentRecord` extends the harness `RunRecord`, so `rag-eval eval` reads the
run files and ignores the extra fields.
"""
import logging
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from langgraph.graph.state import CompiledStateGraph
from pydantic import Field
from rag_eval.data import Question
from rag_eval.records import RunRecord

from .graph import SEARCH_K

PROGRESS_EVERY = 20
logger = logging.getLogger(__name__)


class AgentRecord(RunRecord):
    """A `RunRecord` plus how the agent got there."""

    hops: int = Field(ge=0)
    queries: tuple[str, ...]
    llm_calls: int = Field(ge=0)


class Prefetcher(Protocol):
    """Ranks all first-hop questions in one batch before the graph runs."""

    def prefetch(self, questions: Sequence[Question], k: int) -> None:
        """Rank every question to depth `k`."""
        ...


def to_record(question: Question, state: Mapping[str, Any]) -> AgentRecord:
    """Turn a final graph state into a record.

    `retrieved` lists the final context first, then the rest of the hop-1
    ranking, so all_gold@5 means "both gold paragraphs are in the context".
    """
    titles = list(state['titles'])
    rest = [t for t in state['rankings'][0] if t not in titles]
    return AgentRecord(
        question_id=question.question_id,
        question=question.question,
        contexts=tuple(state['contexts']),
        answer=state['answer'],
        reference=question.answer,
        retrieved=tuple(titles + rest),
        gold_titles=question.gold_titles,
        hops=state['hops'],
        queries=tuple(state['queries']),
        llm_calls=state['llm_calls'],
    )


def run_agent(
        questions: Sequence[Question], graph: CompiledStateGraph,
        searcher: Prefetcher, *,
        prefetch_questions: Sequence[Question] | None = None,
        search_k: int = SEARCH_K,
) -> list[AgentRecord]:
    """Prefetch hop 1 in one batch, then run the graph per question.

    `prefetch_questions` (default: `questions`) should be the full set the
    baseline encoded together, because batch composition can reorder
    near-ties in the dense ranking. Every run question must be among them.
    """
    to_prefetch = questions if prefetch_questions is None else (
        prefetch_questions
    )
    known = {question.question for question in to_prefetch}
    for question in questions:
        if question.question not in known:
            raise ValueError(
                f'Question not in prefetch set: {question.question!r}'
            )
    searcher.prefetch(to_prefetch, search_k)
    records = []
    for done, question in enumerate(questions, start=1):
        state = graph.invoke({'question': question.question})
        records.append(to_record(question, state))
        if done % PROGRESS_EVERY == 0:
            logger.info('%d/%d questions', done, len(questions))
    return records
