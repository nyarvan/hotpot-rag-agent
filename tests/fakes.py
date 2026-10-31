"""Hand-written fakes for the agent graph: no network, no LLM."""
from collections.abc import Mapping, Sequence

from rag_eval.data import Document

from agent.nodes import Decision

TITLES = [f'A{i}' for i in range(1, 9)] + ['B1', 'B2', 'B3', 'B4']
DOCUMENTS: dict[str, Document] = {
    title: Document(
        doc_id=f'id-{title}', title=title, text=f'Text of {title}.',
    )
    for title in TITLES
}


class FakeSearcher:
    """Returns a fixed ranking per query and records every call."""

    def __init__(self, rankings_by_query: Mapping[str, Sequence[str]]) -> None:
        self._rankings_by_query = rankings_by_query
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, k: int) -> Sequence[str]:
        """Record the call and return the canned ranking."""
        self.calls.append((query, k))
        return self._rankings_by_query[query]


class FakeDecider:
    """Returns the decisions in order, then repeats the last one."""

    def __init__(self, decisions: Sequence[Decision]) -> None:
        self._decisions = decisions
        self.seen_contexts: list[list[str]] = []

    def decide(self, question: str, contexts: Sequence[Document]) -> Decision:
        """Record the context titles and return the next decision."""
        index = min(len(self.seen_contexts), len(self._decisions) - 1)
        self.seen_contexts.append([d.title for d in contexts])
        return self._decisions[index]


class RaisingDecider:
    """A decider that fails the test if it is ever called."""

    def decide(self, question: str, contexts: Sequence[Document]) -> Decision:
        """Always raise."""
        raise AssertionError('decide must not be called')


class FakeAnswerer:
    """Returns a fixed answer and records the documents it received."""

    def __init__(self, answer: str = 'fake answer') -> None:
        self._answer = answer
        self.received: list[list[Document]] = []

    def answer(self, question: str, contexts: Sequence[Document]) -> str:
        """Record the contexts and return the canned answer."""
        self.received.append(list(contexts))
        return self._answer
