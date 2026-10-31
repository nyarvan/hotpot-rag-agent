"""Graph nodes and the protocols for their dependencies.

Nodes are built by factories so dependencies are fixed per run. Every node
returns a partial state update and never modifies the state it receives.
"""
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from rag_eval.data import Document
from rag_eval.generate import format_context

from .state import AgentState

logger = logging.getLogger(__name__)

CONTEXT_K = 5
FIRST_HOP_K = 3

Node = Callable[[AgentState], dict[str, Any]]


@dataclass(frozen=True)
class Decision:
    """Whether the context is enough; if not, what to search for next."""

    enough: bool
    follow_up: str = ''


class Searcher(Protocol):
    """Ranked titles for a query."""

    def search(self, query: str, k: int) -> Sequence[str]:
        """Return the top-k titles, best first."""
        ...


class Decider(Protocol):
    """Judges whether the current context answers the question."""

    def decide(self, question: str, contexts: Sequence[Document]) -> Decision:
        """Return a decision for the question given the context."""
        ...


class Answerer(Protocol):
    """Answers a question from a context (matches the harness `Generator`)."""

    def answer(self, question: str, contexts: Sequence[Document]) -> str:
        """Return the final answer."""
        ...


def merge_titles(
        rankings: Sequence[Sequence[str]],
        context_k: int = CONTEXT_K,
        first_hop_k: int = FIRST_HOP_K,
) -> list[str]:
    """Merge per-hop rankings into one context of at most `context_k` titles.

    One hop gives the plain top-`context_k`. With more hops: the top
    `first_hop_k` of hop 1, then new titles from later hops (at most
    `context_k - first_hop_k` in total, so with the defaults a third hop
    only matters when hop 2 added fewer than two), then the rest of hop 1
    fills the budget. Rankings are assumed to hold unique titles, as the
    retrievers return them.

    Raises `ValueError` unless `0 < first_hop_k <= context_k`.
    """
    if not 0 < first_hop_k <= context_k:
        raise ValueError(
            f'first_hop_k ({first_hop_k}) must be between 1 and '
            f'context_k ({context_k})'
        )
    if not rankings:
        return []
    first = rankings[0]
    if len(rankings) == 1:
        return list(first[:context_k])
    taken = list(dict.fromkeys(first[:first_hop_k]))
    new_budget = context_k - first_hop_k
    new_titles = 0
    for ranking in rankings[1:]:
        for title in ranking:
            if new_titles >= new_budget:
                break
            if title not in taken:
                taken.append(title)
                new_titles += 1
    for title in first[first_hop_k:]:
        if len(taken) >= context_k:
            break
        if title not in taken:
            taken.append(title)
    return taken


def _documents_for(
        rankings: Sequence[Sequence[str]],
        documents: Mapping[str, Document],
        context_k: int,
) -> tuple[list[str], list[Document]]:
    """Merged titles and their documents; unknown titles raise KeyError."""
    titles = merge_titles(rankings, context_k)
    return titles, [documents[title] for title in titles]


def make_retrieve(searcher: Searcher, search_k: int) -> Node:
    """Node: search the question on the first hop, the follow-up afterwards."""

    def retrieve(state: AgentState) -> dict[str, Any]:
        query = state['question'] if state['hops'] == 0 else state['follow_up']
        ranking = searcher.search(query, search_k)
        return {'queries': [query], 'rankings': [list(ranking)], 'hops': 1}

    return retrieve


def make_decide(
        decider: Decider, documents: Mapping[str, Document], context_k: int,
) -> Node:
    """Node: ask the decider whether the merged context is enough."""

    def decide(state: AgentState) -> dict[str, Any]:
        _, context = _documents_for(state['rankings'], documents, context_k)
        decision = decider.decide(state['question'], context)
        follow_up = '' if decision.enough else decision.follow_up.strip()
        if follow_up in state['queries']:
            # Searching it again would return the same ranking
            logger.info(
                'Follow-up repeats an earlier query, answering now: %r',
                follow_up,
            )
            follow_up = ''
        return {'follow_up': follow_up, 'llm_calls': 1}

    return decide


def make_generate(
        answerer: Answerer, documents: Mapping[str, Document], context_k: int,
) -> Node:
    """Node: answer from the merged context."""

    def generate(state: AgentState) -> dict[str, Any]:
        titles, context = _documents_for(
            state['rankings'], documents, context_k,
        )
        answer = answerer.answer(state['question'], context)
        return {
            'titles': titles,
            'contexts': [format_context(d) for d in context],
            'answer': answer,
            'llm_calls': 1,
        }

    return generate
