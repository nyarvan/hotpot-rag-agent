"""Hop-limited retrieve / decide / generate graph.

With `human_review=True` an `approve` node sits between `decide` and the
second `retrieve` and pauses the run (`interrupt`) so a person can approve,
edit or reject the follow-up; this needs a `checkpointer` to resume from.
By default the graph has no such node and runs straight through.
"""
from collections.abc import Mapping

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from rag_eval.data import Document

from .nodes import (
    CONTEXT_K,
    FIRST_HOP_K,
    Answerer,
    Decider,
    Searcher,
    make_approve,
    make_decide,
    make_generate,
    make_retrieve,
)
from .state import AgentState

MAX_HOPS = 2
SEARCH_K = 10  # = harness RETRIEVE_K


def build_graph(
        searcher: Searcher, decider: Decider, answerer: Answerer,
        documents: Mapping[str, Document], *,
        max_hops: int = MAX_HOPS,
        search_k: int = SEARCH_K,
        context_k: int = CONTEXT_K,
        human_review: bool = False,
        checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Compile the agent graph; the decider is skipped after the last hop.

    `human_review` adds a pause before each follow-up search and requires a
    `checkpointer`; without one an interrupt would stop the run for good.
    """
    if max_hops < 1:
        raise ValueError(f'max_hops ({max_hops}) must be at least 1')
    if context_k < FIRST_HOP_K:
        raise ValueError(
            f'context_k ({context_k}) must be at least {FIRST_HOP_K}, '
            'the share of hop 1 in a merged context',
        )
    if context_k > search_k:
        raise ValueError(
            f'context_k ({context_k}) must not exceed search_k ({search_k})',
        )

    if human_review and checkpointer is None:
        raise ValueError('human_review requires a checkpointer to resume from')

    def after_retrieve(state: AgentState) -> str:
        return 'generate' if state['hops'] >= max_hops else 'decide'

    def after_decide(state: AgentState) -> str:
        if not state['follow_up']:
            return 'generate'
        return 'approve' if human_review else 'retrieve'

    def after_approve(state: AgentState) -> str:
        return 'retrieve' if state['follow_up'] else 'generate'

    builder = StateGraph(AgentState)
    builder.add_node('retrieve', make_retrieve(searcher, search_k))
    builder.add_node('decide', make_decide(decider, documents, context_k))
    if human_review:
        builder.add_node('approve', make_approve(context_k))
    builder.add_node('generate', make_generate(answerer, documents, context_k))
    builder.add_edge(START, 'retrieve')
    builder.add_conditional_edges(
        'retrieve', after_retrieve, ['decide', 'generate'],
    )
    if human_review:
        builder.add_conditional_edges(
            'decide', after_decide, ['approve', 'generate'],
        )
        builder.add_conditional_edges(
            'approve', after_approve, ['retrieve', 'generate'],
        )
    else:
        builder.add_conditional_edges(
            'decide', after_decide, ['retrieve', 'generate'],
        )
    builder.add_edge('generate', END)
    return builder.compile(checkpointer=checkpointer)
