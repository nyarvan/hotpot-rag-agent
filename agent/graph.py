"""Hop-limited retrieve / decide / generate graph.

Day 3 will add a `checkpointer` keyword to `build_graph`.
"""
from collections.abc import Mapping

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from rag_eval.data import Document

from .nodes import (
    CONTEXT_K,
    FIRST_HOP_K,
    Answerer,
    Decider,
    Searcher,
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
) -> CompiledStateGraph:
    """Compile the agent graph; the decider is skipped after the last hop."""
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

    def after_retrieve(state: AgentState) -> str:
        return 'generate' if state['hops'] >= max_hops else 'decide'

    def after_decide(state: AgentState) -> str:
        return 'retrieve' if state['follow_up'] else 'generate'

    builder = StateGraph(AgentState)
    builder.add_node('retrieve', make_retrieve(searcher, search_k))
    builder.add_node('decide', make_decide(decider, documents, context_k))
    builder.add_node('generate', make_generate(answerer, documents, context_k))
    builder.add_edge(START, 'retrieve')
    builder.add_conditional_edges(
        'retrieve', after_retrieve, ['decide', 'generate'],
    )
    builder.add_conditional_edges(
        'decide', after_decide, ['retrieve', 'generate'],
    )
    builder.add_edge('generate', END)
    return builder.compile()
