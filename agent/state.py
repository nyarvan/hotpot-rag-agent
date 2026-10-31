"""Graph state: partial updates from nodes are merged by the reducers below."""
import operator
from typing import Annotated, TypedDict


class AgentState(TypedDict):
    """State of one agent run; graph input is just `{'question': q}`."""

    question: str
    # Queries sent to search, in order
    queries: Annotated[list[str], operator.add]
    # Ranked titles per hop
    rankings: Annotated[list[list[str]], operator.add]
    hops: Annotated[int, operator.add]  # == len(queries)
    follow_up: str  # '' means "enough"
    titles: list[str]  # final context titles
    contexts: list[str]  # format_context(...) of those titles
    answer: str
    llm_calls: Annotated[int, operator.add]  # decide + generate requests
