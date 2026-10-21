"""What the graph carries from node to node."""
import operator
from typing import Annotated, NotRequired, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class Note(TypedDict):
    claim: str            # one fact, in our words
    quote: str            # the exact sentence in the source that supports it
    url: str
    title: str
    from_memory: NotRequired[bool]


class Page(TypedDict):
    url: str
    title: str
    kind: str
    text: str             # only the passages relevant to the question


class ResearchState(TypedDict):
    question: str
    sub_questions: list[str]
    queries: list[str]                                   # what to search for this round
    messages: Annotated[list[AnyMessage], add_messages]  # tool calls and tool results
    pages: list[Page]
    seen_urls: Annotated[list[str], operator.add]
    searched: Annotated[list[str], operator.add]        # every query a tool has already run
    notes: Annotated[list[Note], operator.add]
    rounds: int
    enough: bool                                         # reflect's verdict
    brief: str
    citations: dict
    trace: Annotated[list[str], operator.add]
