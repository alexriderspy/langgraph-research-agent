"""Wire the nodes into a LangGraph StateGraph."""
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent import nodes
from agent.state import ResearchState
from agent.tools import TOOLS


def build_graph():
    g = StateGraph(ResearchState)
    g.add_node("recall", nodes.recall)
    g.add_node("plan", nodes.plan)
    g.add_node("search", nodes.search)
    g.add_node("tools", ToolNode(TOOLS))
    g.add_node("read", nodes.read)
    g.add_node("take_notes", nodes.take_notes)
    g.add_node("reflect", nodes.reflect)
    g.add_node("write", nodes.write)
    g.add_node("remember", nodes.remember)

    g.add_edge(START, "recall")
    g.add_edge("recall", "plan")
    g.add_edge("plan", "search")
    g.add_conditional_edges("search", tools_condition, {"tools": "tools", END: "read"})
    g.add_edge("tools", "read")
    g.add_edge("read", "take_notes")
    g.add_edge("take_notes", "reflect")
    g.add_conditional_edges("reflect", nodes.should_continue, {"search": "search", "write": "write"})
    g.add_edge("write", "remember")
    g.add_edge("remember", END)
    return g.compile()
