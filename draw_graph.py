"""Draw the agent's graph: ASCII in the terminal, Mermaid text for the README, and a PNG."""
from pathlib import Path

from agent.graph import build_graph

graph = build_graph().get_graph()
print(graph.draw_ascii())
Path("media").mkdir(exist_ok=True)
Path("media/graph.mmd").write_text(graph.draw_mermaid())
Path("media/graph.png").write_bytes(graph.draw_mermaid_png())   # rendered by the mermaid.ink web API
print("wrote media/graph.mmd and media/graph.png")
