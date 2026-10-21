"""Ask the research agent a question: python run.py "your question" """
import json
import re
import sys
import time
from pathlib import Path

from agent.graph import build_graph


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60]


def main(question: str) -> None:
    graph = build_graph()
    state: dict = {}
    t0 = time.time()
    for mode, chunk in graph.stream({"question": question}, {"recursion_limit": 40},
                                    stream_mode=["updates", "values"]):
        if mode == "values":
            state = chunk                      # the full state after each step
            continue
        for node, changes in chunk.items():
            for line in changes.get("trace", [f"{node}: ran {len(changes.get('messages', []))} tool calls"]):
                print(f"[{time.time() - t0:6.1f}s] {line}")
    out = Path("briefs") / f"{slug(question)}.md"
    out.write_text(state["brief"])
    Path("runs").mkdir(exist_ok=True)
    Path("runs", f"{slug(question)}.json").write_text(json.dumps(
        {"question": question, "seconds": round(time.time() - t0, 1), "citations": state["citations"]}, indent=2))
    print(f"\nbrief saved to {out}\n")
    print(state["brief"])


if __name__ == "__main__":
    main(" ".join(sys.argv[1:]))
