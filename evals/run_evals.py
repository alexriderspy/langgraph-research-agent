"""Run the agent on every question in evals/questions.jsonl and score the briefs.

citation validity: the share of cited sentences whose source page, fetched again, supports them
coverage:          the share of must-cover topics the brief mentions
steps:             research rounds, tool calls and sources read, plus wall-clock time
"""
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ["MEMORY_DB"] = "evals/eval_memory.db"    # its own memory, wiped per question
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.graph import build_graph  # noqa: E402
from evals.judge import page_text, supported  # noqa: E402

LINK = re.compile(r"^\d+\. \[(.*)\]\((.*)\)$", re.M)


def cited_sentences(brief: str) -> list[str]:
    """The brief's sentences; render() ends every one with its citations, like "... [1][3]." """
    body = [ln for ln in brief.split("## Sources")[0].splitlines() if ln and not ln.startswith("#")]
    return [s for ln in body for s in re.split(r"(?<=\]\.)\s+", ln.strip()) if s]


def score(item: dict) -> dict:
    Path("evals/eval_memory.db").unlink(missing_ok=True)
    t0 = time.time()
    state = build_graph().invoke({"question": item["question"]}, {"recursion_limit": 40})
    seconds = time.time() - t0
    brief = state["brief"]
    sources = LINK.findall(brief)                       # [(title, url)], numbered from 1
    texts = {url: page_text(url, title) for title, url in sources}
    checks = []
    for sentence in cited_sentences(brief):
        nums = [int(n) for n in re.findall(r"\[(\d+)\]", sentence)]
        claim = re.sub(r"\s*\[\d+\]", "", sentence).strip()
        ok = any(supported(claim, texts[sources[n - 1][1]]) for n in nums if texts[sources[n - 1][1]])
        checks.append(ok)
    covered = [any(alt in brief.lower() for alt in t.split("|")) for t in item["must_cover"]]
    calls = sum(len(getattr(m, "tool_calls", []) or []) for m in state["messages"])
    return {"question": item["question"], "sentences": len(checks), "supported": sum(checks),
            "coverage": sum(covered) / len(covered), "rounds": state["rounds"], "tool_calls": calls,
            "sources_read": len(state["seen_urls"]), "notes": len(state["notes"]), "seconds": round(seconds)}


def main() -> None:
    items = [json.loads(line) for line in Path("evals/questions.jsonl").read_text().splitlines()]
    rows = []
    for item in items:
        r = score(item)
        rows.append(r)
        print(f"{r['question'][:58]:58} cited {r['supported']:2}/{r['sentences']:<2} "
              f"cover {r['coverage']:.0%}  rounds {r['rounds']}  tools {r['tool_calls']:2}  "
              f"read {r['sources_read']:2}  {r['seconds']}s", flush=True)
    n_sent = sum(r["sentences"] for r in rows)
    print(f"\ncitation validity: {sum(r['supported'] for r in rows)}/{n_sent} "
          f"= {sum(r['supported'] for r in rows) / max(n_sent, 1):.0%} of cited sentences supported")
    print(f"coverage:          {sum(r['coverage'] for r in rows) / len(rows):.0%} of must-cover topics")
    print(f"average:           {sum(r['rounds'] for r in rows) / len(rows):.1f} rounds, "
          f"{sum(r['tool_calls'] for r in rows) / len(rows):.1f} tool calls, "
          f"{sum(r['seconds'] for r in rows) / len(rows):.0f}s per question")
    Path("evals/results.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
