"""Call each search tool by hand and read one page, before any LLM is involved."""
from agent.reader import fetch_text, relevant_passages
from agent.tools import search_arxiv, search_hackernews, search_wikipedia

query = "centrifugal pump fault"
for t in (search_arxiv, search_wikipedia, search_hackernews):
    msg = t.invoke({"name": t.name, "args": {"query": query}, "id": "1", "type": "tool_call"})
    print(f"\n== {t.name}: {len(msg.artifact)} hits ==")
    print(msg.content)

page = {"title": "Centrifugal pump", "url": "https://en.wikipedia.org/wiki/Centrifugal_pump",
        "snippet": "", "kind": "wikipedia"}
text = fetch_text(page)
print(f"\n== read {page['title']}: {len(text):,} characters ==")
for p in relevant_passages(text, "what causes cavitation in a centrifugal pump"):
    print("\n---\n" + p[:400])
