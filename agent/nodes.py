"""One function per node. Each takes the state and returns only the keys it changes."""
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from rapidfuzz import fuzz

from agent import citations, prompts
from agent.config import settings
from agent.llm import get_llm
from agent.memory import Memory
from agent.reader import fetch_text, relevant_passages
from agent.schemas import Brief, Notes, Plan, Reflection
from agent.state import Note, Page, ResearchState
from agent.tools import TOOLS, keywords, search_wikipedia


def recall(state: ResearchState) -> dict:
    notes = Memory().recall(state["question"])
    return {"notes": notes, "rounds": 0, "trace": [f"recall: {len(notes)} notes from past research"]}


def plan(state: ResearchState) -> dict:
    known = "\n".join(f"- {n['claim']}" for n in state["notes"])
    known = f"Known notes from past research:\n{known}" if known else "No past research on this."
    out = get_llm().with_structured_output(Plan).invoke(
        prompts.PLAN.format(question=state["question"], known=known))
    subs = out.sub_questions[:4]
    return {"sub_questions": subs, "queries": subs, "trace": [f"plan: {len(subs)} sub-questions"]}


def search(state: ResearchState) -> dict:
    """The LLM picks which tools to call, and with what queries."""
    queries = "\n".join(f"- {q}" for q in state["queries"])
    ai = get_llm().bind_tools(TOOLS).invoke([HumanMessage(prompts.SEARCH.format(queries=queries))])
    calls = ", ".join(f"{c['name']}({c['args'].get('query', '')!r})" for c in ai.tool_calls)
    done = [" ".join(keywords(c["args"].get("query", ""))) for c in ai.tool_calls]
    return {"messages": [ai], "searched": done, "trace": [f"search: {len(ai.tool_calls)} tool calls: {calls}"]}


def latest_hits(state: ResearchState) -> list[dict]:
    """This round's hits, taken in turns from each tool so one tool can't crowd out the others."""
    results = []
    for m in reversed(state["messages"]):
        if isinstance(m, AIMessage):
            break
        if isinstance(m, ToolMessage) and m.artifact:
            results.insert(0, m.artifact)
    return [hits[i] for i in range(max(map(len, results), default=0)) for hits in results if i < len(hits)]


def read(state: ResearchState) -> dict:
    hits = latest_hits(state)
    if not hits:   # the model made no tool calls: fall back to Wikipedia for each query
        calls = [{"name": "search_wikipedia", "args": {"query": q}, "id": str(i), "type": "tool_call"}
                 for i, q in enumerate(state["queries"])]
        hits = [h for c in calls for h in search_wikipedia.invoke(c).artifact]
    seen = set(state["seen_urls"])
    new = list({h["url"]: h for h in hits if h["url"] not in seen}.values())[: settings.sources_per_round]
    pages = []
    for h in new:
        text = fetch_text(h)
        if text:
            passages = relevant_passages(text, state["question"] + " " + " ".join(state["queries"]))
            pages.append(Page(url=h["url"], title=h["title"], kind=h["kind"], text="\n\n".join(passages)))
    return {"pages": pages, "seen_urls": [h["url"] for h in new],
            "trace": [f"read: {len(pages)} of {len(new)} new sources readable"]
                     + [f"   {p['kind']:9} {p['title'][:80]}" for p in pages]}


def quote_found(quote: str, text: str) -> bool:
    """The quote must really be in the page (allowing small whitespace or punctuation changes)."""
    return len(quote) > 20 and fuzz.partial_ratio(quote.lower(), text.lower()) >= 90


def take_notes(state: ResearchState) -> dict:
    llm = get_llm().with_structured_output(Notes)
    kept, dropped, per_page = [], 0, []
    for p in state["pages"]:
        out = llm.invoke(prompts.NOTES.format(question=state["question"], title=p["title"], text=p["text"]))
        per_page.append(f"   {len(out.notes)} notes from {p['title'][:70]}")
        for n in out.notes:
            if quote_found(n.quote, p["text"]):
                kept.append(Note(claim=n.claim, quote=n.quote, url=p["url"], title=p["title"]))
            else:
                dropped += 1
    return {"notes": kept, "trace": per_page + [f"notes: kept {len(kept)}, "
                                                f"dropped {dropped} whose quote wasn't in the source"]}


def reflect(state: ResearchState) -> dict:
    subs = "\n".join(f"- {q}" for q in state["sub_questions"])
    notes = "\n".join(f"- {n['claim']}" for n in state["notes"]) or "none"
    out = get_llm().with_structured_output(Reflection).invoke(
        prompts.REFLECT.format(question=state["question"], sub_questions=subs, notes=notes))
    rounds = state["rounds"] + 1
    new = [q for q in out.follow_up_queries or out.missing if " ".join(keywords(q)) not in state["searched"]]
    return {"rounds": rounds, "queries": new[:3],
            "enough": out.enough,
            "trace": [f"reflect: round {rounds}, enough={out.enough}, missing {len(out.missing)}"]}


def should_continue(state: ResearchState) -> str:
    """Loop back to search, or stop and write. The round limit stops it looping forever."""
    if state["enough"] or state["rounds"] >= settings.max_rounds or not state["queries"]:
        return "write"
    return "search"


def write(state: ResearchState) -> dict:
    if not state["notes"]:
        return {"brief": f"# {state['question']}\n\nNo sources with usable evidence were found.\n",
                "citations": {"sentences": [], "counts": {}}, "trace": ["write: no notes, nothing to write"]}
    urls = citations.number_sources(state["notes"])
    draft = get_llm().with_structured_output(Brief).invoke(prompts.WRITE.format(
        question=state["question"], notes=citations.notes_for_prompt(state["notes"], urls)))
    report = citations.check(draft, state["notes"], urls)
    brief = citations.render(draft, report, state["notes"], urls)
    c = report["counts"]
    return {"brief": brief, "citations": report,
            "trace": [f"write: {sum(c.values())} sentences, {c['ok']} supported, {c['weak']} weak, "
                      f"{c['weak'] + c['uncited'] + c['bad_number']} removed"]}


def remember(state: ResearchState) -> dict:
    new = [n for n in state["notes"] if not n.get("from_memory")]
    saved = Memory().save(state["question"], new)
    return {"trace": [f"remember: saved {saved} new notes"]}
