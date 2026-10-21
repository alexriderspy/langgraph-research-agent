"""Number the sources, check every sentence of the brief against them, and render Markdown."""
import re

from agent.embeddings import embed
from agent.schemas import Brief, Sentence
from agent.state import Note

MIN_SUPPORT = 0.60   # cosine similarity between a sentence and the notes it cites
CITE = re.compile(r"\s*\[\d+\]")


def number_sources(notes: list[Note]) -> list[str]:
    """Unique source URLs in the order the notes found them; source n is urls[n - 1]."""
    return list(dict.fromkeys(n["url"] for n in notes))


def notes_for_prompt(notes: list[Note], urls: list[str]) -> str:
    return "\n".join(f"[{urls.index(n['url']) + 1}] {n['claim']}" for n in notes)


def all_sentences(brief: Brief) -> list[Sentence]:
    return brief.summary + [s for sec in brief.sections for s in sec.sentences]


def check(brief: Brief, notes: list[Note], urls: list[str]) -> dict:
    """Label each sentence: ok, weak (its sources don't seem to say it), uncited or bad_number."""
    rows = []
    for s in all_sentences(brief):
        cited = sorted(set(s.sources))
        if not cited:
            status, support = "uncited", 0.0
        elif any(c < 1 or c > len(urls) for c in cited):
            status, support = "bad_number", 0.0
        else:
            evidence = [f"{n['claim']} {n['quote']}" for n in notes if urls.index(n["url"]) + 1 in cited]
            support = float((embed(evidence) @ embed([s.text])[0]).max())
            status = "ok" if support >= MIN_SUPPORT else "weak"
        rows.append({"sentence": s.text, "cited": cited, "status": status, "support": round(support, 3)})
    counts = {k: sum(r["status"] == k for r in rows) for k in ("ok", "weak", "uncited", "bad_number")}
    return {"sentences": rows, "counts": counts}


def render(brief: Brief, report: dict, notes: list[Note], urls: list[str]) -> str:
    """Markdown with only the supported sentences, sources renumbered in order of first use."""
    ok = {r["sentence"] for r in report["sentences"] if r["status"] == "ok"}
    in_sections = {s.text for sec in brief.sections for s in sec.sentences}
    order: list[int] = []

    def para(sentences: list[Sentence]) -> str:
        out = []
        for s in sentences:
            if s.text in ok:
                cites = sorted(set(s.sources))
                order.extend(c for c in cites if c not in order)
                tags = "".join(f"[{order.index(c) + 1}]" for c in cites)
                text = CITE.sub("", s.text).strip().rstrip(".")   # models sometimes type "[1]" themselves
                out.append(f"{text} {tags}.")
        return " ".join(out)

    # small models like to repeat section sentences in the summary: keep them in their section only
    parts = [f"# {brief.title}", para([s for s in brief.summary if s.text not in in_sections])]
    for sec in brief.sections:
        body = para(sec.sentences)
        if body:
            parts += [f"## {sec.heading}", body]
    titles = {n["url"]: n["title"] for n in notes}
    refs = [f"{i}. [{titles[urls[c - 1]]}]({urls[c - 1]})" for i, c in enumerate(order, 1)]
    return "\n\n".join(p for p in parts if p) + "\n\n## Sources\n\n" + "\n".join(refs) + "\n"
