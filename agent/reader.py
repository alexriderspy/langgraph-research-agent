"""Fetch a source, clean it to plain text and keep only the parts relevant to the question."""
import re

import trafilatura

from agent.embeddings import embed
from agent.tools import HTTP, Hit


def fetch_text(hit: Hit) -> str:
    """Plain text for one search hit, or "" if the page can't be read."""
    try:
        if hit["kind"] == "arxiv":
            return hit["snippet"]                                   # the abstract came with the search
        if hit["kind"] == "wikipedia":
            r = HTTP.get("https://en.wikipedia.org/w/api.php", params={
                "action": "query", "prop": "extracts", "explaintext": 1, "format": "json",
                "titles": hit["title"], "redirects": 1})
            page = next(iter(r.json()["query"]["pages"].values()))
            return page.get("extract", "")
        r = HTTP.get(hit["url"])
        return trafilatura.extract(r.text, include_comments=False, include_tables=False) or ""
    except Exception:
        return ""


def chunk(text: str, size: int = 700) -> list[str]:
    """Split on paragraphs, then pack paragraphs into chunks of about `size` characters."""
    paras = [p.strip()[:1500] for p in re.split(r"\n\s*\n|\n(?==)", text) if len(p.strip()) > 40]
    chunks, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) > size:
            chunks.append(cur)
            cur = ""
        cur = f"{cur}\n{p}".strip()
    return chunks + ([cur] if cur else [])


def relevant_passages(text: str, question: str, k: int = 3) -> list[str]:
    """The k chunks closest in meaning to the question."""
    chunks = chunk(text)
    if len(chunks) <= k:
        return chunks
    scores = embed(chunks) @ embed([question])[0]
    best = sorted(scores.argsort()[::-1][:k])                 # top k, kept in page order
    return [chunks[i] for i in best]
