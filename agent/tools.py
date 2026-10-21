"""Search tools the agent can call. All three are public APIs that need no key."""
import re
import time
import xml.etree.ElementTree as ET
from typing import TypedDict

import httpx
from langchain_core.tools import tool

from agent.config import settings

HTTP = httpx.Client(headers={"User-Agent": settings.user_agent}, timeout=20, follow_redirects=True)
ATOM = {"a": "http://www.w3.org/2005/Atom"}
STOPWORDS = {"the", "and", "for", "with", "how", "what", "are", "does", "from", "using", "used", "into", "recent"}


class Hit(TypedDict):
    title: str
    url: str
    snippet: str
    kind: str


def keywords(query: str, n: int = 4) -> list[str]:
    """Small models write long queries; search engines like a few keywords."""
    words = re.findall(r"[a-z0-9-]+", query.lower())
    return [w for w in words if len(w) > 2 and w not in STOPWORDS][:n]


def get(url: str, params: dict) -> httpx.Response | None:
    """One GET; a network error or bad status becomes None, so a flaky API can't crash the run."""
    try:
        r = HTTP.get(url, params=params)
        return r if r.status_code == 200 else None
    except httpx.HTTPError:
        return None


def _summary(hits: list[Hit]) -> str:
    return "\n".join(f"- {h['title']} ({h['url']})" for h in hits) or "no results"


@tool(response_format="content_and_artifact")
def search_arxiv(query: str) -> tuple[str, list[Hit]]:
    """Search arXiv for research papers. Best for methods, benchmarks and recent results."""
    params = {"search_query": " AND ".join(f"all:{w}" for w in keywords(query)),  # every word must appear
              "max_results": settings.results_per_search}
    for wait in (3, 10, 20):   # arXiv asks for one request every 3 seconds, and says 429 if we rush
        time.sleep(wait)
        if r := get("https://export.arxiv.org/api/query", params):
            break
    else:
        return "arXiv unavailable right now", []
    hits: list[Hit] = []
    for e in ET.fromstring(r.text).findall("a:entry", ATOM):
        hits.append(Hit(title=" ".join(e.findtext("a:title", "", ATOM).split()),
                        url=e.findtext("a:id", "", ATOM).replace("http://", "https://"),
                        snippet=" ".join(e.findtext("a:summary", "", ATOM).split()), kind="arxiv"))
    return _summary(hits), hits


@tool(response_format="content_and_artifact")
def search_wikipedia(query: str) -> tuple[str, list[Hit]]:
    """Search Wikipedia. Best for definitions, background and how something works."""
    r = get("https://en.wikipedia.org/w/api.php", {
        "action": "query", "list": "search", "srsearch": " ".join(keywords(query)),
        "format": "json", "srlimit": settings.results_per_search})
    if r is None:
        return "Wikipedia unavailable right now", []
    hits = [Hit(title=s["title"], url="https://en.wikipedia.org/wiki/" + s["title"].replace(" ", "_"),
                snippet=re.sub("<[^>]+>", "", s["snippet"]), kind="wikipedia")
            for s in r.json()["query"]["search"]]
    return _summary(hits), hits


@tool(response_format="content_and_artifact")
def search_hackernews(query: str) -> tuple[str, list[Hit]]:
    """Search Hacker News stories. Best for practitioner blog posts, tools and real-world experience."""
    r = get("https://hn.algolia.com/api/v1/search", {
        "query": " ".join(keywords(query, 3)), "tags": "story", "hitsPerPage": settings.results_per_search})
    if r is None:
        return "Hacker News unavailable right now", []
    hits = [Hit(title=h["title"], url=h["url"], snippet="", kind="web")
            for h in r.json()["hits"] if h.get("url")]
    return _summary(hits), hits


TOOLS = [search_arxiv, search_wikipedia, search_hackernews]
