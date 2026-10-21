"""An independent check: re-read each cited page and ask the LLM if it supports the sentence."""
from pydantic import BaseModel

from agent.llm import get_llm
from agent.reader import fetch_text, relevant_passages
from agent.tools import Hit

JUDGE = """Passage from a source:
\"\"\"{passage}\"\"\"

Claim: {claim}

Does the passage support the claim? Answer supported=true only if the passage says it."""


class Verdict(BaseModel):
    supported: bool


def page_text(url: str, title: str) -> str:
    """Re-fetch the page fresh, the same way the agent does, so we judge what's really there."""
    kind = "wikipedia" if "wikipedia.org" in url else "web"   # arXiv: download the abstract page itself
    return fetch_text(Hit(title=title, url=url, snippet="", kind=kind))


def supported(claim: str, text: str) -> bool:
    passage = "\n\n".join(relevant_passages(text, claim, k=2))
    return get_llm().with_structured_output(Verdict).invoke(
        JUDGE.format(passage=passage, claim=claim)).supported
