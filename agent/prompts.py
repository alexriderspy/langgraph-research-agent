"""Every prompt the agent uses, kept together so they're easy to read and change."""

PLAN = """You are planning a research brief.
Main question: {question}
{known}
Break the main question into 3 or 4 specific sub-questions that a researcher would search for.
Skip anything the known notes already answer well."""

SEARCH = """You are a research assistant with three search tools.
Call tools to find sources for each of these questions. Use 1 or 2 tool calls per question.
Write each query as 2 to 4 keywords, like "pump cavitation detection", not a full sentence.
Pick arXiv for research papers, Wikipedia for background and Hacker News for practitioner posts.

Questions:
{queries}"""

NOTES = """Main question: {question}

Source: {title}
Passage:
\"\"\"{text}\"\"\"

Take up to 3 notes from this passage that help answer the main question.
A note is a finding, method or number, never just a title.
For each note, copy the supporting sentence from the passage exactly into "quote".
If the passage is not relevant, return no notes."""

REFLECT = """Main question: {question}
Sub-questions:
{sub_questions}

Notes so far:
{notes}

Which sub-questions are still not answered by the notes? Suggest short search queries for the gaps.
Set enough to true only if every sub-question has at least one supporting note."""

WRITE = """Write a research brief that answers: {question}

Use ONLY these notes. Each note starts with its source number in square brackets.
{notes}

Every sentence lists the source numbers it came from. Be specific: keep the names, numbers
and methods from the notes, and leave out general filler. Don't add facts that aren't in the notes."""
