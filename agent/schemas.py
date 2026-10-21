"""Shapes we ask the LLM to answer in. The model's reply is checked against these."""
from pydantic import BaseModel, Field


class Plan(BaseModel):
    sub_questions: list[str] = Field(description="3 or 4 specific questions that together answer the main one")


class NoteOut(BaseModel):
    claim: str = Field(description="one fact from the passage that helps answer the question, in your own words")
    quote: str = Field(description="the sentence from the passage that supports the claim, copied exactly")


class Notes(BaseModel):
    notes: list[NoteOut] = Field(description="0 to 3 notes; empty if the passage is not relevant")


class Reflection(BaseModel):
    missing: list[str] = Field(description="sub-questions the notes do not answer yet")
    follow_up_queries: list[str] = Field(description="short search queries to fill the gaps, at most 3")
    enough: bool = Field(description="true if the notes are enough to write a good brief")


class Sentence(BaseModel):
    text: str = Field(description="one sentence, using only facts from the notes")
    sources: list[int] = Field(description="the source numbers of the notes this sentence came from")


class Section(BaseModel):
    heading: str
    sentences: list[Sentence]


class Brief(BaseModel):
    title: str
    summary: list[Sentence] = Field(description="2 sentences that answer the question directly")
    sections: list[Section] = Field(description="2 to 4 sections")
