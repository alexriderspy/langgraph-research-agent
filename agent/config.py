"""Every setting in one place, read from the environment (or a .env file)."""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    llm_model: str = os.getenv("LLM_MODEL", "ollama:qwen2.5:7b")  # e.g. "anthropic:claude-sonnet-5-5"
    embed_model: str = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    memory_db: str = os.getenv("MEMORY_DB", "memory.db")
    max_rounds: int = int(os.getenv("MAX_ROUNDS", "2"))         # research loops before we must write
    results_per_search: int = int(os.getenv("RESULTS_PER_SEARCH", "3"))
    sources_per_round: int = int(os.getenv("SOURCES_PER_ROUND", "4"))
    recall_threshold: float = float(os.getenv("RECALL_THRESHOLD", "0.65"))
    user_agent: str = "langgraph-research-agent/0.1 (learning project)"


settings = Settings()
