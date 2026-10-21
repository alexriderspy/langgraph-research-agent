"""The chat model. Swap providers by changing one env var: LLM_MODEL."""
from functools import lru_cache

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from agent.config import settings


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    provider, _, model = settings.llm_model.partition(":")
    return init_chat_model(model, model_provider=provider, temperature=0)
