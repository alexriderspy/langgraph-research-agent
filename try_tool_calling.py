"""Does qwen2.5:7b really call tools through LangChain? Ask it once and look at the reply."""
import time

from langchain_core.messages import HumanMessage

from agent.llm import get_llm
from agent.tools import TOOLS

llm = get_llm().bind_tools(TOOLS)
t0 = time.time()
reply = llm.invoke([HumanMessage(
    "Find sources on: 1) what causes cavitation in centrifugal pumps, "
    "2) recent machine learning papers on pump fault detection.")])
print(f"answered in {time.time() - t0:.1f}s")
print("text content:", repr(reply.content))
for call in reply.tool_calls:
    print("tool call:", call["name"], call["args"])
